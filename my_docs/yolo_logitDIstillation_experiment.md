# YOLO Logits Distillation — Experiment Design

## Goal

Systematically measure whether logits-based Knowledge Distillation (KD) improves a pruned YOLOv8-cls student relative to CE-only fine-tuning, and identify which hyperparameter settings produce the best recovery of teacher accuracy. The experiment is run across three datasets — CIFAR10, CIFAR100, and Tiny ImageNet — using the same teacher/student architecture pair (P3) to test whether the KD benefit is consistent across classification complexity.

The experiment is grounded in Hinton, Vinyals & Dean (2015) *"Distilling the Knowledge in a Neural Network"* (the original KD paper). Key theoretical commitments taken from that paper:

- **Soft targets carry "dark knowledge"**: near-zero class probabilities encode similarity structure the teacher learned. Higher temperature T spreads this signal across more classes.
- **T² gradient re-scaling**: the KL-div soft loss is multiplied by T² so the gradient magnitude stays constant regardless of T (see `soft_logit_kl_loss` in `losses.py`).
- **Hard-label CE always uses T=1**: the cross-entropy term always receives raw logits; only the KL-div term is temperature-scaled.
- **Pure soft-target regime (α=1.0)**: Hinton et al. show that for large enough datasets the student can learn effectively from soft targets alone.

---

## Baseline Setup

Common across all pairs:

| Item | Value |
|------|-------|
| Metric | Top-1 accuracy on full val set |
| Batch size | 128 |
| Seed | 42 (fixed for all runs) |

Per-pair dataset details:

| Pair | Dataset | Classes | Train size | Val size | Image size |
|------|---------|---------|------------|----------|------------|
| P1 | Tiny ImageNet | 200 | 100 000 | 10 000 | 64 × 64 |
| P2 | CIFAR100 | 100 | 50 000 | 10 000 | 32 × 32 |
| P3 | CIFAR10 | 10 | 50 000 | 10 000 | 32 × 32 |

---

## Model Pairs

Three pairs are evaluated, all using the same teacher/student architecture. The only difference across pairs is the dataset (and the corresponding teacher fine-tuning target).

| Pair ID | Dataset | Teacher | Student seed | Capacity gap | Rationale |
|---------|---------|---------|--------------|--------------|----------|
| P1 | Tiny ImageNet (200 classes) | `yolov8x-cls` (ImageNet pretrained, Tiny ImageNet fine-tuned) | `yolov8m-cls` (ImageNet pretrained only) | Large | Tests KD at higher classification complexity (200 classes, 64×64 inputs) |
| P2 | CIFAR100 (100 classes) | `yolov8x-cls` (ImageNet pretrained, CIFAR100 fine-tuned) | `yolov8m-cls` (ImageNet pretrained only) | Large | Tests KD at intermediate complexity (100 classes, 32×32 inputs) |
| P3 | CIFAR10 (10 classes) | `yolov8x-cls` (ImageNet pretrained, CIFAR10 fine-tuned) | `yolov8m-cls` (ImageNet pretrained only) | Large | Tests KD at lowest complexity; soft targets least informative (fewest classes) |

In all pairs the student seed is the off-the-shelf ImageNet-pretrained `yolov8m-cls` (no dataset-specific fine-tuning), pruned (unstructured magnitude pruning, `sparsity=0.50`) via MaseGraph before training begins. Training starts directly from the pruned ImageNet weights.

> **Note:** YOLOv8 `-cls` models are pretrained on ImageNet-1k. The detection variants (`yolov8n`, `yolov8x`, etc.) are pretrained on COCO — these are different model families.

---

## Variables

### Fixed across all runs
- Pruning sparsity: `0.50`
- Epochs: `50`
- Optimizer: AdamW, `weight_decay=0.05`
- LR: `5e-4`

### Primary sweep: KD hyperparameters

| Variable | Values | Rationale (from original KD paper) |
|----------|--------|------------------------------------|
| `alpha` (soft-loss weight) | 0.3, 0.5, 0.7, **0.9, 1.0** | The paper recommends testing the pure soft-target limit (α=1.0). High α lets the model learn entirely from dark knowledge without hard-label interference. |
| `temperature` | **1.0**, 2.0, 4.0, 8.0, **16.0** | T=1.0 is a degenerate baseline (nearly hard teacher argmax); T=16 surfaces inter-class similarity at finer granularity. For 10-class CIFAR10, the sweet spot is expected between T=4 and T=8. |

> **Note:** α=1.0 runs skip the CE term entirely. α=0.0 (pure CE) is the fine-tuned baseline and is not part of this grid — it is run separately as `exp_00`.

Full grid: 5 × 5 = **25 KD configurations + 1 CE baseline = 26 runs per pair × 3 pairs = 78 runs total**.

---

## Conditions per Run

For each `(model pair, alpha, temperature)` combination, four models are evaluated:

| Condition | Description |
|-----------|-------------|
| **Teacher** | CIFAR10 fine-tuned teacher, untouched |
| **Pruned (no train)** | Student immediately after pruning, no further training |
| **Fine-tuned (CE only)** | Pruned student trained with CE loss, no teacher |
| **Distilled (KD)** | Pruned student trained with combined CE + KL-div loss |

---

## Training Protocol

```python
distiller = YOLOLogitsDistiller(
    teacher          = teacher_cls_model,   # frozen; forward in train() mode inside no_grad → raw logits
    student          = student_cls_model,   # pruned student
    kd_config        = DistillationLossConfig(alpha=ALPHA, temperature=TEMP),
    device           = "cuda",
    train_loader     = train_loader,
    optimizer        = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    num_train_epochs = EPOCHS,
    val_loader       = val_loader,
    eval_teacher     = True,
)
train_history = distiller.train(save_path=KD_SAVE_PATH)   # saves best epoch checkpoint
# Restore best weights before evaluation (strict=False — pruning masks held in model, not state_dict)
student_cls_model.load_state_dict(torch.load(KD_SAVE_PATH), strict=False)
metrics = distiller.evaluate()
```

**Best-epoch checkpointing** (`save_path` parameter, added Mar 20 2026): `train()` evaluates val top-1 accuracy at the end of every epoch and saves `student.state_dict()` to `save_path` whenever the score improves. After training the caller restores the best weights via `load_state_dict(..., strict=False)` — `strict=False` is required because MASE's pruning pass registers sparsity masks as non-persistent buffers that are excluded from `state_dict()` on save but are already held by the model instance.

The CE-only baseline uses the same per-epoch pattern with `PRUNED_FINETUNED_SAVE_PATH`.

The combined loss follows Hinton et al. (2015) exactly:

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\!\left(\sigma\!\left(\frac{z_s}{T}\right) \,\|\, \sigma\!\left(\frac{z_t}{T}\right)\right) + (1-\alpha) \cdot \text{CE}(z_s, y)$$

where $z_s$ and $z_t$ are raw student/teacher logits, $\sigma$ is softmax, $y$ are hard labels, and the CE term always uses $T=1$.

The CE-only baseline uses `alpha=0.0` (hard-label CE only, run once per model pair, not part of the grid).

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Top-1 accuracy (%) | Primary metric |
| ΔAcc vs pruned-no-train | Recovery from pruning |
| ΔAcc KD vs CE-only | KD gain |
| ΔAcc vs teacher | Remaining gap |
| Avg forward ms / batch | Inference efficiency (student vs teacher) |

---

## Expected Results Table (to be filled)

One row per `(pair, alpha, temperature)` configuration. CE-only (α=0) is the fixed baseline for each pair.

| Pair | Alpha | Temp | Teacher | Pruned | CE-only | KD | KD gain |
|------|-------|------|---------|--------|---------|-----|--------|
| P1 | 0 (baseline) | — | — | — | — | — | — |
| P1 | 0.3 | 1.0 | — | — | — | — | — |
| P1 | 0.3 | 2.0 | — | — | — | — | — |
| P1 | 0.3 | 4.0 | — | — | — | — | — |
| P1 | 0.3 | 8.0 | — | — | — | — | — |
| P1 | 0.3 | 16.0 | — | — | — | — | — |
| P1 | 0.5 | 1.0 | — | — | — | — | — |
| P1 | 0.5 | 2.0 | — | — | — | — | — |
| P1 | 0.5 | 4.0 | — | — | — | — | — |
| P1 | 0.5 | 8.0 | — | — | — | — | — |
| P1 | 0.5 | 16.0 | — | — | — | — | — |
| P1 | 0.7 | 1.0 | — | — | — | — | — |
| P1 | 0.7 | 2.0 | — | — | — | — | — |
| P1 | 0.7 | 4.0 | — | — | — | — | — |
| P1 | 0.7 | 8.0 | — | — | — | — | — |
| P1 | 0.7 | 16.0 | — | — | — | — | — |
| P1 | 0.9 | 1.0 | — | — | — | — | — |
| P1 | 0.9 | 2.0 | — | — | — | — | — |
| P1 | 0.9 | 4.0 | — | — | — | — | — |
| P1 | 0.9 | 8.0 | — | — | — | — | — |
| P1 | 0.9 | 16.0 | — | — | — | — | — |
| P1 | 1.0 | 1.0 | — | — | — | — | — |
| P1 | 1.0 | 2.0 | — | — | — | — | — |
| P1 | 1.0 | 4.0 | — | — | — | — | — |
| P1 | 1.0 | 8.0 | — | — | — | — | — |
| P1 | 1.0 | 16.0 | — | — | — | — | — |
| P2 | 0 (baseline) | — | — | — | — | — | — |
| P2 | 0.3 | 1.0 | — | — | — | — | — |
| P2 | 0.3 | 2.0 | — | — | — | — | — |
| P2 | 0.3 | 4.0 | — | — | — | — | — |
| P2 | 0.3 | 8.0 | — | — | — | — | — |
| P2 | 0.3 | 16.0 | — | — | — | — | — |
| P2 | 0.5 | 1.0 | — | — | — | — | — |
| P2 | 0.5 | 2.0 | — | — | — | — | — |
| P2 | 0.5 | 4.0 | — | — | — | — | — |
| P2 | 0.5 | 8.0 | — | — | — | — | — |
| P2 | 0.5 | 16.0 | — | — | — | — | — |
| P2 | 0.7 | 1.0 | — | — | — | — | — |
| P2 | 0.7 | 2.0 | — | — | — | — | — |
| P2 | 0.7 | 4.0 | — | — | — | — | — |
| P2 | 0.7 | 8.0 | — | — | — | — | — |
| P2 | 0.7 | 16.0 | — | — | — | — | — |
| P2 | 0.9 | 1.0 | — | — | — | — | — |
| P2 | 0.9 | 2.0 | — | — | — | — | — |
| P2 | 0.9 | 4.0 | — | — | — | — | — |
| P2 | 0.9 | 8.0 | — | — | — | — | — |
| P2 | 0.9 | 16.0 | — | — | — | — | — |
| P2 | 1.0 | 1.0 | — | — | — | — | — |
| P2 | 1.0 | 2.0 | — | — | — | — | — |
| P2 | 1.0 | 4.0 | — | — | — | — | — |
| P2 | 1.0 | 8.0 | — | — | — | — | — |
| P2 | 1.0 | 16.0 | — | — | — | — | — |
| P3 | 0 (baseline) | — | — | — | — | — | — |
| P3 | 0.3 | 1.0 | — | — | — | — | — |
| P3 | 0.3 | 2.0 | — | — | — | — | — |
| P3 | 0.3 | 4.0 | — | — | — | — | — |
| P3 | 0.3 | 8.0 | — | — | — | — | — |
| P3 | 0.3 | 16.0 | — | — | — | — | — |
| P3 | 0.5 | 1.0 | — | — | — | — | — |
| P3 | 0.5 | 2.0 | — | — | — | — | — |
| P3 | 0.5 | 4.0 | — | — | — | — | — |
| P3 | 0.5 | 8.0 | — | — | — | — | — |
| P3 | 0.5 | 16.0 | — | — | — | — | — |
| P3 | 0.7 | 1.0 | — | — | — | — | — |
| P3 | 0.7 | 2.0 | — | — | — | — | — |
| P3 | 0.7 | 4.0 | — | — | — | — | — |
| P3 | 0.7 | 8.0 | — | — | — | — | — |
| P3 | 0.7 | 16.0 | — | — | — | — | — |
| P3 | 0.9 | 1.0 | — | — | — | — | — |
| P3 | 0.9 | 2.0 | — | — | — | — | — |
| P3 | 0.9 | 4.0 | — | — | — | — | — |
| P3 | 0.9 | 8.0 | — | — | — | — | — |
| P3 | 0.9 | 16.0 | — | — | — | — | — |
| P3 | 1.0 | 1.0 | — | — | — | — | — |
| P3 | 1.0 | 2.0 | — | — | — | — | — |
| P3 | 1.0 | 4.0 | — | — | — | — | — |
| P3 | 1.0 | 8.0 | — | — | — | — | — |
| P3 | 1.0 | 16.0 | — | — | — | — | — |

---

## Known Issues and Mitigations

### 1 — Double-softmax on teacher logits (FIXED)
The ultralytics `Classify` head returns `x.softmax(1)` in eval mode. Running `teacher.eval()` then calling `teacher(images)` feeds probabilities (not raw logits) into `soft_logit_kl_loss`, which applies a second softmax — producing severely distorted uniform soft targets.

**Mitigation (implemented in `src/mase_kd/vision/yolo_kd.py`):** teacher is temporarily switched to `train()` inside a `torch.no_grad()` block for each forward call, then restored to `eval()`.

### 2 — Learning rate too high causes post-pruning collapse
`lr = 1e-3` (default Adam LR) is suitable for training from scratch but destroys pre-trained weights over a few epochs when the model already achieves >80% accuracy.

**Mitigation:** use `lr = 5e-4` with AdamW (`weight_decay=0.05`), matching the notebook baseline.

### 3 — `_align_logits` strict shape assertion
Previously truncated teacher/student logits silently when dimensions mismatched. Now raises `ValueError` on any shape mismatch so regressions are caught immediately.

### 4 — Double-softmax in `evaluate()` and standalone `evaluate_model` (FIXED in `yolo_kd.py`; OUTSTANDING in notebooks)
The `_eval_model` closure inside `YOLOLogitsDistiller.evaluate()` ran the model in `eval()` mode, so the Classify head returned `x.softmax(1)` instead of raw logits. `hard_label_ce_loss` then operated on already-softmaxed values, producing incorrect CE loss numbers. **Top-1 accuracy is unaffected** (argmax is monotonic over softmax).

**Fixed in `src/mase_kd/vision/yolo_kd.py` (Mar 21 2026):** `_eval_model` now temporarily switches the model to `train()` inside `@torch.no_grad()`, matching the teacher forward pattern in `train_step`.

**Fixed in `P3_grid_search.py` (Mar 21 2026):** `evaluate_model` now uses the train-mode trick (switches to `model.train()` inside `torch.no_grad()` for the forward call) and also returns `top5_acc`.

**Outstanding in individual experiment notebooks (`exp_00`–`exp_25`):** The standalone `evaluate_model` function in those notebooks still calls `model.eval()` then `model(images)` without the train-mode trick. Reported `avg_ce_loss` for pruned and CE-only baselines will be incorrect; top-1 accuracy is fine.

---

## Progress Status (Mar 20–21, 2026)

**P3 grid search script (`P3_grid_search.py`) is the canonical runner.** The 26 individual experiment notebooks (`exp_00`–`exp_25`) that were previously generated via `generate_notebooks.py` have been deleted (Mar 22 2026) — `P3_grid_search.py` covers all runs more efficiently and is the single source of truth for P3 results.

**Best-model checkpointing added (Mar 20 2026):** Every notebook saves and restores the best-epoch weights:
- `exp_00`: CE finetune loop evaluates val top-1 after each epoch; checkpoint saved to `data/best_pruned_finetuned_exp_00.pt`; best weights restored via `load_state_dict(..., strict=False)` before final evaluation.
- `exp_01`–`exp_25`: `distiller.train(save_path=KD_SAVE_PATH)` saves best student per epoch to `data/best_student_exp_NN.pt`; best weights restored before `distiller.evaluate()`.

**Actual batch size used: 128** (differs from table above which reflected an earlier draft).

**`P3_grid_search.py` added (Mar 20 2026):** A single Python script that runs all 26 P3 experiments sequentially, sharing the teacher, dataloaders, and pruned student snapshot across runs. Equivalent to running all notebooks back-to-back but more efficient (no repeated model loading / pruning). Checkpoints saved to `data/P3_grid_search/`. Run with `cd cw && python YOLO_logitKD_experiments/P3_grid_search.py`.

**Code quality fixes in `yolo_kd.py` (Mar 21 2026):**
- Moved `import math` / `import time` to top-level import block.
- Fixed `_eval_model` double-softmax bug in `evaluate()` (see Known Issue #4).
- Simplified `.detach().cpu().item()` → `.item()` in `train_step`.
- Removed redundant `.to(device)` in `train()` inner loop (already done by `train_step`).

**`P3_grid_search.py` updates (Mar 21 2026):**
- `EPOCHS` increased from `8` → `50`.
- Optimizer changed from `Adam` to `AdamW`; `LR` set to `5e-4`; `WEIGHT_DECAY = 0.05`; both AdamW optimizers (CE and KD) now pass `weight_decay=WEIGHT_DECAY`. These values match the notebook (`yolo_distillation_cls_cifar10.ipynb`).
- CIFAR10 data augmentation added: `cifar_transform_train` uses `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ToTensor()`, `RandomErasing(p=0.1)`; `cifar_transform_eval` uses plain `ToTensor()`. Applied to train/val datasets respectively.
- `top5_acc` added: `evaluate_model` now tracks `correct_top5` and returns it; `teacher_top5`, `pruned_top5`, `ce_only_top5`, `kd_top5` columns added to CSV; per-experiment summary table widened with a `Top-5` column.
- `evaluate_model` double-softmax fixed: uses `model.train()` inside `torch.no_grad()` for each forward call, matching the pattern in `YOLOLogitsDistiller`.
- `torch.load(..., weights_only=True)` applied to both checkpoint restore calls.

---

## Implementation Checklist

- [x] Verify P3 teacher checkpoint exists (`yolov8x-cls` CIFAR10 fine-tuned weights confirmed)
- [x] Confirm `yolov8x-cls` CIFAR10 fine-tuning is complete
- [x] Fine-tune `yolov8x-cls` teacher on CIFAR100 and verify checkpoint (P2) — checkpoint at `cw/data/cifar100_yolov8x_cls_adamW/runs/yolov8x_cls_cifar100_finetune/weights/best.pt`
- [ ] Fine-tune `yolov8x-cls` teacher on Tiny ImageNet and verify checkpoint (P1)
- [x] Generate 26 individual notebooks in `cw/YOLO_logitKD_experiments/` covering full P3 α×T grid (deleted Mar 22 2026 — superseded by `P3_grid_search.py`)
- [x] Generate 26 individual notebooks for P2 (CIFAR100) α×T grid — not needed; will use a `P2_grid_search.py` script instead
- [x] Generate 26 individual notebooks for P1 (Tiny ImageNet) α×T grid — not needed; will use a `P1_grid_search.py` script instead
- [x] Add a result-logging cell that appends one row to `results.csv` after each run
- [x] Add per-epoch best-model checkpointing (`save_path` in `distiller.train()`; `strict=False` restore after training)
- [x] Create `P3_grid_search.py` — merged script running all 26 P3 experiments in one go
- [x] Fix `evaluate_model` double-softmax in `P3_grid_search.py` (Known Issue #4)
- [ ] Validate T=1.0 run produces the same result as nearly-hard-label CE (sanity check)
- [ ] Validate α=1.0 run trains without CE term (confirm `compute_distillation_loss` handles `targets=None` gracefully when α=1.0)
- [ ] Run P3 primary sweep (26 runs via `P3_grid_search.py`) and record results
- [ ] Run P2 primary sweep (26 runs, CIFAR100) and record results
- [ ] Run P1 primary sweep (26 runs, Tiny ImageNet) and record results
- [ ] Plot 5×5 heatmap of top-1 KD gain vs alpha and temperature (per pair)
- [ ] Plot KD gain vs dataset complexity (P3→P2→P1) at the best (α, T) config
- [ ] Report final summary table

---

## File References

| File | Role |
|------|------|
| `src/mase_kd/vision/yolo_kd.py` | `YOLOLogitsDistiller` — training and evaluation loop |
| `src/mase_kd/core/losses.py` | `DistillationLossConfig`, `soft_logit_kl_loss`, `hard_label_ce_loss` |
| `cw/yolo_distillation_cls_cifar10.ipynb` | Main experiment notebook (P3 as current baseline, no pruning) |
| `cw/yolo_distillation_cls_cifar100.ipynb` | P2 general distillation notebook (baseline for CIFAR100 run) |
| `cw/YOLO_logitKD_experiments/P3_grid_search.py` | Merged script running all 26 P3 experiments sequentially |
| `cw/YOLO_logitKD_experiments/generate_notebooks.py` | Generator script for the 26 individual notebooks |
| `cw/data/cifar10_yolov8x_cls/` | P3 teacher checkpoint directory (`yolov8x-cls` fine-tuned on CIFAR10) |
| `cw/data/cifar100_yolov8x_cls_adamW/` | P2 teacher checkpoint directory (`yolov8x-cls` fine-tuned on CIFAR100 with AdamW) |
| `cw/data/tinyimagenet_yolov8x_cls/` | P1 teacher checkpoint directory (`yolov8x-cls` fine-tuned on Tiny ImageNet) |
| `yolov8m-cls.pt` (ultralytics default) | Student seed — ImageNet pretrained weights (loaded via `YOLO('yolov8m-cls.pt').model`) |
