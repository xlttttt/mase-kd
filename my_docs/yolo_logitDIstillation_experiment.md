# YOLO Logits Distillation — Experiment Design

## Goal

Systematically measure whether logits-based Knowledge Distillation (KD) improves a pruned YOLOv8-cls student relative to CE-only fine-tuning on CIFAR10, and identify which hyperparameter settings produce the best recovery of teacher accuracy.

The experiment is grounded in Hinton, Vinyals & Dean (2015) *"Distilling the Knowledge in a Neural Network"* (the original KD paper). Key theoretical commitments taken from that paper:

- **Soft targets carry "dark knowledge"**: near-zero class probabilities encode similarity structure the teacher learned. Higher temperature T spreads this signal across more classes.
- **T² gradient re-scaling**: the KL-div soft loss is multiplied by T² so the gradient magnitude stays constant regardless of T (see `soft_logit_kl_loss` in `losses.py`).
- **Hard-label CE always uses T=1**: the cross-entropy term always receives raw logits; only the KL-div term is temperature-scaled.
- **Pure soft-target regime (α=1.0)**: Hinton et al. show that for large enough datasets the student can learn effectively from soft targets alone.

---

## Baseline Setup

| Item | Value |
|------|-------|
| Dataset | CIFAR10 (50 000 train / 10 000 val) |
| Image size | 32 × 32 (native CIFAR10, no resize) |
| Metric | Top-1 accuracy on full val set |
| Batch size | 128 |
| Seed | 42 (fixed for all runs) |

---

## Model Pairs

Three teacher/student pairs are evaluated to test whether KD gains are consistent across capacity gaps.

| Pair ID | Teacher | Student seed | Capacity gap | KD paper rationale |
|---------|---------|--------------|--------------|--------------------|
| P1 | `yolov8n-cls` (CIFAR10 fine-tuned) | `yolov8n-cls` (CIFAR10 fine-tuned) | None (self-distillation) | Tests regularisation effect of soft targets — Hinton et al. show self-KD still helps |
| P2 | `yolov8m-cls` (CIFAR10 fine-tuned) | `yolov8n-cls` (CIFAR10 fine-tuned) | Medium | Primary sweep pair; moderate capacity gap where KD benefit is clearest |
| P3 | `yolov8x-cls` (CIFAR10 fine-tuned) | `yolov8m-cls` (CIFAR10 fine-tuned) | Large | Tests whether a large gap limits knowledge transfer as noted in the paper |

Each student seed is pruned (unstructured magnitude pruning, `sparsity=0.50`) via MaseGraph before training begins.

---

## Variables

### Fixed across all runs
- Pruning sparsity: `0.50`
- Epochs: `5`
- Optimizer: Adam
- LR: `1e-4` (safe post-pruning fine-tuning rate; see Known Issues)

### Primary sweep: KD hyperparameters

| Variable | Values | Rationale (from original KD paper) |
|----------|--------|------------------------------------|
| `alpha` (soft-loss weight) | 0.3, 0.5, 0.7, **0.9, 1.0** | The paper recommends testing the pure soft-target limit (α=1.0). High α lets the model learn entirely from dark knowledge without hard-label interference. |
| `temperature` | **1.0**, 2.0, 4.0, 8.0, **16.0** | T=1.0 is a degenerate baseline (nearly hard teacher argmax); T=16 surfaces inter-class similarity at finer granularity. For 10-class CIFAR10, the sweet spot is expected between T=4 and T=8. |

> **Note:** α=1.0 runs skip the CE term entirely. α=0.0 (pure CE) is the fine-tuned baseline and is not part of this grid — it is run separately once per model pair.

Full grid: 5 × 5 = **25 configurations per model pair** (75 runs total for P1–P3).

### Secondary sweep: learning rate sensitivity (P2 only)

| LR | Rationale |
|----|-----------|
| `5e-5` | Conservative |
| `1e-4` | Default recommendation |
| `3e-4` | Aggressive upper bound |

Run with the best `(alpha, temperature)` from the primary sweep.

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
    optimizer        = Adam(student.parameters(), lr=LR),
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

One row per `(pair, alpha, temperature)` configuration. CE-only (α=0) is listed once per pair as a fixed baseline.

| Pair | Alpha | Temp | Teacher | Pruned | CE-only | KD | KD gain |
|------|-------|------|---------|--------|---------|-----|------|
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
| P2 | 0.7 | 4.0 | — | — | — | — | — |
| P2 | 0.7 | 8.0 | — | — | — | — | — |
| P2 | 0.7 | 16.0 | — | — | — | — | — |
| P2 | 0.9 | 4.0 | — | — | — | — | — |
| P2 | 0.9 | 8.0 | — | — | — | — | — |
| P2 | 0.9 | 16.0 | — | — | — | — | — |
| P2 | 1.0 | 4.0 | — | — | — | — | — |
| P2 | 1.0 | 8.0 | — | — | — | — | — |
| P2 | 1.0 | 16.0 | — | — | — | — | — |
| P1 | 0 (baseline) | — | — | — | — | — | — |
| P1 | best from P2 | best from P2 | — | — | — | — | — |
| P3 | 0 (baseline) | — | — | — | — | — | — |
| P3 | best from P2 | best from P2 | — | — | — | — | — |

---

## Known Issues and Mitigations

### 1 — Double-softmax on teacher logits (FIXED)
The ultralytics `Classify` head returns `x.softmax(1)` in eval mode. Running `teacher.eval()` then calling `teacher(images)` feeds probabilities (not raw logits) into `soft_logit_kl_loss`, which applies a second softmax — producing severely distorted uniform soft targets.

**Mitigation (implemented in `src/mase_kd/vision/yolo_kd.py`):** teacher is temporarily switched to `train()` inside a `torch.no_grad()` block for each forward call, then restored to `eval()`.

### 2 — Learning rate too high causes post-pruning collapse
`lr = 1e-3` (default Adam LR) is suitable for training from scratch but destroys pre-trained weights over a few epochs when the model already achieves >80% accuracy.

**Mitigation:** use `lr = 1e-4` as the default; include `5e-5` and `3e-4` in the secondary LR sweep.

### 3 — `_align_logits` strict shape assertion
Previously truncated teacher/student logits silently when dimensions mismatched. Now raises `ValueError` on any shape mismatch so regressions are caught immediately.

---

## Progress Status (Mar 20, 2026)

**P3 experiment notebooks fully generated and updated with best-model checkpointing.** Instead of parameterising a single notebook, 26 standalone notebooks were generated via `cw/YOLO_logitKD_experiments/generate_notebooks.py`:

| Notebook | Description |
|----------|-------------|
| `exp_00_P3_baseline_CE.ipynb` | CE-only fine-tuning baseline (α=0) |
| `exp_01_P3_a0.3_T1.0.ipynb` – `exp_05_P3_a0.3_T16.0.ipynb` | α=0.3 sweep (T=1.0, 2.0, 4.0, 8.0, 16.0) |
| `exp_06_P3_a0.5_T1.0.ipynb` – `exp_10_P3_a0.5_T16.0.ipynb` | α=0.5 sweep |
| `exp_11_P3_a0.7_T1.0.ipynb` – `exp_15_P3_a0.7_T16.0.ipynb` | α=0.7 sweep |
| `exp_16_P3_a0.9_T1.0.ipynb` – `exp_20_P3_a0.9_T16.0.ipynb` | α=0.9 sweep |
| `exp_21_P3_a1.0_T1.0.ipynb` – `exp_25_P3_a1.0_T16.0.ipynb` | α=1.0 sweep (pure soft targets) |

All 26 notebooks are JSON-valid and share a common `cw/YOLO_logitKD_experiments/results.csv`. Each notebook appends one result row to the CSV after training.

**Best-model checkpointing added (Mar 20 2026):** Every notebook saves and restores the best-epoch weights:
- `exp_00`: CE finetune loop evaluates val top-1 after each epoch; checkpoint saved to `data/best_pruned_finetuned_exp_00.pt`; best weights restored via `load_state_dict(..., strict=False)` before final evaluation.
- `exp_01`–`exp_25`: `distiller.train(save_path=KD_SAVE_PATH)` saves best student per epoch to `data/best_student_exp_NN.pt`; best weights restored before `distiller.evaluate()`.

**Actual batch size used: 128** (differs from table above which reflected an earlier draft).

---

## Implementation Checklist

- [x] Verify P3 teacher checkpoints exist (`yolov8x-cls` and `yolov8m-cls` CIFAR10 fine-tuned weights confirmed)
- [x] Confirm `yolov8x-cls` and `yolov8m-cls` CIFAR10 fine-tuning is complete
- [x] Generate 26 individual notebooks in `cw/YOLO_logitKD_experiments/` covering full P3 α×T grid (replaced single-notebook parameterisation)
- [x] Add a result-logging cell that appends one row to `results.csv` after each run
- [x] Add per-epoch best-model checkpointing (`save_path` in `distiller.train()`; `strict=False` restore after training)
- [ ] Verify P1 and P2 teacher checkpoints exist (`yolov8n-cls`, `yolov8m-cls`)
- [ ] Validate T=1.0 run produces the same result as nearly-hard-label CE (sanity check)
- [ ] Validate α=1.0 run trains without CE term (confirm `compute_distillation_loss` handles `targets=None` gracefully when α=1.0)
- [ ] Run P3 primary sweep (26 notebooks) and record results
- [ ] Run P2 primary sweep (25 configs + baseline) and record results
- [ ] Run P1 and P3 baseline (CE-only) + best config from P2
- [ ] Run P2 secondary LR sweep (`5e-5`, `1e-4`, `3e-4`) with best `(alpha, temperature)`
- [ ] Plot 5×5 heatmap of top-1 KD gain vs alpha and temperature (P3)
- [ ] Plot KD gain vs capacity gap (P1/P2/P3) at the best config
- [ ] Report final summary table

---

## File References

| File | Role |
|------|------|
| `src/mase_kd/vision/yolo_kd.py` | `YOLOLogitsDistiller` — training and evaluation loop |
| `src/mase_kd/core/losses.py` | `DistillationLossConfig`, `soft_logit_kl_loss`, `hard_label_ce_loss` |
| `cw/yolo_pruning_distillation_cls_5.ipynb` | Main experiment notebook (P3 as current baseline) |
| `cw/data/cifar10_yolov8x_cls/` | Teacher checkpoint directory |
| `cw/data/cifar10_yolov8m_cls/` | Student seed checkpoint directory |
