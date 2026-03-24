# YOLO Distillation Progress

## Current status (Mar 4, 2026)

### Detection pipeline (`cw/yolo_pruning_distillation.ipynb`)
- End-to-end **YOLOv8n detection** teacher → pruned student → logits KD flow complete.
- Evaluation: teacher / pruned-no-KD / distilled student on CIFAR10 val with CE and KD losses.
- Current result: detection KD pipeline running end-to-end with measurable (small) KD improvement over no-KD baseline.

### Classification pipeline (`cw/yolo_pruning_distillation_cls.ipynb`)
- Fine-tuned `yolov8n-cls` on CIFAR10 (20 epochs, imgsz=32, batch 64) → **top-1 ~77.6%**, checkpoint at `cw/data/cifar10_yolov8n_cls/runs/yolov8n_cls_cifar10_finetune4/weights/best.pt`.
- Updated notebook to use the fine-tuned teacher as the source for both teacher inference and student initialisation:
	- teacher loaded from `best.pt` (10-class head, nc=10),
	- student initialised from teacher weights via `MaseYoloClassificationModel(cfg="yolov8n-cls.yaml", nc=10)` + `patch_yolo` + `load_state_dict` (bypasses `get_yolo_classification_model` assertion/class-mismatch),
	- `transforms.Resize` removed — CIFAR10 native 32×32 matches `imgsz=32` used during fine-tuning,
	- `task_loss_fn` / `classification_hard_loss` wrapper removed — `YOLOLogitsDistiller.train_step` handles CE internally via `compute_distillation_loss(..., targets=targets)`,
	- `_extract_logits_with_batch` moved into `YOLOLogitsDistiller` as a `@staticmethod` in `src/mase_kd/vision/yolo_kd.py`,
	- loss calls in notebook delegate to `hard_label_ce_loss` and `soft_logit_kl_loss` from `src/mase_kd/core/losses.py`,
	- evaluation compares teacher / pruned-only (no KD) / distilled student with top-1 accuracy and CE loss on full CIFAR10 val (now meaningful with CIFAR10-trained teacher).
- **KD training loop encapsulated in `YOLOLogitsDistiller`** (`src/mase_kd/vision/yolo_kd.py`):
	- `train_loader` and `optimizer` are now constructor arguments (stored on the instance),
	- `train_step` `optimizer` parameter made optional — falls back to `self.optimizer`,
	- new `train()` method drives the full epoch-based loop and returns `loss_history`; no arguments required for a standard run,
	- notebook KD cell reduced to constructing the distiller and calling `distiller_cls.train()`.
- **Evaluation integrated into `YOLOLogitsDistiller`**:
	- two new constructor arguments: `val_loader: DataLoader | None = None` and `eval_teacher: bool = True`,
	- new `evaluate()` method (`@torch.no_grad()`) runs on `val_loader` and returns a dict with keys `"student"`, `"teacher"` (only when `eval_teacher=True`), `"val_kd_loss"`, and `"kd_batches"`; each model dict contains `top1_acc`, `avg_ce_loss`, `avg_forward_ms_per_batch`, `samples`, and `batches`,
	- `hard_label_ce_loss` and `soft_logit_kl_loss` imported directly into `yolo_kd.py` to support the evaluation logic,
	- notebook evaluation cell updated to call `distiller_cls.evaluate()` for teacher and distilled student; the pruned-no-KD baseline is kept as a standalone `evaluate_model_on_cifar10_val` call (that model is not managed by the distiller).
- **Switched from step-based to epoch-based training**:
	- `YOLOLogitsDistiller.__init__` accepts `num_train_epochs: int = 1`; `train()` iterates over the full loader for that many epochs,
	- config variable renamed from `cifar_kd_steps` to `cifar_kd_epochs` (currently set to 1).
- **Full CIFAR10 dataset** used for both training and evaluation (previously capped at 2048 train / 512 val subsets); `Subset` and related index logic removed; variables renamed `train_full` → `train_dataset`, `val_full` → `val_dataset`.

### Teacher/student logits dimension mismatch fix (`src/mase_kd/vision/yolo_kd.py`)
- **Root cause identified**: the ultralytics `Classify` head returns a plain `[B, nc]` tensor in training mode but a `(softmax_probs, raw_logits)` tuple of two `[B, nc]` tensors in eval mode. The teacher runs in eval mode (frozen), the student runs in training mode, so `_flatten_logits` was concatenating the teacher's two tensors into `[B, 2*nc]` while the student produced `[B, nc]` — a 2× dimension mismatch. `_align_logits` was silently masking this by truncating to the smaller dimension.
- **Fix (Approach 1 + 3)**:
  - New `_unwrap_classify_output` static method: detects the 2-element same-shaped tensor tuple and returns the second element (raw logits), passes everything else through unchanged. Applied to both teacher and student outputs in `train_step` and in the KD-loss loop inside `evaluate()` before `_flatten_logits` is called.
  - `_align_logits` rewritten as a strict shape assertion — raises `ValueError` on any mismatch instead of silently truncating, so future regressions surface immediately.

## Current status (Mar 5, 2026)

### Classification notebook extended with finetuned-pruned baseline and memory cleanup

- **4-model comparison**: final evaluation now compares teacher, pruned (no finetune/no KD), pruned (finetuned CE-only), and distilled student.
- **"Evaluate pruned model" cell**: `evaluate_model_on_cifar10_val` defined here; evaluates `pruned_no_kd_model` immediately after pruning and stores results in `pruned_no_kd_metrics`.
- **"Finetune and evaluate pruned model (No Distillation)" cell**: deep-copies `pruned_no_kd_model` into `pruned_finetuned_model`, trains it for `EPOCHS` epochs with CE loss only (no teacher), evaluates into `pruned_finetuned_metrics`.
- **Memory cleanup cell** inserted between the finetune section and the distillation section: deletes `ultra_teacher`, `student_seed_cls_model`, `mg_cls`, `pruned_no_kd_model`, `pruned_finetuned_model`, and `ft_optimizer` from globals, then calls `gc.collect()` and `torch.cuda.empty_cache()`. Metrics dicts are preserved; `teacher_cls_model` and `student_cls_model` are kept for distillation.
- **Final summary table** widened to 45 chars and rows updated for all 4 models.

## Current status (Mar 20, 2026)

### Upgraded teacher/student pair: yolov8x-cls → yolov8m-cls (`cw/yolo_pruning_distillation_cls_5.ipynb`)

- Teacher upgraded to `yolov8x-cls` (fine-tuned on CIFAR10), student seed upgraded to `yolov8m-cls` (fine-tuned on CIFAR10).
- Post-pruning accuracy reported as **82.74%**, but after CE-only fine-tuning or KD the model collapsed to ~13.76% (near-random for 10 classes).

### Root-cause analysis and fixes for post-pruning training collapse

Two distinct bugs identified:

**Bug 1 — Learning rate too high**
- `lr = 1e-3` is the default Adam LR for training from scratch; it is far too aggressive when fine-tuning a model that already achieves 82.74%.
- Over 3 epochs × ~3125 batches the pre-trained weights are overwritten before the model can re-converge, causing accuracy to collapse. The fix is to reduce `lr` to `1e-4` or `5e-5`.

**Bug 2 — Double-softmax on teacher logits (fixed in `src/mase_kd/vision/yolo_kd.py`)**
- The ultralytics `Classify` head does `return x if self.training else x.softmax(1)`. Because `__init__` calls `self.teacher.eval()`, every teacher forward during `train_step` and the KD-loss loop in `evaluate()` returned **softmax probabilities**, not raw logits.
- `_unwrap_classify_output` did not help — it only matches a 2-element tuple; a single softmax tensor passes through unchanged.
- `soft_logit_kl_loss` then computed `F.softmax(softmax_probs / T, dim=-1)` — a double-softmax that produces severely distorted, uniform soft targets.
- **Fix**: teacher is now temporarily switched to `train()` inside a `torch.no_grad()` block for each forward call (in both `train_step` and the KD-loss loop in `evaluate()`), then immediately restored to `eval()`. This makes the Classify head return raw logits without computing or accumulating any gradients.
- `_unwrap_classify_output` docstring corrected to reflect actual ultralytics behaviour and document the train-mode pattern.

## Current status (Mar 20, 2026) — continued

### Best-model checkpointing in `YOLOLogitsDistiller.train()` and notebook

**`src/mase_kd/vision/yolo_kd.py`**
- `train()` now accepts an optional `save_path: str | None = None` parameter.
- At the end of each epoch the method computes an epoch-level metric: epoch-average top-1 accuracy when targets are available, otherwise epoch-average total loss (lower is better).
- When the metric improves over all previous epochs, `self.student.state_dict()` is saved to `save_path` via `torch.save` and a `[checkpoint]` line is printed.
- Fully backwards-compatible: default `save_path=None` disables checkpointing, matching previous behaviour.

**`cw/yolo_pruning_distillation_cls_5.ipynb`**
- Config cell: added `PRUNED_FINETUNED_SAVE_PATH = "data/best_pruned_finetuned_5.pt"` and `KD_SAVE_PATH = "data/best_student_5.pt"` alongside other hyperparameters.
- Finetune cell: saves best checkpoint per epoch to `PRUNED_FINETUNED_SAVE_PATH`; after the loop restores best weights via `load_state_dict(..., strict=False)` before evaluation.
- KD distiller cell: passes `save_path=KD_SAVE_PATH` to `distiller_cls.train()`; after training restores best student weights via `load_state_dict(..., strict=False)` so `distiller_cls.evaluate()` uses the best model.
- `strict=False` required in both `load_state_dict` calls because MASE's pruning pass registers sparsity masks as non-persistent buffers via `torch.nn.utils.parametrize`; these are excluded from `state_dict()` on save but still expected by the model — since the model instance already holds the correct masks, only trained weight values need restoring.

## Current status (Mar 21, 2026)

### Code quality fixes in `src/mase_kd/vision/yolo_kd.py`

- **Moved `import math` and `import time`** from inside method bodies to the top-level import block.
- **Fixed `_eval_model` double-softmax bug**: the inner `_eval_model` closure in `evaluate()` previously ran the model in eval mode and passed the output directly to `_extract_logits_with_batch`, so the Classify head returned `x.softmax(1)` and `hard_label_ce_loss` operated on already-softmaxed values. Fixed by temporarily switching the model to train mode inside the `@torch.no_grad()` block (matching the pattern used for the teacher in `train_step`), then restoring eval mode; `_unwrap_classify_output` is now also applied before extraction.
- **Simplified `.detach().cpu().item()` → `.item()`** in the `YOLOLogitsKDOutput` construction inside `train_step`; `.item()` already detaches, transfers to CPU, and returns a Python float so the extra calls were redundant.
- **Removed double device-move in `train()`**: the inner loop was calling `.to(self.device)` when building the batch dict, but `train_step` was already doing the same move on the values it receives — the first `.to()` was a no-op.
- **Removed extra blank line** between `train()` and `evaluate()`.

### Outstanding issue in `cw/yolo_pruning_distillation_cls_5.ipynb`

- `evaluate_model_on_cifar10_val` (cell 12) has the same double-softmax bug: it runs the model in eval mode and passes the output straight to `_extract_logits_with_batch` without the train-mode trick. Top-1 accuracy is unaffected (argmax is monotonic over softmax), but the reported `avg_ce_loss` values for `pruned_no_kd_metrics` and `pruned_finetuned_metrics` are incorrect.

## Current status (Mar 21, 2026) — continued

### New notebook `cw/yolo_pruning_distillation_cls_6.ipynb` — validation-set tracking and top-5 metrics

**`src/mase_kd/vision/yolo_kd.py` — `train()` now returns per-epoch validation metrics**
- Added a `torch.no_grad()` validation loop at the end of each epoch, using the same train-mode-raw-logits trick as `train_step` and `_eval_model`.
- Computes val top-1, val top-5, and val CE loss per epoch; appends one value per epoch to three new history lists.
- `train()` return dict updated: training keys renamed with `train_` prefix (`train_total_loss`, `train_top1_acc`, `train_top5_acc`); validation keys added as `val_top1_acc`, `val_top5_acc`, `val_loss` (per-epoch lists, NaN-filled when no `val_loader` is set).
- Checkpointing logic updated to prefer val top-1 → train top-1 → train loss (in that order), so the saved checkpoint is the best-generalising model rather than the best-fitting one.

**`src/mase_kd/vision/yolo_kd.py` — `evaluate()` now returns `top5_acc` per model**
- `_eval_model` closure adds `correct_top5` counter and top-k comparison; returns `top5_acc` alongside `top1_acc` in each per-model metrics dict.

**`cw/yolo_pruning_distillation_cls_6.ipynb` — consistent print format and top-5 everywhere**
- `evaluate_model_on_cifar10_val` updated to compute and return `top5_acc`; all existing callers updated.
- Finetune cell: added `ft_val_top5_history`; per-epoch val line now reports both `top1_acc` and `top5_acc`; checkpoints on val top-1 instead of train top-1.
- Distillation cell: extracts `val_top1_history`, `val_top5_history`, `val_loss_history` from `train_history`.
- Evaluation cell: `_fmt_metrics` helper ensures all model prints share the same format (`top1_acc=…% | top5_acc=…% | CE_loss=… | fwd_ms/batch=… | samples=…`); summary table widened with a `Top-5 Acc` column.
- Plotting cell: shows 4 lines per panel (finetuned train, finetuned val dashed, distilled train, distilled val dashed), making overfitting visible at a glance.

## Current status (Mar 22, 2026)

### Notebooks renamed: `cw/yolo_distillation_cls_cifar10.ipynb` and `cw/yolo_distillation_cls_cifar100.ipynb`

- `cw/yolo_pruning_distillation_cls_7.ipynb` renamed to `cw/yolo_distillation_cls_cifar10.ipynb` — now the canonical CIFAR10 baseline notebook ("pruning" removed from filenames since neither notebook prunes the student).
- `cw/yolo_pruning_distillation_cls_cifar100.ipynb` renamed to `cw/yolo_distillation_cls_cifar100.ipynb` for the same reason.

### `cw/yolo_distillation_cls_cifar100.ipynb` — CIFAR100 distillation

- Ported the full workflow from `cw/yolo_distillation_cls_cifar10.ipynb` to CIFAR100.
- **Dataset**: `torchvision.datasets.CIFAR100` (50k train / 10k val, 100 classes).
- **Teacher**: CIFAR100-fine-tuned YOLOv8x-cls loaded from `data/cifar100_yolov8x_cls_adamW/runs/yolov8x_cls_cifar100_finetune/weights/best.pt`; same two-step loading pattern (YOLO() → MaseYoloClassificationModel + patch_yolo + load_state_dict).
- **Student**: randomly-initialised YOLOv8m-cls (`nc=100`), no pruning.
- **Hyperparameters** match `_7.ipynb`: `EPOCHS=50`, `lr=5e-4`, `WEIGHT_DECAY=0.05`, `KD_ALPHA=0.6`, `KD_TEMPERATURE=8.0`, `BATCH_SIZE=128`.
- **Save paths**: `data/best_student_finetuned_cifar100.pt` (CE-only finetune best) and `data/best_student_cifar100.pt` (KD best).
- **Evaluation helper** renamed to `evaluate_model_on_cifar100_val`; all eval logic (train-mode raw-logits trick, `_unwrap_classify_output`, top-1/top-5/CE loss) identical to `_7.ipynb`.
- **4-section structure preserved**: baseline eval → CE-only finetune → memory cleanup → KD distillation → evaluation + summary table → training/validation curves plot.

## Next steps

- Run `cw/yolo_distillation_cls_cifar100.ipynb` and record accuracy for all 4 models (teacher / student-no-training / finetuned / distilled).
- Re-run `cw/yolo_pruning_distillation_cls_6.ipynb` with `lr = 1e-5` and record accuracy for all 4 models (teacher / student-no-training / finetuned / distilled).
- Begin BERT KD pipeline (`src/mase_kd/nlp/bert_kd.py`).
