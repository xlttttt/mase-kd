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

## Next step

- Re-run `cw/yolo_pruning_distillation_cls_5.ipynb` with `lr = 1e-4` after the double-softmax fix and record accuracy for all 4 models.
- Begin BERT KD pipeline (`src/mase_kd/nlp/bert_kd.py`).
