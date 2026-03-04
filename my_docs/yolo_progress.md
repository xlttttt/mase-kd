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
- **Switched from step-based to epoch-based training**:
	- `YOLOLogitsDistiller.__init__` accepts `num_train_epochs: int = 1`; `train()` iterates over the full loader for that many epochs,
	- config variable renamed from `cifar_kd_steps` to `cifar_kd_epochs` (currently set to 5).
- **Full CIFAR10 dataset** used for both training and evaluation (previously capped at 2048 train / 512 val subsets); `Subset` and related index logic removed; variables renamed `train_full` → `train_dataset`, `val_full` → `val_dataset`.

## Next step

- Run the updated classification notebook end-to-end and verify meaningful accuracy differences between the three models.
- Begin BERT KD pipeline (`src/mase_kd/nlp/bert_kd.py`).
