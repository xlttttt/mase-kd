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
	- evaluation compares teacher / pruned-only (no KD) / distilled student with top-1 accuracy and CE loss on full CIFAR10 val subset (now meaningful with CIFAR10-trained teacher).

## Next step

- Run the updated classification notebook end-to-end and verify meaningful accuracy differences between the three models.
- Consider increasing `cifar_kd_steps` or `cifar_train_subset_size` for stronger KD recovery.
- Begin BERT KD pipeline (`src/mase_kd/nlp/bert_kd.py`).
