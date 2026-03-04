# YOLO Distillation Progress

## Current status (Mar 3, 2026)

- Built and iterated on the coursework notebook pipeline in `cw/yolo_pruning_distillation.ipynb`.
- Implemented end-to-end **YOLOv8n detection** teacher → pruned student → logits KD flow on CIFAR10 image batches:
	- load Ultralytics teacher checkpoint,
	- build `MaseGraph` from YOLO detection model,
	- run unstructured pruning to form the student,
	- run logits distillation training loop,
	- evaluate teacher/student on CIFAR10 validation subset.
- Added validation reporting in notebook:
	- teacher validation loss,
	- student validation loss (with KD),
	- pruned-student validation loss (without KD baseline),
	- validation KD loss.
- Resolved key integration/debug issues:
	- import path resolution for `mase_kd`,
	- FX placeholder/dummy input mapping for metadata passes,
	- pruning pass config and metadata requirements,
	- robust output flattening/alignment for teacher-student comparisons,
	- notebook cell stability across reruns (missing-state guards).
- Current result: detection KD pipeline is running end-to-end with measurable (small) KD improvement over no-KD baseline.
- Built and iterated on `cw/yolo_pruning_distillation_cls.ipynb` for classification KD:
	- switched teacher/student to `yolov8n-cls`,
	- enabled CIFAR10 label-based hard loss (CE) during KD,
	- fixed batch-shape issues caused by FX-specialized pruned graph,
	- added validation reporting for CE, KD loss, and top-1 accuracy (teacher / KD student / no-KD student).
- Current classification observation: raw top-1 accuracy against CIFAR10 labels is not yet meaningful with the current ImageNet-style `yolov8n-cls` teacher head.

- Fine-tuned `yolov8n-cls` on CIFAR10 (20 epochs, imgsz=32, batch 64) → **top-1 ~77.6%**, saved to `cw/data/cifar10_yolov8n_cls/runs/yolov8n_cls_cifar10_finetune4/weights/best.pt`.
- Updated `cw/yolo_pruning_distillation_cls.ipynb` to use the CIFAR10-fine-tuned teacher:
	- teacher loaded from `best.pt` (10-class head, nc=10),
	- student is a pruned copy of the teacher (not from the ImageNet checkpoint),
	- bypassed `get_yolo_classification_model` (assertion + class mismatch) with manual `MaseYoloClassificationModel(cfg="yolov8n-cls.yaml", nc=10)` + `patch_yolo` + `load_state_dict`,
	- image size changed to 32 (matching teacher fine-tuning distribution),
	- evaluation now compares teacher / pruned-only (no KD) / distilled student with top-1 accuracy and CE loss on full CIFAR10 val subset.

## Next step

- Run the updated notebook end-to-end and verify meaningful accuracy differences between the three models.
- Consider increasing `cifar_kd_steps` or `cifar_train_subset_size` for stronger KD recovery.
- Begin BERT KD pipeline (`src/mase_kd/nlp/bert_kd.py`).
