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

## Next step

- Fine-tune `yolov8n-cls` on CIFAR10 so it becomes a proper teacher for this task:
	- adapt/fine-tune the classification head for CIFAR10 label space,
	- train and validate to obtain strong CIFAR10 top-1 accuracy,
	- save the fine-tuned checkpoint for teacher initialization,
	- re-run teacher→pruned-student KD comparison with task-faithful accuracy metrics.
