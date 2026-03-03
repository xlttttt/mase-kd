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

## Next step

- Move to **classification** distillation as the next milestone:
	- use `yolov8n-cls` teacher/student setup,
	- include true classification hard loss (cross-entropy with CIFAR10 labels),
	- track top-1 accuracy + CE on validation,
	- compare pruned no-KD vs KD student fairly.
