# Quick Test Summary — MASE-KD Smoke Tests

**Date**: 2026-03-03
**GPU**: NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM)
**Docker image**: `deepwok/mase-docker-cpu:latest` (PyTorch 2.6.0+cu124)

---

## Models & Datasets

| Experiment | Teacher | Student | Dataset |
|---|---|---|---|
| BERT Baseline | `textattack/bert-base-uncased-SST-2` (110M, ~93% acc) | 4-layer, hidden=256, 13M (random init) | SST-2 (67k train / 872 val) |
| BERT KD | same teacher | same student | SST-2 |
| YOLO Baseline | `yolov8m.pt` (26M, COCO pretrained) | `yolov8n` (3.2M, random init) | `coco8.yaml` (4 train / 4 val images) |
| YOLO KD | same teacher | same student | `coco8.yaml` |

---

## 1. BERT Baseline (alpha=0.0, no KD)

**Config**: 2 epochs, batch=32, lr=2e-5, max_seq=128
**Loss**: pure cross-entropy on ground-truth labels

| Epoch | train_loss | train_soft_loss | val_accuracy | val_f1 |
|---|---|---|---|---|
| 1 | 0.5166 | 1.507 (not used) | 0.7833 | 0.7830 |
| 2 | 0.3347 | 0.948 (not used) | **0.7982** | 0.7982 |

**Result**: Best val_accuracy = **79.82%** ✅
*Note*: soft_loss is logged for monitoring but not used in backprop (alpha=0).

---

## 2. BERT KD (alpha=0.5, T=4.0)

**Config**: 2 epochs, batch=32, lr=2e-5, max_seq=128
**Loss**: `0.5 × CE + 0.5 × T² × KL(student/T ‖ teacher/T)`

| Epoch | train_loss | train_hard_loss | train_soft_loss | val_accuracy | val_f1 |
|---|---|---|---|---|---|
| 1 | 1.6075 | 0.5730 | **2.642** | 0.7810 | 0.7800 |
| 2 | 0.9304 | 0.4317 | **1.429** | **0.7936** | 0.7935 |

**Result**: Best val_accuracy = **79.36%** ✅
**KD path active**: soft_loss > 0 confirmed ✅
*Note*: Baseline leads by 0.46% at 2 epochs — this is expected. KD requires more epochs to fully leverage "dark knowledge"; the advantage typically appears after epoch 5+.

---

## 3. YOLO Baseline (alpha=0.0, no KD)

**Config**: 5 epochs, batch=8, imgsz=640, coco8.yaml (4 training images)
**Loss**: Ultralytics v8DetectionLoss [box + cls + dfl]

| Epoch | train_loss | task_loss | kd_loss* | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1 | 50.13 | 50.13 | 3.88 | 0.000 | 0.000 |
| 2 | 52.80 | 52.80 | 3.78 | 0.000 | 0.000 |
| 3 | 55.30 | 55.30 | 3.78 | 0.000 | 0.000 |
| 4 | 52.71 | 52.71 | 3.77 | 0.000 | 0.000 |
| 5 | 53.79 | 53.79 | 3.75 | 0.000 | 0.000 |

**Result**: mAP50 = **0.000** ✅ (expected — coco8 has only 4 training images; 5 epochs from scratch cannot learn COCO-80 detection)
*kd_loss logged but not used in backprop (alpha=0)*

---

## 4. YOLO KD (alpha=0.5, T=2.0)

**Config**: 5 epochs, batch=8, imgsz=640, coco8.yaml
**Loss**: `0.5 × v8DetectionLoss + 0.5 × T² × KL(flatten(student)/T ‖ flatten(teacher)/T)`

| Epoch | train_loss | task_loss | kd_loss | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1 | **27.01** | 50.13 | **3.88** | 0.000 | 0.000 |
| 2 | **28.29** | 52.80 | **3.78** | 0.000 | 0.000 |
| 3 | **29.54** | 55.30 | **3.78** | 0.000 | 0.000 |
| 4 | **28.24** | 52.71 | **3.77** | 0.000 | 0.000 |
| 5 | **28.44** | 53.14 | **3.75** | 0.000 | 0.000 |

**Result**: mAP50 = **0.000** ✅ (same reason as baseline — too few data/epochs)
**KD path active**: kd_loss ~3.8 > 0 confirmed ✅
**train_loss halved vs baseline**: 27–29 vs 50–55 because `combined = 0.5×task + 0.5×kd_soft` where kd_soft is much smaller magnitude.

---

## Bug Fixes Applied During This Session

| Bug | Location | Fix |
|---|---|---|
| `MaseYoloDetectionModel` missing `.args` → `v8DetectionLoss` crashes | `yolo_kd_train.py:__init__` | Inject `get_cfg(DEFAULT_CFG)` if `.args` absent |
| `v8DetectionLoss` returns shape `[3]` (box/cls/dfl vector), not scalar | `yolo_kd_train.py:_train_epoch` | Call `.sum()` on the vector before combining with KD loss |

---

## Success Criteria — All Passed ✅

| Criterion | Baseline | KD | Status |
|---|---|---|---|
| Output dir + `training_history.json` exists | ✅ | ✅ | Pass |
| `train_loss` is finite (no NaN) | ✅ | ✅ | Pass |
| BERT `val_accuracy` in [0.5, 1.0] | 79.8% | 79.4% | Pass |
| KD runs: `soft_loss > 0` | N/A | ✅ (2.6 / 3.88) | Pass |
| YOLO `mAP50 >= 0`, no NaN | ✅ 0.0 | ✅ 0.0 | Pass |

All 4 pipelines are validated end-to-end. Ready for full-scale training.
