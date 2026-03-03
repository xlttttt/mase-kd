# MASE-KD Usage Guide

Knowledge Distillation experiments for YOLO (object detection) and BERT (sequence classification), built on top of the MASE framework.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)
3. [BERT Experiments](#3-bert-experiments)
   - [Baseline Training](#31-baseline-training)
   - [KD Training](#32-kd-training)
   - [CLI Reference](#33-bert-cli-reference)
4. [YOLO Experiments](#4-yolo-experiments)
   - [Dataset Setup](#41-dataset-setup)
   - [Baseline Training](#42-baseline-training)
   - [KD Training](#43-kd-training)
5. [Ablation Studies](#5-ablation-studies)
6. [Evaluation & Comparison](#6-evaluation--comparison)
7. [Running Tests](#7-running-tests)
8. [Config Reference](#8-config-reference)
9. [Common Issues](#9-common-issues)

---

## 1. Environment Setup

```bash
# Clone and install in editable mode
cd /path/to/mase-kd
pip install -e .
```

**Requirements:** Python ≥ 3.11.9, PyTorch 2.6, GPU with ≥ 8 GB VRAM recommended.

Verify installation:
```bash
python -c "import mase_kd; import chop; print('OK')"
```

---

## 2. Project Structure

```
mase-kd/
├── src/mase_kd/              # Source code
│   ├── core/losses.py        # Shared KD loss functions
│   ├── nlp/bert_kd.py        # BERT KD trainer
│   ├── nlp/eval.py           # NLP evaluation utilities
│   ├── vision/yolo_kd.py     # YOLO logits distiller (core step)
│   ├── vision/yolo_kd_train.py  # YOLO full training loop
│   ├── runners/run_nlp.py    # BERT CLI entry point
│   └── runners/run_vision.py # YOLO CLI entry point
├── experiments/
│   ├── configs/              # TOML/YAML experiment configs
│   └── scripts/              # Training and evaluation scripts
├── cw/                       # Coursework tests
│   ├── unit/                 # Unit tests (fast, no GPU)
│   ├── integration/          # Smoke tests (CPU, toy models)
│   └── regression/           # Artifact regression tests
└── outputs/                  # Experiment outputs (created at runtime)
```

---

## 3. BERT Experiments

### Teacher–Student Pair

| Model | Source | Params | SST-2 Accuracy |
|---|---|---|---|
| **Teacher** | `textattack/bert-base-uncased-SST-2` | 110 M | ~93% |
| **Student** | Custom BERT (4-layer, 256-dim) | ~13 M | to measure |

The teacher is a BERT-base already fine-tuned on SST-2 (no additional training needed). The student is trained from scratch using either hard labels only (baseline) or combined hard + soft KD labels.

### 3.1 Baseline Training

Train the student with cross-entropy only (no KD):

```bash
# From the repo root
python experiments/scripts/run_bert_baseline.py
```

Or with CLI overrides:
```bash
python experiments/scripts/run_bert_baseline.py \
    --epochs 5 \
    --batch-size 32 \
    --output-dir outputs/bert_baseline
```

Outputs saved to `outputs/bert_baseline/`:
- `best_student/` — HuggingFace checkpoint of the best epoch
- `training_history.json` — per-epoch loss and accuracy

### 3.2 KD Training

Train the student with soft KD from the teacher:

```bash
python experiments/scripts/run_bert_kd.py
```

Key KD hyperparameters:
```bash
python experiments/scripts/run_bert_kd.py \
    --alpha 0.5 \          # 0=hard only, 1=soft only, 0.5=balanced
    --temperature 4.0 \    # Higher T → softer teacher distribution
    --output-dir outputs/bert_kd
```

**What alpha controls:**
```
L_total = (1 - alpha) × L_hard + alpha × T² × L_soft
```
- `alpha=0.0` → equivalent to the baseline (pure CE)
- `alpha=1.0` → student only sees teacher's soft targets (no GT labels)
- `alpha=0.5` → balanced (recommended starting point)

### 3.3 BERT CLI Reference

The `run_nlp.py` script can also be used directly:

```bash
# Full KD run
python -m mase_kd.runners.run_nlp \
    --alpha 0.6 \
    --temperature 4.0 \
    --epochs 5 \
    --student-layers 4 \
    --student-hidden 256 \
    --output-dir outputs/bert_kd_t4

# Baseline mode (no KD)
python -m mase_kd.runners.run_nlp --baseline --output-dir outputs/bert_baseline

# Evaluate saved checkpoint
python -m mase_kd.runners.run_nlp \
    --eval-only outputs/bert_kd/best_student \
    --output-dir outputs/bert_kd
```

**Using a pretrained small BERT as student initialisation** (faster convergence):
```bash
python experiments/scripts/run_bert_kd.py \
    --config experiments/configs/bert_kd.toml \
    --output-dir outputs/bert_kd_pretrained_student
```
Or edit `bert_kd.toml`:
```toml
[student]
pretrained_name = "google/bert_uncased_L-4_H-256_A-4"
```

---

## 4. YOLO Experiments

### Teacher–Student Pair

| Model | Source | Params | COCO mAP@50 |
|---|---|---|---|
| **Teacher** | `yolov8m.pt` | 26 M | ~63% |
| **Student** | `yolov8n` (scratch) | 3.2 M | to measure |

### 4.1 Dataset Setup

**Quick test** (built-in Ultralytics 8-image subset):
```bash
# coco8.yaml is bundled with ultralytics — no download needed
# Used by default in configs
```

**Full COCO** (80 classes, ~20 GB):
```bash
# Download via ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.yaml').train(data='coco.yaml', epochs=1)"
# This downloads COCO to ~/.cache/ultralytics/datasets/coco/
# Then use --data coco.yaml in all scripts
```

**Custom dataset** (COCO-format YAML):
```yaml
# my_dataset.yaml
path: /data/my_dataset
train: images/train
val: images/val
nc: 3
names: ['cat', 'dog', 'bird']
```

### 4.2 Baseline Training

Train YOLOv8n from scratch without KD (pure Ultralytics v8DetectionLoss):

```bash
python experiments/scripts/run_yolo_baseline.py
```

Override options:
```bash
python experiments/scripts/run_yolo_baseline.py \
    --data coco.yaml \
    --epochs 100 \
    --batch-size 16 \
    --output-dir outputs/yolo_baseline_coco
```

### 4.3 KD Training

Distil YOLOv8m teacher into YOLOv8n student:

```bash
python experiments/scripts/run_yolo_kd.py
```

The training loss combines:
- **Hard loss** `(1-alpha)`: Ultralytics v8DetectionLoss (classification + bbox + DFL)
- **Soft loss** `alpha`: Temperature-scaled KL divergence on raw detection head outputs

```bash
python experiments/scripts/run_yolo_kd.py \
    --alpha 0.5 \
    --temperature 2.0 \
    --data coco.yaml \
    --epochs 100 \
    --output-dir outputs/yolo_kd_coco
```

**Key note on teacher forward mode:** During training, both student and teacher run in `train()` mode so the detection head returns raw logits (pre-NMS) rather than post-processed bounding boxes. This is required for the soft KD loss to be meaningful.

**Memory tips for 8–12 GB GPU:**
```bash
# Reduce image size and batch size
python experiments/scripts/run_yolo_kd.py \
    --imgsz 416 \
    --batch-size 8
# AMP is on by default — disable only if you see NaN losses:
# Edit yolo_kd.yaml: use_amp: false
```

---

## 5. Ablation Studies

The `ablation_sweep.py` script trains a grid of (alpha, temperature) combinations and reports the best metric for each.

### BERT Ablation

```bash
python experiments/scripts/ablation_sweep.py \
    --task bert \
    --alphas 0.0 0.3 0.5 0.7 1.0 \
    --temperatures 2.0 4.0 6.0 \
    --epochs 5 \
    --output-dir outputs/ablation_bert
```

Results are saved to `outputs/ablation_bert/bert_ablation_summary.json` and printed as a table:

```
   Alpha    Temp    Accuracy
────────────────────────────
    0.00    2.00      0.8142  ← baseline
    0.30    2.00      0.8356
    0.50    2.00      0.8421
    0.50    4.00      0.8498  ← often best
    0.70    6.00      0.8312
    1.00    6.00      0.7900
```

### YOLO Ablation

```bash
python experiments/scripts/ablation_sweep.py \
    --task yolo \
    --alphas 0.0 0.3 0.5 0.7 \
    --temperatures 2.0 4.0 \
    --data coco8.yaml \
    --yolo-epochs 30 \
    --output-dir outputs/ablation_yolo
```

### Interpreting Results

| alpha | Effect |
|---|---|
| 0.0 | Baseline — no KD, only hard-label task loss |
| 0.3–0.5 | Balanced — good starting point for most tasks |
| 0.7–0.9 | Heavy KD — works well when teacher quality is high |
| 1.0 | Soft-only — risky; student has no GT supervision |

| temperature | Effect |
|---|---|
| 1.0–2.0 | Peaked soft labels (close to hard labels) |
| 3.0–6.0 | Smoother distributions, more "dark knowledge" |
| > 8.0 | Very uniform; diminishing returns |

---

## 6. Evaluation & Comparison

After running baseline and KD experiments, generate a comparison table:

### BERT Comparison

```bash
python experiments/scripts/evaluate_all.py \
    --task bert \
    --baseline-dir outputs/bert_baseline \
    --kd-dir outputs/bert_kd \
    --output outputs/bert_comparison.json
```

Prints:
```
Metric                  Baseline      KD Student       Delta
─────────────────────────────────────────────────────────────
Accuracy                  0.8142          0.8498      +0.0356
F1 (macro)                0.8130          0.8490      +0.0360
Parameters            13,107,202      13,107,202      +0
Latency (ms)               12.40           12.40      +0.00
```

Also saves a Markdown table to `outputs/bert_comparison.md`.

### YOLO Comparison

```bash
python experiments/scripts/evaluate_all.py \
    --task yolo \
    --baseline-dir outputs/yolo_baseline \
    --kd-dir outputs/yolo_kd \
    --data coco8.yaml \
    --output outputs/yolo_comparison.json
```

### Standalone Evaluation of a Saved Checkpoint

```bash
# BERT
python -m mase_kd.runners.run_nlp \
    --eval-only outputs/bert_kd/best_student \
    --output-dir outputs/bert_kd

# YOLO: use YOLOKDRunner.evaluate() or Ultralytics val directly
```

---

## 7. Running Tests

Tests live in `cw/` and follow the project structure from `my_docs/project_structure.md`.

### Fast unit tests (no GPU, no datasets)

```bash
pytest cw/unit/ -v
```

### Integration smoke tests (CPU, toy models)

```bash
pytest cw/integration/ -v -m integration
```

Skip slow integration tests:
```bash
pytest cw/ -m "not integration" -v
```

### Regression tests

```bash
pytest cw/regression/ -v
```

### All coursework tests

```bash
pytest cw/ -v
```

### Single test

```bash
pytest cw/unit/test_kd_losses.py::TestSoftLogitKLLoss::test_zero_when_identical -v
```

### Full MASE test suite

```bash
make test-sw
```

---

## 8. Config Reference

### BERT Config (`experiments/configs/bert_kd.toml`)

| Key | Type | Default | Description |
|---|---|---|---|
| `teacher.model_name` | str | `textattack/bert-base-uncased-SST-2` | HuggingFace teacher checkpoint |
| `student.num_hidden_layers` | int | 4 | Transformer layers in student |
| `student.hidden_size` | int | 256 | Hidden dimension |
| `student.num_attention_heads` | int | 4 | Attention heads |
| `student.pretrained_name` | str\|null | null | HF checkpoint to init student |
| `kd.alpha` | float [0,1] | 0.5 | KD mixing weight |
| `kd.temperature` | float >0 | 4.0 | Softmax temperature |
| `training.batch_size` | int | 32 | Batch size |
| `training.learning_rate` | float | 2e-5 | AdamW LR |
| `training.num_epochs` | int | 5 | Training epochs |
| `training.seed` | int | 42 | Random seed |

### YOLO Config (`experiments/configs/yolo_kd.yaml`)

| Key | Type | Default | Description |
|---|---|---|---|
| `teacher_weights` | str | `yolov8m.pt` | Ultralytics teacher checkpoint |
| `student_arch` | str | `yolov8n.yaml` | Student architecture YAML |
| `student_weights` | str\|null | null | Init student from these weights |
| `data_yaml` | str | `coco8.yaml` | Dataset config |
| `kd.alpha` | float [0,1] | 0.5 | KD mixing weight |
| `kd.temperature` | float >0 | 2.0 | Softmax temperature |
| `training.epochs` | int | 50 | Training epochs |
| `training.batch_size` | int | 16 | Batch size |
| `training.imgsz` | int | 640 | Input image size |
| `training.use_amp` | bool | true | Mixed-precision training |

---

## 9. Common Issues

### `CUDA out of memory`

- **BERT**: Reduce `batch_size` (try 8 or 16), reduce `max_seq_length` to 64.
- **YOLO**: Reduce `imgsz` to 416, reduce `batch_size` to 8, ensure `use_amp: true`.

### `FileNotFoundError: yolov8m.pt` or `yolov8n.yaml`

These files are downloaded automatically by Ultralytics on first use. Ensure internet access, or pre-download:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### `ValueError: No shared logits dimension` (YOLO KD)

The teacher is running in inference (post-NMS) mode instead of training mode. This shouldn't happen with `YOLOKDRunner`, but if you see it with custom models, ensure teacher is in `.train()` mode during distillation.

### `HuggingFace download timeout`

Set HF cache or use offline mode:
```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Ablation sweep is slow

Reduce `--epochs` or sweep a smaller grid first:
```bash
python experiments/scripts/ablation_sweep.py \
    --task bert \
    --alphas 0.0 0.5 \
    --temperatures 2.0 4.0 \
    --epochs 2
```

### Test discovery issues

Ensure `src/` is on `PYTHONPATH`. The `cw/conftest.py` does this automatically when running `pytest cw/`. For running individual scripts, add it manually:
```bash
PYTHONPATH=src pytest cw/unit/ -v
```
