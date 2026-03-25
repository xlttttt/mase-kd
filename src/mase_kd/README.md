# mase-kd

Knowledge distillation pipelines built on top of the MASE framework.

---

## YOLOv8

Logits-based knowledge distillation for YOLOv8 classification models using `YOLOLogitsDistiller`.

### Minimal usage

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

from chop.models.yolo.yolov8 import MaseYoloClassificationModel, patch_yolo
from mase_kd.core.losses import DistillationLossConfig
from mase_kd.vision.yolo_kd import YOLOLogitsDistiller

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NC = 10  # number of classes

# --- Dataloaders ---
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.CIFAR10("./data", train=True, transform=transform, download=True),
    batch_size=128, shuffle=True, num_workers=2,
)
val_loader = DataLoader(
    datasets.CIFAR10("./data", train=False, transform=transform),
    batch_size=128, shuffle=False, num_workers=2,
)

# --- Teacher (fine-tuned YOLOv8x-cls checkpoint) ---
ultra_teacher = YOLO("path/to/yolov8x_cls_best.pt")
teacher = MaseYoloClassificationModel(cfg="yolov8x-cls.yaml", nc=NC)
teacher = patch_yolo(teacher)
teacher.load_state_dict(ultra_teacher.model.state_dict())
teacher = teacher.to(DEVICE)
del ultra_teacher

# --- Student (randomly initialised YOLOv8m-cls) ---
student = MaseYoloClassificationModel(cfg="yolov8m-cls.yaml", nc=NC)
student = patch_yolo(student)
student = student.to(DEVICE)

# --- Distillation config ---
kd_config = DistillationLossConfig(alpha=0.7, temperature=4.0)
# alpha:       weight on the soft KD loss  (1-alpha weights the hard CE loss)
# temperature: softens the teacher/student distributions before KL divergence

# --- Distiller ---
optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=0.05)

distiller = YOLOLogitsDistiller(
    teacher=teacher,
    student=student,
    kd_config=kd_config,
    device=DEVICE,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_train_epochs=60,
    eval_teacher=True,
)

# --- Train ---
history = distiller.train(save_path="best_student.pt")
# Saves a checkpoint whenever validation top-1 improves.
# history keys: "train_total_loss", "train_top1_acc", "train_top5_acc",
#               "val_top1_acc", "val_top5_acc", "val_loss"

# Restore best weights before evaluation
student.load_state_dict(torch.load("best_student.pt", weights_only=True), strict=False)

# --- Evaluate ---
results = distiller.evaluate()
# results["student"]  → {"top1_acc", "top5_acc", "avg_ce_loss", ...}
# results["teacher"]  → same keys (when eval_teacher=True)
# results["val_kd_loss"] → scalar KL-divergence loss on validation set

print(f"Teacher  top-1: {results['teacher']['top1_acc']*100:.2f}%")
print(f"Student  top-1: {results['student']['top1_acc']*100:.2f}%")
```

### Key API

| Symbol | Description |
|---|---|
| `DistillationLossConfig(alpha, temperature)` | Hyper-parameters for the KD objective. `alpha` weights the soft KL loss; `1-alpha` weights the hard CE loss. |
| `YOLOLogitsDistiller(teacher, student, kd_config, …)` | Main distillation trainer. Accepts dataloaders and optimizer at construction time. |
| `distiller.train(save_path=…)` | Runs the full epoch loop. Checkpoints the student whenever validation top-1 improves. Returns per-step and per-epoch history dicts. |
| `distiller.evaluate()` | Evaluates teacher and student on `val_loader`. Returns `top1_acc`, `top5_acc`, `avg_ce_loss`, and `val_kd_loss`. |

---

## ResNet18

Logits-based knowledge distillation for ResNet18 on CIFAR-10 / CIFAR-100, with a full A–E experimental pipeline (dense → prune → fine-tune / KD / KD+FT).

### Minimal usage — CLI

```bash
# Smoke run (validates A–E pipeline, ~2 min on CPU, no GPU required)
PYTHONPATH=src python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 --profile smoke --sparsity 0.5

# Full run (single sparsity, reproducible seed)
PYTHONPATH=src python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 --profile full --sparsity 0.85 --seed 0

# CIFAR-100
PYTHONPATH=src python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar100 --profile full --sparsity 0.7 --seed 0
```

Outputs land in `cw/kx725/outputs/resnet18/<dataset>/sparsity_<s>/` and include:
- Per-step `metrics.json` (accuracy, params, sparsity)
- `comparison_table.{md,json}` — side-by-side A–E results
- `trade_off_plot.png` — accuracy vs variant bar chart

### Minimal usage — Python API

```python
import torch
from mase_kd.vision.resnet_kd import (
    build_cifar_resnet18,
    build_resnet_kd_trainer,
    load_cifar10_dataloaders,
)
from mase_kd.core.losses import DistillationLossConfig
from mase_kd.config.schema import ResNetKDConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Build model ---
student = build_cifar_resnet18(num_classes=10)  # CIFAR-friendly (3×3 conv, no maxpool)

# --- Config ---
cfg = ResNetKDConfig(
    teacher_weights="",          # empty → alpha=0 baseline (no teacher needed)
    num_classes=10,
    dataset="cifar10",
    kd=DistillationLossConfig(alpha=0.0, temperature=4.0),
    epochs=2,
    batch_size=128,
    learning_rate=0.1,
    data_dir="datasets/cifar10",
    output_dir="outputs/resnet_demo",
)

# --- Train ---
trainer = build_resnet_kd_trainer(cfg, DEVICE)
trainer.train()                              # saves best_student.pth + training_history.json

# --- Evaluate ---
results = trainer.evaluate("test")          # {"accuracy": float}
print(f"Test accuracy: {results['accuracy']*100:.2f}%")
```

### Knowledge distillation (self-KD from dense checkpoint)

```python
cfg_kd = ResNetKDConfig(
    teacher_weights="outputs/A_dense/best_student.pth",  # dense checkpoint as teacher
    student_weights="outputs/B_pruned/pruned_student.pth",
    num_classes=10,
    dataset="cifar10",
    kd=DistillationLossConfig(alpha=0.5, temperature=4.0),
    # alpha:       weight on soft KL loss  (1-alpha weights the hard CE loss)
    # temperature: softens teacher/student distributions before KL divergence
    epochs=40,
    learning_rate=0.001,  # low LR critical after dense convergence (avoids catastrophic forgetting)
    output_dir="outputs/D_kd",
)
trainer_kd = build_resnet_kd_trainer(cfg_kd, DEVICE)
trainer_kd.train()
```

### Full A–E pipeline (programmatic)

```python
from mase_kd.passes.pipeline import ResNetPipeline
import yaml

with open("cw/kx725/configs/resnet18_cifar10_smoke.yaml") as f:
    config = yaml.safe_load(f)

results = ResNetPipeline().run(
    config=config,
    output_dir="cw/kx725/outputs/resnet18/cifar10",
    sparsity=0.5,
    device=DEVICE,
)
# results = {"A": {...}, "B": {...}, "C": {...}, "D": {...}, "E": {...}}
# Each value: {"accuracy", "val_accuracy", "params_nonzero", "params_total", "sparsity"}
print(f"Dense: {results['A']['accuracy']*100:.2f}%  →  KD+FT: {results['E']['accuracy']*100:.2f}%")
```

### Key API

| Symbol | Description |
|---|---|
| `build_cifar_resnet18(num_classes)` | ResNet18 with CIFAR-friendly first conv (3×3, stride 1, no maxpool). Preserves 32×32 spatial resolution. |
| `ResNetKDConfig(...)` | Full pipeline config: teacher/student weights, dataset, training hyperparameters, output dir. Call `.validate()` before use. |
| `DistillationLossConfig(alpha, temperature)` | KD loss hyper-parameters. `alpha=0` → pure hard-label baseline; `alpha=1` → pure soft KD. |
| `build_resnet_kd_trainer(cfg, device)` | Factory that wires up dataloaders, models, and optimizer from a `ResNetKDConfig`. |
| `ResNetKDTrainer.train()` | Runs the full training loop. Saves `best_student.pth` and `training_history.json` to `cfg.output_dir`. |
| `ResNetKDTrainer.evaluate(split)` | Evaluates on `"val"` or `"test"` split. Returns `{"accuracy": float}`. |
| `PrunePass.run(model, PruneConfig(...), {})` | Global L1 unstructured magnitude pruning. `make_permanent=True` fuses mask into weights. |
| `ResNetPipeline.run(config, output_dir, sparsity, device)` | Orchestrates all five A–E steps. Returns `{A…E}` metrics dict and writes comparison artifacts. |

### A–E experimental matrix

| Step | Dir | Description |
|---|---|---|
| A | `A_dense/` | Train ResNet18 from scratch (alpha=0). Checkpoint reused as KD teacher for D/E. |
| B | `B_pruned/` | Global L1 unstructured pruning applied to A's weights. No recovery training. |
| C | `C_ft/` | Load B's pruned weights, fine-tune with hard labels (alpha=0). |
| D | `D_kd/` | Load B's pruned weights, distil from A's checkpoint (alpha=0.5 by default). |
| E | `E_kd_ft/` | Load D's best checkpoint, fine-tune with hard labels (alpha=0). |

---

## BERT

> _Coming soon._

---

## GPT-2

Logits-based knowledge distillation for GPT-2-style sequence classification on IMDb, using a decoder-only `distilgpt2` student and a GPT-2-family teacher (`mnoukhov/gpt2-imdb-sentiment-classifier`). This track also supports a full A–E experimental pipeline (dense → prune → fine-tune / KD / KD+FT) plus a small KD hyper-parameter sweep.

### Minimal usage — Python API

```python
import torch
from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.gpt2_imdb_kd import (
    GPT2IMDbKDConfig,
    GPT2StudentConfig,
    build_gpt2_imdb_kd_trainer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = GPT2IMDbKDConfig(
    teacher_model_name="mnoukhov/gpt2-imdb-sentiment-classifier",
    student=GPT2StudentConfig(
        model_name="distilgpt2",
        num_labels=2,
    ),
    kd=DistillationLossConfig(alpha=0.5, temperature=4.0),
    max_seq_length=256,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=2,
    warmup_ratio=0.06,
    weight_decay=0.01,
    seed=42,
    train_subset=5000,
    test_subset=2000,
    output_dir="outputs/gpt2_imdb_kd_full",
)

trainer = build_gpt2_imdb_kd_trainer(cfg, device=DEVICE)
trainer.train()                  # saves best checkpoint + training history
results = trainer.evaluate()     # {"val_loss", "val_accuracy", "val_f1"}
print(results)
```

### Minimal usage — CLI

```bash
# Full KD run
PYTHONPATH=src python3 -m experiments.scripts.run_gpt2_imdb_kd_best

# Small KD grid search
PYTHONPATH=src python3 -m experiments.scripts.run_gpt2_imdb_grid_search \
    --config experiments/configs/distilgpt2_imdb_grid_search.yaml
```

Outputs include:
- `metrics.json` for the best-config run
- `results.json`, `results.csv`, and `best.json` for the grid search

### Key API

| Symbol | Description |
|---|---|
| `GPT2StudentConfig(model_name, num_labels)` | Lightweight config for the decoder-only student model. Defaults to `distilgpt2`. |
| `GPT2IMDbKDConfig(...)` | Full training config: teacher/student model names, KD hyper-parameters, IMDb subset sizes, optimiser settings, output dir, and optional `student_weights_path`. |
| `DistillationLossConfig(alpha, temperature)` | KD loss hyper-parameters. `alpha` weights the soft KL loss; `1-alpha` weights the hard CE loss. |
| `build_gpt2_imdb_kd_trainer(cfg, device)` | Factory that wires up tokenizer, dataloaders, teacher, student, and trainer from a `GPT2IMDbKDConfig`. |
| `GPT2IMDbKDTrainer.train()` | Runs the full training loop. Saves the best checkpoint and training history to `cfg.output_dir`. |
| `GPT2IMDbKDTrainer.evaluate()` | Evaluates on the IMDb validation split. Returns `{"val_loss", "val_accuracy", "val_f1"}`. |

### GPT-2-specific implementation notes

- GPT-2 tokenizers do not define a padding token by default, so the EOS token is reused as `pad_token` for batched IMDb classification.
- Pruning targets both `nn.Linear` and Hugging Face `Conv1D` modules; pruning only linear layers leaves much of GPT-2 effectively untouched.
- Zero-mask preservation is used after loading pruned checkpoints so later fine-tuning / KD stages keep the same sparse structure.
- The final teacher/student pairing is GPT-2-family to avoid tokenizer/vocabulary mismatch during teacher forward passes.

### A–E experimental matrix

| Step | Dir | Description |
|---|---|---|
| A | `A_dense/` | Train dense `distilgpt2` on IMDb with hard labels only. |
| B | `B_pruned/` | Apply pruning to A’s checkpoint. No recovery training. |
| C | `C_ft/` | Load B’s sparse checkpoint and fine-tune with hard labels only. |
| D | `D_kd/` | Load B’s sparse checkpoint and distil from the GPT-2 teacher. |
| E | `E_kd_ft/` | Load D’s best checkpoint and fine-tune again with hard labels only. |



