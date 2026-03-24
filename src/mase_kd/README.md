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

> _Coming soon._

---

## BERT

> _Coming soon._

---

## GPT-2

> _Coming soon._
