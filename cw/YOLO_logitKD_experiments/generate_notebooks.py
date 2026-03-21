#!/usr/bin/env python3
"""Generate all 26 experiment notebooks for P3 YOLO logit KD sweep."""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

# ── Shared cell source fragments ──────────────────────────────────────────────

IMPORTS_CELL = """\
import copy
import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/")

repo_root = find_repo_root(Path.cwd().resolve())
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from chop import MaseGraph
import chop.passes as passes
from chop.models.yolo.yolov8 import MaseYoloClassificationModel, patch_yolo

from mase_kd.core.losses import DistillationLossConfig, hard_label_ce_loss
from mase_kd.vision.yolo_kd import YOLOLogitsDistiller

if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"

torch.manual_seed(SEED)
random.seed(SEED)

print(f"Repo root: {repo_root}")
print(f"Device:    {DEVICE}")
"""

DATALOADER_CELL = """\
cifar_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True,  transform=cifar_transform, download=True)
val_dataset   = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=cifar_transform, download=True)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, drop_last=True,
)

print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
"""

TEACHER_CELL = """\
ultra_teacher = YOLO(TEACHER_CHECKPOINT)
nc = ultra_teacher.model.yaml.get("nc", 10)

teacher_cls_model = MaseYoloClassificationModel(cfg=TEACHER_CFG, nc=nc)
teacher_cls_model = patch_yolo(teacher_cls_model)
teacher_cls_model.load_state_dict(ultra_teacher.model.state_dict())
teacher_cls_model = teacher_cls_model.to(DEVICE)
teacher_cls_model.eval()

print(f"Teacher loaded: {TEACHER_CHECKPOINT}  (nc={nc})")
"""

PRUNE_CELL = """\
ultra_student = YOLO(STUDENT_CHECKPOINT)

student_seed = MaseYoloClassificationModel(cfg=STUDENT_CFG, nc=nc)
student_seed = patch_yolo(student_seed)
student_seed.load_state_dict(ultra_student.model.state_dict())

mg = MaseGraph(student_seed)
mg, _ = passes.init_metadata_analysis_pass(mg)

trace_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
placeholder_names = [n.name for n in mg.fx_graph.nodes if n.op == "placeholder"]
dummy_in = {name: trace_input for name in placeholder_names}

mg, _ = passes.add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": dummy_in})

pruning_config = {
    "weight":     {"sparsity": PRUNE_SPARSITY, "method": "l1-norm", "scope": "local"},
    "activation": {"sparsity": PRUNE_SPARSITY, "method": "l1-norm", "scope": "local"},
}
mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)

student_cls_model = mg.model.to(DEVICE)
pruned_snapshot = copy.deepcopy(student_cls_model).to(DEVICE)
pruned_snapshot.eval()

print(f"Student loaded: {STUDENT_CHECKPOINT}")
print(f"Pruning complete ({PRUNE_SPARSITY*100:.0f}% sparsity)")
"""

EVAL_FN_AND_PRUNED_CELL = """\
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    batches = samples = correct = 0
    total_ce = total_ms = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits = YOLOLogitsDistiller._extract_logits_with_batch(outputs, images.shape[0])
        if logits is None or logits.numel() == 0:
            continue
        total_ms += (t1 - t0) * 1000.0
        batches += 1
        samples += images.shape[0]
        if logits.shape[1] > int(labels.max().item()):
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_ce += hard_label_ce_loss(logits, labels).item()
    return {
        "top1_acc": correct / max(samples, 1),
        "avg_ce_loss": total_ce / max(batches, 1),
        "avg_forward_ms": total_ms / max(batches, 1),
        "samples": samples,
        "batches": batches,
    }

pruned_metrics = evaluate_model(pruned_snapshot, val_loader, DEVICE)
print(f"Pruned (no train) — top1: {pruned_metrics['top1_acc']*100:.2f}%  CE: {pruned_metrics['avg_ce_loss']:.4f}")
"""


# ── Helper functions ──────────────────────────────────────────────────────────

def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source: str):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None}


def make_config_cell(exp_id, alpha=None, temp=None):
    lines = [
        '# ── Configuration ──────────────────────────────────────────────────────────────',
        f'EXPERIMENT_ID = "{exp_id}"',
        'PAIR = "P3"',
        '',
        'TEACHER_CHECKPOINT = "data/cifar10_yolov8x_cls/runs/yolov8x_cls_cifar10_finetune/weights/best.pt"',
        'STUDENT_CHECKPOINT = "data/cifar10_yolov8m_cls/runs/yolov8m_cls_cifar10_finetune/weights/best.pt"',
        'TEACHER_CFG = "yolov8x-cls.yaml"',
        'STUDENT_CFG = "yolov8m-cls.yaml"',
        '',
        'DEVICE = "cuda"',
        'DATA_ROOT = "./data"',
        '',
        'IMAGE_SIZE = 32',
        'BATCH_SIZE = 128',
        '',
        'PRUNE_SPARSITY = 0.50',
    ]
    if alpha is not None:
        lines.append(f'KD_ALPHA = {alpha}')
        lines.append(f'KD_TEMPERATURE = {temp}')
    lines += [
        'EPOCHS = 5',
        'LR = 1e-4',
        '',
        'SEED = 42',
        '',
        'RESULTS_CSV = "YOLO_logitKD_experiments/results.csv"',
    ]
    if alpha is None:
        lines.append(f'PRUNED_FINETUNED_SAVE_PATH = "data/best_pruned_finetuned_{exp_id}.pt"')
    else:
        lines.append(f'KD_SAVE_PATH = "data/best_student_{exp_id}.pt"')
    return code("\n".join(lines))


def fmt(v):
    if v == int(v):
        return f"{int(v)}.0" if v != 1 else "1.0"
    return f"{v}"


# ── Purpose text for each (alpha, temp) ──────────────────────────────────────

ALPHA_DESC = {
    0.3: "low soft-loss weight (\u03b1=0.3) \u2014 student relies mostly on hard labels",
    0.5: "balanced soft/hard loss (\u03b1=0.5)",
    0.7: "high soft-loss weight (\u03b1=0.7) \u2014 student favours teacher\u2019s dark knowledge",
    0.9: "near-pure soft-target regime (\u03b1=0.9) \u2014 minimal hard-label influence",
    1.0: "pure soft-target distillation (\u03b1=1.0) \u2014 no hard labels, student learns entirely from teacher",
}
TEMP_DESC = {
    1.0:  "T=1.0 is a degenerate baseline \u2014 teacher softmax approaches hard argmax, providing minimal dark knowledge.",
    2.0:  "T=2.0 provides moderate softening of teacher probabilities.",
    4.0:  "T=4.0 is in the expected sweet spot for 10-class CIFAR10.",
    8.0:  "T=8.0 spreads probability mass more evenly across classes.",
    16.0: "T=16.0 maximally surfaces inter-class similarity structure (fine-grained dark knowledge).",
}


def make_setup_md_kd(exp_num, alpha, temp):
    a = fmt(alpha)
    t = fmt(temp)
    ce_w = 1.0 - alpha
    if alpha < 1.0:
        loss_eq = (
            rf"$$\mathcal{{L}} = {a} \cdot T^2 \cdot "
            rf"\text{{KL}}\!\left(\sigma\!\left(\frac{{z_s}}{{T}}\right) "
            rf"\,\|\, \sigma\!\left(\frac{{z_t}}{{T}}\right)\right) + "
            rf"{ce_w:.1f} \cdot \text{{CE}}(z_s, y)$$"
        )
    else:
        loss_eq = (
            rf"$$\mathcal{{L}} = T^2 \cdot "
            rf"\text{{KL}}\!\left(\sigma\!\left(\frac{{z_s}}{{T}}\right) "
            rf"\,\|\, \sigma\!\left(\frac{{z_t}}{{T}}\right)\right)$$"
        )
    temp_short = TEMP_DESC[temp].split(' \u2014 ')[0].lower()
    purpose = f"Test logits distillation with {ALPHA_DESC[alpha]} and {temp_short}.  \n{TEMP_DESC[temp]}"
    return md(
        f"# Experiment {exp_num:02d} \u2014 P3 KD: \u03b1={a}, T={t}\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Item | Value |\n"
        "|------|-------|\n"
        "| **Pair** | P3 \u2014 `yolov8x-cls` (teacher) \u2192 `yolov8m-cls` (student) |\n"
        "| **Dataset** | CIFAR10 (50 000 train / 10 000 val, 32\u00d732, no resize) |\n"
        "| **Pruning** | 50% unstructured magnitude (L1-norm, local scope) |\n"
        f"| **KD \u03b1** | {a} (soft-loss weight) |\n"
        f"| **KD T** | {t} (temperature) |\n"
        "| **Epochs** | 5 |\n"
        "| **Optimizer** | Adam, lr=1e-4 |\n"
        "| **Batch size** | 128 |\n"
        "| **Seed** | 42 |\n"
        "\n"
        "### Purpose\n"
        "\n"
        f"{purpose}\n"
        "\n"
        "### Loss function\n"
        "\n"
        f"{loss_eq}\n"
        "\n"
        "### Conditions evaluated\n"
        "\n"
        "1. **Teacher** \u2014 CIFAR10 fine-tuned `yolov8x-cls`, untouched  \n"
        "2. **Pruned (no train)** \u2014 Student immediately after 50% pruning  \n"
        "3. **Distilled (KD)** \u2014 Pruned student trained 5 epochs with KD loss  \n"
        "\n"
        "CE-only baseline is recorded in `exp_00_P3_baseline_CE.ipynb`."
    )


def make_kd_cell(alpha, temp):
    a, t = fmt(alpha), fmt(temp)
    return code(
        "kd_config = DistillationLossConfig(alpha=KD_ALPHA, temperature=KD_TEMPERATURE)\n"
        "optimizer = torch.optim.Adam(student_cls_model.parameters(), lr=LR)\n"
        "\n"
        "distiller = YOLOLogitsDistiller(\n"
        "    teacher=teacher_cls_model,\n"
        "    student=student_cls_model,\n"
        "    kd_config=kd_config,\n"
        "    device=DEVICE,\n"
        "    train_loader=train_loader,\n"
        "    val_loader=val_loader,\n"
        "    optimizer=optimizer,\n"
        "    num_train_epochs=EPOCHS,\n"
        "    eval_teacher=True,\n"
        ")\n"
        "\n"
        "train_history = distiller.train(save_path=KD_SAVE_PATH)\n"
        'loss_history = train_history["total_loss"]\n'
        'top1_history = train_history["top1_acc"]\n'
        'top5_history = train_history["top5_acc"]\n'
        "\n"
        "# Restore best student weights before evaluation\n"
        "student_cls_model.load_state_dict(torch.load(KD_SAVE_PATH), strict=False)\n"
        'print(f"Best student weights restored from {KD_SAVE_PATH}")'
    )


def make_kd_eval_cell(alpha, temp):
    a, t = fmt(alpha), fmt(temp)
    return code(
        "eval_results = distiller.evaluate()\n"
        'teacher_metrics = eval_results.get("teacher")\n'
        'student_metrics = eval_results["student"]\n'
        'val_kd_loss = eval_results["val_kd_loss"]\n'
        'kd_batches = eval_results["kd_batches"]\n'
        "\n"
        "print(f\"{'Model':<40} {'Top-1':>8} {'CE Loss':>10}\")\n"
        "print(f\"{'─'*40} {'─'*8} {'─'*10}\")\n"
        "if teacher_metrics:\n"
        "    print(f\"{'Teacher (yolov8x-cls)':<40} {teacher_metrics['top1_acc']*100:>7.2f}% {teacher_metrics['avg_ce_loss']:>10.4f}\")\n"
        "print(f\"{'Pruned (no train, 50%)':<40} {pruned_metrics['top1_acc']*100:>7.2f}% {pruned_metrics['avg_ce_loss']:>10.4f}\")\n"
        f"print(f\"{{f'Distilled (\u03b1={a}, T={t})':<40}} {{student_metrics['top1_acc']*100:>7.2f}}% {{student_metrics['avg_ce_loss']:>10.4f}}\")\n"
        'print(f"\\nVal KD loss: {val_kd_loss:.6f} ({kd_batches} batches)")'
    )


def make_kd_csv_cell():
    return code(
        "row = {\n"
        '    "pair": PAIR,\n'
        '    "alpha": KD_ALPHA,\n'
        '    "temperature": KD_TEMPERATURE,\n'
        '    "lr": LR,\n'
        '    "epochs": EPOCHS,\n'
        '    "sparsity": PRUNE_SPARSITY,\n'
        '    "batch_size": BATCH_SIZE,\n'
        '    "seed": SEED,\n'
        '    "teacher_top1": round(teacher_metrics["top1_acc"] * 100, 4) if teacher_metrics else "",\n'
        '    "pruned_top1": round(pruned_metrics["top1_acc"] * 100, 4),\n'
        '    "ce_only_top1": "",\n'
        '    "kd_top1": round(student_metrics["top1_acc"] * 100, 4),\n'
        '    "kd_gain_vs_ce": "",\n'
        '    "teacher_ce_loss": round(teacher_metrics["avg_ce_loss"], 6) if teacher_metrics else "",\n'
        '    "pruned_ce_loss": round(pruned_metrics["avg_ce_loss"], 6),\n'
        '    "ce_only_ce_loss": "",\n'
        '    "kd_ce_loss": round(student_metrics["avg_ce_loss"], 6),\n'
        '    "val_kd_loss": round(val_kd_loss, 6),\n'
        '    "notebook": EXPERIMENT_ID,\n'
        "}\n"
        "\n"
        "csv_path = Path(RESULTS_CSV)\n"
        "file_exists = csv_path.exists() and csv_path.stat().st_size > 0\n"
        'with open(csv_path, "a", newline="") as f:\n'
        "    writer = csv.DictWriter(f, fieldnames=row.keys())\n"
        "    if not file_exists:\n"
        "        writer.writeheader()\n"
        "    writer.writerow(row)\n"
        "\n"
        'print(f"Results appended to {RESULTS_CSV}")'
    )


def make_kd_plot_cell():
    return code(
        "import matplotlib.pyplot as plt\n"
        "import math\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        'fig.suptitle(f"{EXPERIMENT_ID} — KD α={KD_ALPHA}, T={KD_TEMPERATURE} (P3, 50% pruned)", fontsize=13)\n'
        "\n"
        "batches_per_epoch = len(loss_history) // EPOCHS\n"
        "\n"
        "ax = axes[0]\n"
        "ax.plot(loss_history, alpha=0.8, linewidth=0.8)\n"
        "for e in range(1, EPOCHS):\n"
        '    ax.axvline(e * batches_per_epoch, color="gray", ls="--", lw=0.6, alpha=0.5)\n'
        'ax.set_xlabel("Batch")\n'
        'ax.set_ylabel("KD Loss")\n'
        'ax.set_title("Training Loss")\n'
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "ax = axes[1]\n"
        'clean_top1 = [v if not math.isnan(v) else float("nan") for v in top1_history]\n'
        "ax.plot(clean_top1, alpha=0.8, linewidth=0.8)\n"
        "for e in range(1, EPOCHS):\n"
        '    ax.axvline(e * batches_per_epoch, color="gray", ls="--", lw=0.6, alpha=0.5)\n'
        'ax.set_xlabel("Batch")\n'
        'ax.set_ylabel("Top-1 Accuracy")\n'
        'ax.set_title("Training Top-1")\n'
        'ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))\n'
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


# ── Build CE-only baseline (exp_00) ──────────────────────────────────────────

def build_baseline():
    cells = [
        md(
            "# Experiment 00 \u2014 P3 Baseline: CE-Only Fine-Tuning (No Distillation)\n"
            "\n"
            "## Setup\n"
            "\n"
            "| Item | Value |\n"
            "|------|-------|\n"
            "| **Pair** | P3 \u2014 `yolov8x-cls` (teacher) \u2192 `yolov8m-cls` (student) |\n"
            "| **Dataset** | CIFAR10 (50 000 train / 10 000 val, 32\u00d732, no resize) |\n"
            "| **Pruning** | 50% unstructured magnitude (L1-norm, local scope) |\n"
            "| **Training** | CE-only (\u03b1=0.0, no teacher signal) |\n"
            "| **Epochs** | 5 |\n"
            "| **Optimizer** | Adam, lr=1e-4 |\n"
            "| **Batch size** | 128 |\n"
            "| **Seed** | 42 |\n"
            "\n"
            "### Purpose\n"
            "\n"
            "This notebook establishes the **CE-only fine-tuning baseline** for pair P3.  \n"
            "The pruned `yolov8m-cls` student is trained using only hard-label cross-entropy loss  \n"
            "(no teacher supervision). All KD experiment notebooks (`exp_01`\u2013`exp_25`) compare  \n"
            "against this baseline to measure the incremental benefit of knowledge distillation.\n"
            "\n"
            "### Conditions evaluated\n"
            "\n"
            "1. **Teacher** \u2014 CIFAR10 fine-tuned `yolov8x-cls`, untouched  \n"
            "2. **Pruned (no train)** \u2014 Student immediately after 50% pruning, no further training  \n"
            "3. **Fine-tuned (CE only)** \u2014 Pruned student trained 5 epochs with CE loss"
        ),
        make_config_cell("exp_00"),
        md("## Imports and setup"),
        code(IMPORTS_CELL),
        md("## CIFAR10 dataloaders"),
        code(DATALOADER_CELL),
        md("## Load CIFAR10-fine-tuned teacher (yolov8x-cls)"),
        code(TEACHER_CELL),
        md("## Build student from checkpoint and prune (50% sparsity)"),
        code(PRUNE_CELL),
        md("## Evaluate pruned model (no training)"),
        code(EVAL_FN_AND_PRUNED_CELL),
        md("## Fine-tune pruned student with CE loss only (no distillation)"),
        code(
            "ce_model = copy.deepcopy(pruned_snapshot).to(DEVICE)\n"
            "ce_optimizer = torch.optim.Adam(ce_model.parameters(), lr=LR)\n"
            "\n"
            "ce_loss_history = []\n"
            "ce_top1_history = []\n"
            "best_val_top1 = 0.0\n"
            "\n"
            "for epoch in range(1, EPOCHS + 1):\n"
            "    ce_model.train()\n"
            "    epoch_loss = 0.0\n"
            "    num_batches = len(train_loader)\n"
            "    for batch_idx, (images, labels) in enumerate(train_loader, start=1):\n"
            "        images, labels = images.to(DEVICE), labels.to(DEVICE)\n"
            "        ce_optimizer.zero_grad(set_to_none=True)\n"
            "        outputs = ce_model(images)\n"
            "        logits = YOLOLogitsDistiller._extract_logits_with_batch(outputs, images.shape[0])\n"
            "        loss = hard_label_ce_loss(logits, labels)\n"
            "        loss.backward()\n"
            "        ce_optimizer.step()\n"
            "        epoch_loss += loss.item()\n"
            "\n"
            "        with torch.no_grad():\n"
            "            top1 = float((logits.argmax(dim=1) == labels).float().mean())\n"
            "        ce_loss_history.append(loss.item())\n"
            "        ce_top1_history.append(top1)\n"
            "\n"
            "        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_batches:\n"
            '            print(f"  Epoch {epoch}/{EPOCHS} | Batch {batch_idx:04d}/{num_batches} | "\n'
            '                  f"loss={loss.item():.6f} | top1={top1*100:.2f}%")\n'
            '    print(f"Epoch {epoch} avg loss: {epoch_loss / num_batches:.6f}")\n'
            "\n"
            "    # Save best checkpoint per epoch\n"
            "    epoch_val = evaluate_model(ce_model, val_loader, DEVICE)\n"
            "    if epoch_val['top1_acc'] > best_val_top1:\n"
            "        best_val_top1 = epoch_val['top1_acc']\n"
            "        torch.save(ce_model.state_dict(), PRUNED_FINETUNED_SAVE_PATH)\n"
            "        print(f\"  [checkpoint] val top1={best_val_top1*100:.2f}% \u2014 saved to {PRUNED_FINETUNED_SAVE_PATH}\")\n"
            "\n"
            "# Restore best weights and evaluate\n"
            "ce_model.load_state_dict(torch.load(PRUNED_FINETUNED_SAVE_PATH), strict=False)\n"
            "ce_metrics = evaluate_model(ce_model, val_loader, DEVICE)\n"
            "print(f\"\\nCE-only fine-tuned (best) \u2014 top1: {ce_metrics['top1_acc']*100:.2f}%  CE: {ce_metrics['avg_ce_loss']:.4f}\")"
        ),
        md("## Evaluate teacher"),
        code(
            "teacher_metrics = evaluate_model(teacher_cls_model, val_loader, DEVICE)\n"
            "print(f\"Teacher \u2014 top1: {teacher_metrics['top1_acc']*100:.2f}%  CE: {teacher_metrics['avg_ce_loss']:.4f}\")"
        ),
        md("## Summary"),
        code(
            "print(f\"{'Model':<40} {'Top-1':>8} {'CE Loss':>10}\")\n"
            "print(f\"{'─'*40} {'─'*8} {'─'*10}\")\n"
            "print(f\"{'Teacher (yolov8x-cls)':<40} {teacher_metrics['top1_acc']*100:>7.2f}% {teacher_metrics['avg_ce_loss']:>10.4f}\")\n"
            "print(f\"{'Pruned (no train, 50%)':<40} {pruned_metrics['top1_acc']*100:>7.2f}% {pruned_metrics['avg_ce_loss']:>10.4f}\")\n"
            "print(f\"{'CE-only (5 epochs)':<40} {ce_metrics['top1_acc']*100:>7.2f}% {ce_metrics['avg_ce_loss']:>10.4f}\")"
        ),
        md("## Save results to CSV"),
        code(
            "row = {\n"
            '    "pair": PAIR,\n'
            '    "alpha": 0.0,\n'
            '    "temperature": "",\n'
            '    "lr": LR,\n'
            '    "epochs": EPOCHS,\n'
            '    "sparsity": PRUNE_SPARSITY,\n'
            '    "batch_size": BATCH_SIZE,\n'
            '    "seed": SEED,\n'
            '    "teacher_top1": round(teacher_metrics["top1_acc"] * 100, 4),\n'
            '    "pruned_top1": round(pruned_metrics["top1_acc"] * 100, 4),\n'
            '    "ce_only_top1": round(ce_metrics["top1_acc"] * 100, 4),\n'
            '    "kd_top1": "",\n'
            '    "kd_gain_vs_ce": "",\n'
            '    "teacher_ce_loss": round(teacher_metrics["avg_ce_loss"], 6),\n'
            '    "pruned_ce_loss": round(pruned_metrics["avg_ce_loss"], 6),\n'
            '    "ce_only_ce_loss": round(ce_metrics["avg_ce_loss"], 6),\n'
            '    "kd_ce_loss": "",\n'
            '    "val_kd_loss": "",\n'
            '    "notebook": EXPERIMENT_ID,\n'
            "}\n"
            "\n"
            "csv_path = Path(RESULTS_CSV)\n"
            "file_exists = csv_path.exists() and csv_path.stat().st_size > 0\n"
            'with open(csv_path, "a", newline="") as f:\n'
            "    writer = csv.DictWriter(f, fieldnames=row.keys())\n"
            "    if not file_exists:\n"
            "        writer.writeheader()\n"
            "    writer.writerow(row)\n"
            "\n"
            'print(f"Results appended to {RESULTS_CSV}")'
        ),
        md("## Training curves"),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            'fig.suptitle(f"{EXPERIMENT_ID} — CE-Only Fine-Tuning (P3, 50% pruned)", fontsize=13)\n'
            "\n"
            "batches_per_epoch = len(ce_loss_history) // EPOCHS\n"
            "\n"
            "ax = axes[0]\n"
            "ax.plot(ce_loss_history, alpha=0.8, linewidth=0.8)\n"
            "for e in range(1, EPOCHS):\n"
            '    ax.axvline(e * batches_per_epoch, color="gray", ls="--", lw=0.6, alpha=0.5)\n'
            'ax.set_xlabel("Batch")\n'
            'ax.set_ylabel("CE Loss")\n'
            'ax.set_title("Training Loss")\n'
            "ax.grid(True, alpha=0.3)\n"
            "\n"
            "ax = axes[1]\n"
            "ax.plot(ce_top1_history, alpha=0.8, linewidth=0.8)\n"
            "for e in range(1, EPOCHS):\n"
            '    ax.axvline(e * batches_per_epoch, color="gray", ls="--", lw=0.6, alpha=0.5)\n'
            'ax.set_xlabel("Batch")\n'
            'ax.set_ylabel("Top-1 Accuracy")\n'
            'ax.set_title("Training Top-1")\n'
            'ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))\n'
            "ax.grid(True, alpha=0.3)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


# ── Build KD notebook ────────────────────────────────────────────────────────

def build_kd(exp_num, alpha, temp):
    a, t = fmt(alpha), fmt(temp)
    exp_id = f"exp_{exp_num:02d}"
    cells = [
        make_setup_md_kd(exp_num, alpha, temp),
        make_config_cell(exp_id, alpha, temp),
        md("## Imports and setup"),
        code(IMPORTS_CELL),
        md("## CIFAR10 dataloaders"),
        code(DATALOADER_CELL),
        md(f"## Load CIFAR10-fine-tuned teacher (yolov8x-cls)"),
        code(TEACHER_CELL),
        md("## Build student from checkpoint and prune (50% sparsity)"),
        code(PRUNE_CELL),
        md("## Evaluate pruned model (no training)"),
        code(EVAL_FN_AND_PRUNED_CELL),
        md(f"## Knowledge Distillation (\u03b1={a}, T={t})"),
        make_kd_cell(alpha, temp),
        md("## Evaluation: teacher vs pruned vs distilled (CIFAR10 val)"),
        make_kd_eval_cell(alpha, temp),
        md("## Save results to CSV"),
        make_kd_csv_cell(),
        md("## Training curves"),
        make_kd_plot_cell(),
    ]
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


# ── Generate all notebooks ───────────────────────────────────────────────────

def main():
    # exp_00 — CE baseline
    nb = build_baseline()
    path = OUT_DIR / "exp_00_P3_baseline_CE.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Created {path.name}")

    # exp_01 .. exp_25 — KD grid
    grid = []
    for alpha in [0.3, 0.5, 0.7, 0.9, 1.0]:
        for temp in [1.0, 2.0, 4.0, 8.0, 16.0]:
            grid.append((alpha, temp))

    for i, (alpha, temp) in enumerate(grid, start=1):
        a, t = fmt(alpha), fmt(temp)
        exp_id = f"exp_{i:02d}"
        filename = f"{exp_id}_P3_a{a}_T{t}.ipynb"
        nb = build_kd(i, alpha, temp)
        path = OUT_DIR / filename
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"Created {filename}")

    print(f"\nDone — 26 notebooks generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
