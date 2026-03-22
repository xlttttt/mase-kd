#!/usr/bin/env python3
"""P3 evaluation — loads all checkpoints from data/P3_grid_search/ and writes results.csv.

Overwrites the existing results.csv with a clean, correctly-aligned file.

Usage:
    cd cw
    python3 YOLO_logitKD_experiments/P3_eval.py
"""

import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────

PAIR = "P3"
TEACHER_CHECKPOINT = "data/cifar10_yolov8x_cls/runs/yolov8x_cls_cifar10_finetune/weights/best.pt"
TEACHER_CFG = "yolov8x-cls.yaml"
STUDENT_CFG = "yolov8m-cls.yaml"

DEVICE = "cuda"
DATA_ROOT = "./data"

IMAGE_SIZE = 32
BATCH_SIZE = 128

PRUNE_SPARSITY = 0.0
EPOCHS = 60
LR = 5e-4
WEIGHT_DECAY = 0.05
SEED = 42

CHECKPOINT_DIR = Path("data/P3_grid_search")
RESULTS_CSV = Path("YOLO_logitKD_experiments/results.csv")

ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]
TEMPERATURES = [1.0, 2.0, 4.0, 8.0, 16.0]

FIELDNAMES = [
    "pair", "alpha", "temperature", "lr", "epochs", "sparsity",
    "batch_size", "seed",
    "teacher_top1", "teacher_top5",
    "pruned_top1", "pruned_top5",
    "ce_only_top1", "ce_only_top5",
    "kd_top1", "kd_top5",
    "kd_gain_vs_ce",
    "teacher_ce_loss", "pruned_ce_loss", "ce_only_ce_loss",
    "kd_ce_loss", "val_kd_loss",
    "notebook",
]

# ── Repo / imports ─────────────────────────────────────────────────────────────


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

from mase_kd.core.losses import (
    DistillationLossConfig,
    hard_label_ce_loss,
    soft_logit_kl_loss,
)
from mase_kd.vision.yolo_kd import YOLOLogitsDistiller

if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"

torch.manual_seed(SEED)
random.seed(SEED)

print(f"Repo root: {repo_root}")
print(f"Device:    {DEVICE}")

# ── Evaluation helper ──────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate a model on the given loader, returning top1/top5/CE metrics."""
    model.eval()
    batches = samples = correct = correct_top5 = 0
    total_ce = total_ms = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()  # Classify head returns raw logits in train mode
        outputs = model(images)
        model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        outputs = YOLOLogitsDistiller._unwrap_classify_output(outputs)
        logits = YOLOLogitsDistiller._extract_logits_with_batch(outputs, images.shape[0])
        if logits is None or logits.numel() == 0:
            continue
        total_ms += (t1 - t0) * 1000.0
        batches += 1
        samples += images.shape[0]
        if logits.shape[1] > int(labels.max().item()):
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            k = min(5, logits.shape[1])
            top_k_indices = logits.topk(k, dim=1).indices
            correct_top5 += int(
                (top_k_indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            )
            total_ce += hard_label_ce_loss(logits, labels).item()
    return {
        "top1_acc": correct / max(samples, 1),
        "top5_acc": correct_top5 / max(samples, 1),
        "avg_ce_loss": total_ce / max(batches, 1),
        "avg_forward_ms": total_ms / max(batches, 1),
        "samples": samples,
        "batches": batches,
    }


@torch.no_grad()
def compute_val_kd_loss(teacher, student, loader, kd_config, device):
    """Compute the average KD loss on the validation set."""
    teacher.eval()
    student.eval()
    total_kd = 0.0
    batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        teacher.train()
        t_out = teacher(images)
        teacher.eval()
        t_out = YOLOLogitsDistiller._unwrap_classify_output(t_out)
        t_logits = YOLOLogitsDistiller._extract_logits_with_batch(t_out, images.shape[0])

        student.train()
        s_out = student(images)
        student.eval()
        s_out = YOLOLogitsDistiller._unwrap_classify_output(s_out)
        s_logits = YOLOLogitsDistiller._extract_logits_with_batch(s_out, images.shape[0])

        if t_logits is None or s_logits is None:
            continue

        kd_loss = soft_logit_kl_loss(s_logits, t_logits, kd_config.temperature)
        total_kd += kd_loss.item()
        batches += 1

    return total_kd / max(batches, 1)


# ── CIFAR10 dataloaders ───────────────────────────────────────────────────────

cifar_transform_eval = transforms.Compose([transforms.ToTensor()])
val_dataset = datasets.CIFAR10(
    root=DATA_ROOT, train=False, transform=cifar_transform_eval, download=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, drop_last=True,
)
print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

# ── Load teacher ───────────────────────────────────────────────────────────────

ultra_teacher = YOLO(TEACHER_CHECKPOINT)
nc = ultra_teacher.model.yaml.get("nc", 10)

teacher_cls_model = MaseYoloClassificationModel(cfg=TEACHER_CFG, nc=nc)
teacher_cls_model = patch_yolo(teacher_cls_model)
teacher_cls_model.load_state_dict(ultra_teacher.model.state_dict())
teacher_cls_model = teacher_cls_model.to(DEVICE)
teacher_cls_model.eval()
del ultra_teacher

print(f"Teacher loaded: {TEACHER_CHECKPOINT}  (nc={nc})")

teacher_metrics = evaluate_model(teacher_cls_model, val_loader, DEVICE)
print(
    f"Teacher — top1: {teacher_metrics['top1_acc']*100:.2f}%  "
    f"top5: {teacher_metrics['top5_acc']*100:.2f}%  "
    f"CE: {teacher_metrics['avg_ce_loss']:.4f}"
)

# ── Build pruned student shell (for loading checkpoints) ──────────────────────

student_shell = MaseYoloClassificationModel(cfg=STUDENT_CFG, nc=nc)
student_shell = patch_yolo(student_shell)

mg = MaseGraph(student_shell)
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
student_shell = mg.model.to(DEVICE)
del mg, trace_input, dummy_in

print("Student shell built (pruned architecture ready for checkpoint loading)")

# Evaluate the pruned model before any training (same for all experiments)
pruned_metrics = evaluate_model(student_shell, val_loader, DEVICE)
print(
    f"Pruned (no train) — top1: {pruned_metrics['top1_acc']*100:.2f}%  "
    f"top5: {pruned_metrics['top5_acc']*100:.2f}%  "
    f"CE: {pruned_metrics['avg_ce_loss']:.4f}"
)

# ── Helper: load a checkpoint into a fresh copy of the student ─────────────────


def load_student_checkpoint(ckpt_path: str | Path):
    """Load checkpoint weights into a fresh copy of the pruned student shell."""
    import copy
    model = copy.deepcopy(student_shell).to(DEVICE)
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    return model


# ── Build experiment grid mapping ──────────────────────────────────────────────

experiments = []

# exp_00 — CE-only baseline
experiments.append({
    "exp_id": "exp_00",
    "alpha": 0.0,
    "temperature": "",
    "checkpoint": CHECKPOINT_DIR / "best_pruned_finetuned_exp_00.pt",
    "is_baseline": True,
})

# exp_01 .. exp_25 — KD grid
exp_num = 0
for alpha in ALPHAS:
    for temp in TEMPERATURES:
        exp_num += 1
        exp_id = f"exp_{exp_num:02d}"
        experiments.append({
            "exp_id": exp_id,
            "alpha": alpha,
            "temperature": temp,
            "checkpoint": CHECKPOINT_DIR / f"best_student_{exp_id}.pt",
            "is_baseline": False,
        })

print(f"\nTotal experiments to evaluate: {len(experiments)}")

# ══════════════════════════════════════════════════════════════════════════════
# Evaluate all checkpoints
# ══════════════════════════════════════════════════════════════════════════════

rows = []
ce_metrics = None  # will be set after evaluating exp_00

for exp in experiments:
    exp_id = exp["exp_id"]
    ckpt = exp["checkpoint"]

    if not ckpt.exists():
        print(f"\n[SKIP] {exp_id}: checkpoint not found at {ckpt}")
        continue

    print(f"\n{'=' * 70}")
    print(f"Evaluating {exp_id} — α={exp['alpha']}, T={exp['temperature']}")
    print(f"{'=' * 70}")

    model = load_student_checkpoint(ckpt)
    metrics = evaluate_model(model, val_loader, DEVICE)

    print(
        f"  top1: {metrics['top1_acc']*100:.2f}%  "
        f"top5: {metrics['top5_acc']*100:.2f}%  "
        f"CE: {metrics['avg_ce_loss']:.4f}"
    )

    if exp["is_baseline"]:
        # CE-only baseline
        ce_metrics = metrics
        val_kd_loss = ""
        kd_top1 = ""
        kd_top5 = ""
        kd_ce_loss = ""
        kd_gain = ""
        row_ce_top1 = round(metrics["top1_acc"] * 100, 4)
        row_ce_top5 = round(metrics["top5_acc"] * 100, 4)
        row_ce_ce_loss = round(metrics["avg_ce_loss"], 6)
    else:
        # KD experiment
        kd_config = DistillationLossConfig(
            alpha=exp["alpha"], temperature=exp["temperature"]
        )
        val_kd_loss = round(
            compute_val_kd_loss(
                teacher_cls_model, model, val_loader, kd_config, DEVICE
            ),
            6,
        )
        kd_top1 = round(metrics["top1_acc"] * 100, 4)
        kd_top5 = round(metrics["top5_acc"] * 100, 4)
        kd_ce_loss = round(metrics["avg_ce_loss"], 6)
        kd_gain = round(
            metrics["top1_acc"] * 100 - ce_metrics["top1_acc"] * 100, 4
        ) if ce_metrics else ""
        # Re-use CE baseline metrics for these columns
        row_ce_top1 = round(ce_metrics["top1_acc"] * 100, 4) if ce_metrics else ""
        row_ce_top5 = round(ce_metrics["top5_acc"] * 100, 4) if ce_metrics else ""
        row_ce_ce_loss = round(ce_metrics["avg_ce_loss"], 6) if ce_metrics else ""
        print(f"  KD gain vs CE: {kd_gain:+.2f}%" if isinstance(kd_gain, float) else "")

    row = {
        "pair": PAIR,
        "alpha": exp["alpha"],
        "temperature": exp["temperature"],
        "lr": LR,
        "epochs": EPOCHS,
        "sparsity": PRUNE_SPARSITY,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "teacher_top1": round(teacher_metrics["top1_acc"] * 100, 4),
        "teacher_top5": round(teacher_metrics["top5_acc"] * 100, 4),
        "pruned_top1": round(pruned_metrics["top1_acc"] * 100, 4),
        "pruned_top5": round(pruned_metrics["top5_acc"] * 100, 4),
        "ce_only_top1": row_ce_top1,
        "ce_only_top5": row_ce_top5,
        "kd_top1": kd_top1,
        "kd_top5": kd_top5,
        "kd_gain_vs_ce": kd_gain,
        "teacher_ce_loss": round(teacher_metrics["avg_ce_loss"], 6),
        "pruned_ce_loss": round(pruned_metrics["avg_ce_loss"], 6),
        "ce_only_ce_loss": row_ce_ce_loss,
        "kd_ce_loss": kd_ce_loss,
        "val_kd_loss": val_kd_loss,
        "notebook": exp_id,
    }
    rows.append(row)
    del model
    torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# Write results CSV (overwrite)
# ══════════════════════════════════════════════════════════════════════════════

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'=' * 70}")
print(f"ALL {len(rows)} EXPERIMENTS EVALUATED")
print(f"{'=' * 70}")
print(f"Results written to {RESULTS_CSV}")

# ── Summary table ──────────────────────────────────────────────────────────────

print(f"\n{'Exp':<8} {'α':>5} {'T':>5} {'Top-1':>8} {'Top-5':>8} {'CE Loss':>10} {'KD Gain':>9}")
print(f"{'─'*8} {'─'*5} {'─'*5} {'─'*8} {'─'*8} {'─'*10} {'─'*9}")
for row in rows:
    t1 = row["kd_top1"] if row["kd_top1"] != "" else row["ce_only_top1"]
    t5 = row["kd_top5"] if row["kd_top5"] != "" else row["ce_only_top5"]
    ce = row["kd_ce_loss"] if row["kd_ce_loss"] != "" else row["ce_only_ce_loss"]
    gain = row["kd_gain_vs_ce"]
    gain_str = f"{gain:+.2f}%" if isinstance(gain, float) else "N/A"
    print(
        f"{row['notebook']:<8} {row['alpha']:>5} {str(row['temperature']):>5} "
        f"{t1:>7.2f}% {t5:>7.2f}% {ce:>10.4f} {gain_str:>9}"
    )
