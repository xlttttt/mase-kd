#!/usr/bin/env python3
"""P3 full grid search — runs all 26 experiments (CE baseline + 25 KD configs) in one go.

Equivalent to running exp_00_P3_baseline_CE.ipynb followed by exp_01..exp_25 notebooks
sequentially, but shares the teacher, dataloaders, and pruning step across runs.

Usage:
    cd cw
    python3 YOLO_logitKD_experiments/P3_grid_search.py
"""

import copy
import csv
import gc
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
STUDENT_CHECKPOINT = "data/cifar10_yolov8m_cls/runs/yolov8m_cls_cifar10_finetune/weights/best.pt"
TEACHER_CFG = "yolov8x-cls.yaml"
STUDENT_CFG = "yolov8m-cls.yaml"

DEVICE = "cuda"
DATA_ROOT = "./data"

IMAGE_SIZE = 32
BATCH_SIZE = 128

PRUNE_SPARSITY = 0.50
EPOCHS = 8
LR = 1e-6
SEED = 42

RESULTS_CSV = "YOLO_logitKD_experiments/results.csv"
SAVE_DIR = Path("data/P3_grid_search")

ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]
TEMPERATURES = [1.0, 2.0, 4.0, 8.0, 16.0]

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

from mase_kd.core.losses import DistillationLossConfig, hard_label_ce_loss
from mase_kd.vision.yolo_kd import YOLOLogitsDistiller

if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"

torch.manual_seed(SEED)
random.seed(SEED)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Repo root: {repo_root}")
print(f"Device:    {DEVICE}")
print(f"Grid:      {len(ALPHAS)} alphas × {len(TEMPERATURES)} temps + 1 CE baseline = {len(ALPHAS)*len(TEMPERATURES)+1} runs")

# ── Evaluation helper ──────────────────────────────────────────────────────────


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


def append_csv(row: dict):
    csv_path = Path(RESULTS_CSV)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── CIFAR10 dataloaders ───────────────────────────────────────────────────────

cifar_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=cifar_transform, download=True)
val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=cifar_transform, download=True)

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
print(f"Teacher — top1: {teacher_metrics['top1_acc']*100:.2f}%  CE: {teacher_metrics['avg_ce_loss']:.4f}")

# ── Build student seed and prune ───────────────────────────────────────────────

ultra_student = YOLO(STUDENT_CHECKPOINT)
student_seed = MaseYoloClassificationModel(cfg=STUDENT_CFG, nc=nc)
student_seed = patch_yolo(student_seed)
student_seed.load_state_dict(ultra_student.model.state_dict())
del ultra_student

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

# This is the pruned student — we deep-copy it for each experiment
pruned_model = mg.model.to(DEVICE)
pruned_snapshot_state = copy.deepcopy(pruned_model.state_dict())
del mg, student_seed, trace_input, dummy_in

print(f"Student loaded: {STUDENT_CHECKPOINT}")
print(f"Pruning complete ({PRUNE_SPARSITY*100:.0f}% sparsity)")

# Evaluate pruned model (once — same for all experiments)
pruned_model.eval()
pruned_metrics = evaluate_model(pruned_model, val_loader, DEVICE)
print(f"Pruned (no train) — top1: {pruned_metrics['top1_acc']*100:.2f}%  CE: {pruned_metrics['avg_ce_loss']:.4f}")


# ── Helper: reset student to pruned state ──────────────────────────────────────

def reset_student():
    """Return a fresh copy of the pruned student model."""
    model = copy.deepcopy(pruned_model).to(DEVICE)
    model.load_state_dict(pruned_snapshot_state, strict=False)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 00 — CE-only baseline
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("EXP 00 — CE-only baseline (α=0.0, no teacher)")
print("=" * 80)

ce_model = reset_student()
ce_optimizer = torch.optim.Adam(ce_model.parameters(), lr=LR)
ce_save_path = str(SAVE_DIR / "best_pruned_finetuned_exp_00.pt")

best_val_top1 = 0.0

for epoch in range(1, EPOCHS + 1):
    ce_model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        ce_optimizer.zero_grad(set_to_none=True)
        outputs = ce_model(images)
        logits = YOLOLogitsDistiller._extract_logits_with_batch(outputs, images.shape[0])
        loss = hard_label_ce_loss(logits, labels)
        loss.backward()
        ce_optimizer.step()
        epoch_loss += loss.item()
        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == num_batches:
            with torch.no_grad():
                top1 = float((logits.argmax(dim=1) == labels).float().mean())
            print(f"  Epoch {epoch}/{EPOCHS} | Batch {batch_idx:04d}/{num_batches} | "
                  f"loss={loss.item():.6f} | top1={top1*100:.2f}%")
    print(f"Epoch {epoch} avg loss: {epoch_loss / num_batches:.6f}")

    epoch_val = evaluate_model(ce_model, val_loader, DEVICE)
    if epoch_val["top1_acc"] > best_val_top1:
        best_val_top1 = epoch_val["top1_acc"]
        torch.save(ce_model.state_dict(), ce_save_path)
        print(f"  [checkpoint] val top1={best_val_top1*100:.2f}% — saved to {ce_save_path}")

ce_model.load_state_dict(torch.load(ce_save_path), strict=False)
ce_metrics = evaluate_model(ce_model, val_loader, DEVICE)
print(f"\nCE-only fine-tuned (best) — top1: {ce_metrics['top1_acc']*100:.2f}%  CE: {ce_metrics['avg_ce_loss']:.4f}")

append_csv({
    "pair": PAIR,
    "alpha": 0.0,
    "temperature": "",
    "lr": LR,
    "epochs": EPOCHS,
    "sparsity": PRUNE_SPARSITY,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "teacher_top1": round(teacher_metrics["top1_acc"] * 100, 4),
    "pruned_top1": round(pruned_metrics["top1_acc"] * 100, 4),
    "ce_only_top1": round(ce_metrics["top1_acc"] * 100, 4),
    "kd_top1": "",
    "kd_gain_vs_ce": "",
    "teacher_ce_loss": round(teacher_metrics["avg_ce_loss"], 6),
    "pruned_ce_loss": round(pruned_metrics["avg_ce_loss"], 6),
    "ce_only_ce_loss": round(ce_metrics["avg_ce_loss"], 6),
    "kd_ce_loss": "",
    "val_kd_loss": "",
    "notebook": "exp_00",
})
print("Results appended to", RESULTS_CSV)

del ce_model, ce_optimizer
gc.collect()
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Experiments 01–25 — KD grid (α × T)
# ══════════════════════════════════════════════════════════════════════════════

exp_num = 0
for alpha in ALPHAS:
    for temp in TEMPERATURES:
        exp_num += 1
        exp_id = f"exp_{exp_num:02d}"
        kd_save_path = str(SAVE_DIR / f"best_student_{exp_id}.pt")

        print(f"\n{'=' * 80}")
        print(f"{exp_id.upper()} — KD α={alpha}, T={temp}")
        print(f"{'=' * 80}")

        # Fresh pruned student for this experiment
        student = reset_student()
        optimizer = torch.optim.Adam(student.parameters(), lr=LR)
        kd_config = DistillationLossConfig(alpha=alpha, temperature=temp)

        distiller = YOLOLogitsDistiller(
            teacher=teacher_cls_model,
            student=student,
            kd_config=kd_config,
            device=DEVICE,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_train_epochs=EPOCHS,
            eval_teacher=True,
        )

        train_history = distiller.train(save_path=kd_save_path)

        # Restore best student weights
        student.load_state_dict(torch.load(kd_save_path), strict=False)
        print(f"Best student weights restored from {kd_save_path}")

        # Evaluate
        eval_results = distiller.evaluate()
        t_metrics = eval_results.get("teacher")
        s_metrics = eval_results["student"]
        val_kd_loss = eval_results["val_kd_loss"]

        print(f"\n{'Model':<40} {'Top-1':>8} {'CE Loss':>10}")
        print(f"{'─' * 40} {'─' * 8} {'─' * 10}")
        if t_metrics:
            print(f"{'Teacher (yolov8x-cls)':<40} {t_metrics['top1_acc']*100:>7.2f}% {t_metrics['avg_ce_loss']:>10.4f}")
        print(f"{'Pruned (no train, 50%)':<40} {pruned_metrics['top1_acc']*100:>7.2f}% {pruned_metrics['avg_ce_loss']:>10.4f}")
        print(f"{'CE-only (baseline)':<40} {ce_metrics['top1_acc']*100:>7.2f}% {ce_metrics['avg_ce_loss']:>10.4f}")
        print(f"{f'Distilled (α={alpha}, T={temp})':<40} {s_metrics['top1_acc']*100:>7.2f}% {s_metrics['avg_ce_loss']:>10.4f}")

        kd_gain = round(s_metrics["top1_acc"] * 100 - ce_metrics["top1_acc"] * 100, 4)

        append_csv({
            "pair": PAIR,
            "alpha": alpha,
            "temperature": temp,
            "lr": LR,
            "epochs": EPOCHS,
            "sparsity": PRUNE_SPARSITY,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "teacher_top1": round(t_metrics["top1_acc"] * 100, 4) if t_metrics else "",
            "pruned_top1": round(pruned_metrics["top1_acc"] * 100, 4),
            "ce_only_top1": round(ce_metrics["top1_acc"] * 100, 4),
            "kd_top1": round(s_metrics["top1_acc"] * 100, 4),
            "kd_gain_vs_ce": kd_gain,
            "teacher_ce_loss": round(t_metrics["avg_ce_loss"], 6) if t_metrics else "",
            "pruned_ce_loss": round(pruned_metrics["avg_ce_loss"], 6),
            "ce_only_ce_loss": round(ce_metrics["avg_ce_loss"], 6),
            "kd_ce_loss": round(s_metrics["avg_ce_loss"], 6),
            "val_kd_loss": round(val_kd_loss, 6),
            "notebook": exp_id,
        })
        print(f"Results appended to {RESULTS_CSV}  (KD gain vs CE: {kd_gain:+.2f}%)")

        del student, optimizer, distiller, kd_config
        gc.collect()
        torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("ALL 26 P3 EXPERIMENTS COMPLETE")
print("=" * 80)
print(f"Results CSV: {RESULTS_CSV}")
print(f"Checkpoints: {SAVE_DIR}/")
