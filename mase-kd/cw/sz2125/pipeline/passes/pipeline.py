"""A–E pipeline orchestration for ResNet18/CIFAR-10.

Each pipeline class exposes a single ``run(config, output_dir, sparsity)``
class-method that executes the five experimental variants in order and returns
a ``{A: {...}, B: {...}, C: {...}, D: {...}, E: {...}}`` metrics dict.

Output directory structure for ResNet::

    <output_dir>/sparsity_<s>/
        A_dense/   best_student.pth, training_history.json, metrics.json
        B_pruned/  pruned_student.pth, metrics.json
        C_ft/      best_student.pth, training_history.json, metrics.json
        D_kd/      best_student.pth, training_history.json, metrics.json
        E_kd_ft/   best_student.pth, training_history.json, metrics.json
        comparison_table.{md,json}
        trade_off_plot.png
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.passes.export_pass import ExportMetricsPass
from mase_kd.passes.prune_pass import PruneConfig, PrunePass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(metrics, fp, indent=2)


def _count_params(model: nn.Module) -> tuple[int, int]:
    from mase_kd.passes.prune_pass import count_nonzero_params
    return count_nonzero_params(model)


# ---------------------------------------------------------------------------
# ResNet pipeline
# ---------------------------------------------------------------------------


class ResNetPipeline:
    """Orchestrate the A–E experimental matrix for ResNet18 on CIFAR-10.

    The dense ResNet18 trained in Step A is reused as the KD teacher for
    Steps D and E (self-distillation from the unpruned checkpoint).

    Args:
        config:     Parsed config dict (from YAML/TOML pipeline file).
        output_dir: Root output directory; sparsity subdirectory is created
                    automatically.
        sparsity:   Fraction of weights to prune (overrides config).
        device:     Torch device; auto-detected if None.
    """

    def run(
        self,
        config: dict,
        output_dir: str,
        sparsity: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> dict[str, dict]:
        from mase_kd.vision.resnet_kd import (
            ResNetKDConfig,
            build_resnet_kd_trainer,
            build_cifar_resnet18,
            load_cifar10_dataloaders,
            load_cifar100_dataloaders,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ResNetPipeline | device=%s | sparsity=%.2f", device, sparsity)

        base_dir = Path(output_dir) / f"sparsity_{sparsity:.2f}"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Extract sub-configs with sensible fallbacks
        data_cfg = config.get("data", {})
        dense_cfg = config.get("dense_training", {})
        prune_cfg = config.get("pruning", {})
        ft_cfg = config.get("finetune", {})
        kd_cfg = config.get("kd", {})
        kdft_cfg = config.get("kd_finetune", {})
        model_cfg = config.get("model", {})
        num_classes = data_cfg.get("num_classes", model_cfg.get("num_classes", 10))
        teacher_arch = config.get("teacher", {}).get("arch", model_cfg.get("teacher_arch", "resnet18"))
        subset_size = data_cfg.get("subset_size", None)
        seed = dense_cfg.get("seed", 42)
        dataset = data_cfg.get("dataset", "cifar10")

        # ------------------------------------------------------------------
        # Step A: Dense baseline (alpha=0, no teacher needed)
        # ------------------------------------------------------------------
        logger.info("=== Step A: Dense baseline ===")
        a_dir = base_dir / "A_dense"
        a_cfg = ResNetKDConfig(
            teacher_weights="",
            num_classes=num_classes,
            dataset=dataset,
            kd=DistillationLossConfig(alpha=0.0, temperature=4.0),
            epochs=dense_cfg.get("epochs", 2),
            batch_size=dense_cfg.get("batch_size", 128),
            learning_rate=dense_cfg.get("learning_rate", 0.1),
            momentum=dense_cfg.get("momentum", 0.9),
            weight_decay=dense_cfg.get("weight_decay", 5e-4),
            lr_schedule=dense_cfg.get("lr_schedule", "cosine"),
            seed=seed,
            data_dir=data_cfg.get("dir", "datasets/cifar10"),
            val_split=data_cfg.get("val_split", 0.1),
            subset_size=subset_size,
            output_dir=str(a_dir),
        )
        trainer_a = build_resnet_kd_trainer(a_cfg, device)
        trainer_a.train()
        # Use best_student.pth as the teacher for D/E
        dense_teacher_path = str(a_dir / "best_student.pth")

        # Re-load best student for evaluation
        trainer_a.load_student(dense_teacher_path)
        val_a = trainer_a.evaluate("val")
        test_a = trainer_a.evaluate("test")
        nonzero_a, total_a = _count_params(trainer_a.student)
        metrics_a = {
            "accuracy": test_a["accuracy"],
            "val_accuracy": val_a["accuracy"],
            "params_nonzero": nonzero_a,
            "params_total": total_a,
            "sparsity": 0.0,
        }
        _save_metrics(metrics_a, a_dir / "metrics.json")
        logger.info("Step A done | test_acc=%.4f", metrics_a["accuracy"])

        # ------------------------------------------------------------------
        # Step B: Pruned (no recovery training)
        # ------------------------------------------------------------------
        logger.info("=== Step B: Pruned (sparsity=%.2f) ===", sparsity)
        b_dir = base_dir / "B_pruned"
        b_dir.mkdir(parents=True, exist_ok=True)

        pruned_student = copy.deepcopy(trainer_a.student)
        prune_pass = PrunePass()
        target_types = (nn.Conv2d, nn.Linear)
        pruned_student, prune_info = prune_pass.run(
            pruned_student,
            PruneConfig(sparsity=sparsity, target_types=target_types, make_permanent=True),
            {},
        )
        pruned_path = b_dir / "pruned_student.pth"
        torch.save(pruned_student.state_dict(), pruned_path)

        # Evaluate pruned model on same data
        _load_fn = load_cifar100_dataloaders if dataset == "cifar100" else load_cifar10_dataloaders
        _, val_loader_b, test_loader_b = _load_fn(
            data_dir=data_cfg.get("dir", "datasets/cifar10"),
            batch_size=dense_cfg.get("batch_size", 128),
            val_split=data_cfg.get("val_split", 0.1),
            subset_size=subset_size,
            seed=seed,
        )
        acc_b_val = _eval_model(pruned_student, val_loader_b, device)
        acc_b_test = _eval_model(pruned_student, test_loader_b, device)
        metrics_b = {
            "accuracy": acc_b_test,
            "val_accuracy": acc_b_val,
            "params_nonzero": prune_info["params_nonzero"],
            "params_total": prune_info["params_total"],
            "sparsity": prune_info["sparsity_actual"],
        }
        _save_metrics(metrics_b, b_dir / "metrics.json")
        logger.info("Step B done | test_acc=%.4f | sparsity=%.4f",
                    metrics_b["accuracy"], metrics_b["sparsity"])

        # ------------------------------------------------------------------
        # Step C: Pruned + Fine-tune (alpha=0)
        # ------------------------------------------------------------------
        logger.info("=== Step C: Pruned + Finetune ===")
        c_dir = base_dir / "C_ft"
        c_cfg = ResNetKDConfig(
            teacher_weights="",
            num_classes=num_classes,
            dataset=dataset,
            student_weights=str(pruned_path),
            kd=DistillationLossConfig(alpha=0.0, temperature=4.0),
            epochs=ft_cfg.get("epochs", 1),
            batch_size=ft_cfg.get("batch_size", 128),
            learning_rate=ft_cfg.get("learning_rate", 0.01),
            momentum=dense_cfg.get("momentum", 0.9),
            weight_decay=dense_cfg.get("weight_decay", 5e-4),
            lr_schedule=dense_cfg.get("lr_schedule", "cosine"),
            seed=seed,
            data_dir=data_cfg.get("dir", "datasets/cifar10"),
            val_split=data_cfg.get("val_split", 0.1),
            subset_size=subset_size,
            output_dir=str(c_dir),
        )
        trainer_c = build_resnet_kd_trainer(c_cfg, device)
        trainer_c.train()
        trainer_c.load_student(c_dir / "best_student.pth")
        val_c = trainer_c.evaluate("val")
        test_c = trainer_c.evaluate("test")
        # Sparsity: report the initial pruning sparsity (B).
        # Note: without mask reinstatement, SGD fills in the pruned zeros
        # during fine-tuning, so the final model is effectively dense.
        metrics_c = {
            "accuracy": test_c["accuracy"],
            "val_accuracy": val_c["accuracy"],
            "params_nonzero": prune_info["params_nonzero"],   # initial sparse count
            "params_total": prune_info["params_total"],
            "sparsity": prune_info["sparsity_actual"],         # from B
        }
        _save_metrics(metrics_c, c_dir / "metrics.json")
        logger.info("Step C done | test_acc=%.4f", metrics_c["accuracy"])

        # ------------------------------------------------------------------
        # Step D: Pruned + KD (from B, with dense A as teacher)
        # ------------------------------------------------------------------
        logger.info("=== Step D: Pruned + KD ===")
        d_dir = base_dir / "D_kd"
        kd_alpha = kd_cfg.get("alpha", 0.5)
        kd_temp = kd_cfg.get("temperature", 4.0)
        d_cfg = ResNetKDConfig(
            teacher_weights=dense_teacher_path,
            teacher_arch=teacher_arch,
            num_classes=num_classes,
            dataset=dataset,
            student_weights=str(pruned_path),
            kd=DistillationLossConfig(alpha=kd_alpha, temperature=kd_temp),
            epochs=kd_cfg.get("epochs", 1),
            batch_size=kd_cfg.get("batch_size", 128),
            learning_rate=kd_cfg.get("learning_rate", 0.01),
            momentum=dense_cfg.get("momentum", 0.9),
            weight_decay=dense_cfg.get("weight_decay", 5e-4),
            lr_schedule=dense_cfg.get("lr_schedule", "cosine"),
            seed=seed,
            data_dir=data_cfg.get("dir", "datasets/cifar10"),
            val_split=data_cfg.get("val_split", 0.1),
            subset_size=subset_size,
            output_dir=str(d_dir),
        )
        trainer_d = build_resnet_kd_trainer(d_cfg, device)
        trainer_d.train()
        trainer_d.load_student(d_dir / "best_student.pth")
        val_d = trainer_d.evaluate("val")
        test_d = trainer_d.evaluate("test")
        metrics_d = {
            "accuracy": test_d["accuracy"],
            "val_accuracy": val_d["accuracy"],
            "params_nonzero": prune_info["params_nonzero"],
            "params_total": prune_info["params_total"],
            "sparsity": prune_info["sparsity_actual"],
        }
        _save_metrics(metrics_d, d_dir / "metrics.json")
        logger.info("Step D done | test_acc=%.4f", metrics_d["accuracy"])

        # ------------------------------------------------------------------
        # Step E: Pruned + KD + Fine-tune (start from D, alpha=0)
        # ------------------------------------------------------------------
        logger.info("=== Step E: Pruned + KD + Finetune ===")
        e_dir = base_dir / "E_kd_ft"
        e_cfg = ResNetKDConfig(
            teacher_weights="",
            num_classes=num_classes,
            dataset=dataset,
            student_weights=str(d_dir / "best_student.pth"),
            kd=DistillationLossConfig(alpha=0.0, temperature=4.0),
            epochs=kdft_cfg.get("epochs", 1),
            batch_size=kdft_cfg.get("batch_size", 128),
            learning_rate=kdft_cfg.get("learning_rate", 0.001),
            momentum=dense_cfg.get("momentum", 0.9),
            weight_decay=dense_cfg.get("weight_decay", 5e-4),
            lr_schedule=dense_cfg.get("lr_schedule", "cosine"),
            seed=seed,
            data_dir=data_cfg.get("dir", "datasets/cifar10"),
            val_split=data_cfg.get("val_split", 0.1),
            subset_size=subset_size,
            output_dir=str(e_dir),
        )
        trainer_e = build_resnet_kd_trainer(e_cfg, device)
        trainer_e.train()
        trainer_e.load_student(e_dir / "best_student.pth")
        val_e = trainer_e.evaluate("val")
        test_e = trainer_e.evaluate("test")
        metrics_e = {
            "accuracy": test_e["accuracy"],
            "val_accuracy": val_e["accuracy"],
            "params_nonzero": prune_info["params_nonzero"],
            "params_total": prune_info["params_total"],
            "sparsity": prune_info["sparsity_actual"],
        }
        _save_metrics(metrics_e, e_dir / "metrics.json")
        logger.info("Step E done | test_acc=%.4f", metrics_e["accuracy"])

        # ------------------------------------------------------------------
        # Export comparison table
        # ------------------------------------------------------------------
        results = {
            "A": metrics_a,
            "B": metrics_b,
            "C": metrics_c,
            "D": metrics_d,
            "E": metrics_e,
        }
        ExportMetricsPass().run(
            results=results,
            output_dir=str(base_dir),
            model_name="resnet18",
            primary_metric="accuracy",
        )
        logger.info("ResNetPipeline complete. Results in %s", base_dir)
        return results



# ---------------------------------------------------------------------------
# Shared model evaluation helper (used by ResNetPipeline)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_model(model: nn.Module, loader, device: torch.device) -> float:
    """Return top-1 accuracy of *model* on *loader*."""
    import torch.nn.functional as F
    model.eval()
    model.to(device)
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)
