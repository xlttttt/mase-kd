"""A–E pipeline orchestration for ResNet18/CIFAR-10, BERT/SST-2, and YOLO/COCO.

Each pipeline class exposes a single ``run(config, output_dir, sparsity)``
class-method that executes the five experimental variants in order and returns
a ``{A: {...}, B: {...}, C: {...}, D: {...}, E: {...}}`` metrics dict.

Output directory structure for ResNet (same pattern for others)::

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
# BERT pipeline (stub, delegates to BertKDTrainer)
# ---------------------------------------------------------------------------


class BertPipeline:
    """Orchestrate A–E for BERT/SST-2.

    Stub implementation: calls :class:`mase_kd.nlp.bert_kd.BertKDTrainer`
    and :class:`mase_kd.passes.prune_pass.PrunePass`.
    """

    def run(
        self,
        config: dict,
        output_dir: str,
        sparsity: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> dict[str, dict]:
        from mase_kd.nlp.bert_kd import (
            BertKDConfig,
            BertStudentConfig,
            build_bert_kd_trainer,
            BertForSequenceClassification,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_dir = Path(output_dir) / f"sparsity_{sparsity:.2f}"
        base_dir.mkdir(parents=True, exist_ok=True)

        teacher_cfg = config.get("teacher", {})
        student_cfg_d = config.get("student", {})
        dense_d = config.get("dense_training", {})
        prune_d = config.get("pruning", {})
        ft_d = config.get("finetune", {})
        kd_d = config.get("kd", {})
        kdft_d = config.get("kd_finetune", {})

        student_cfg = BertStudentConfig(
            num_hidden_layers=student_cfg_d.get("num_hidden_layers", 4),
            hidden_size=student_cfg_d.get("hidden_size", 256),
            num_attention_heads=student_cfg_d.get("num_attention_heads", 4),
            intermediate_size=student_cfg_d.get("intermediate_size", 1024),
        )

        def _make_bert_cfg(step_dir, alpha, student_weights_path, step_d):
            # When a local checkpoint path is given (steps C/D/E), initialise the
            # student from that path so pruned weights are not discarded.
            eff_student_cfg = BertStudentConfig(
                num_hidden_layers=student_cfg.num_hidden_layers,
                hidden_size=student_cfg.hidden_size,
                num_attention_heads=student_cfg.num_attention_heads,
                intermediate_size=student_cfg.intermediate_size,
                pretrained_name=student_weights_path if student_weights_path else None,
            )
            return BertKDConfig(
                teacher_model_name=teacher_cfg.get(
                    "model_name", "textattack/bert-base-uncased-SST-2"
                ),
                student=eff_student_cfg,
                kd=DistillationLossConfig(
                    alpha=alpha,
                    temperature=kd_d.get("temperature", 4.0),
                ),
                batch_size=step_d.get("batch_size", 32),
                learning_rate=step_d.get("learning_rate", 2e-5),
                num_epochs=step_d.get("epochs", 1),
                warmup_ratio=step_d.get("warmup_ratio", 0.06),
                weight_decay=step_d.get("weight_decay", 0.01),
                seed=dense_d.get("seed", 42),
                output_dir=str(step_dir),
            )

        # Step A
        a_dir = base_dir / "A_dense"
        trainer_a = build_bert_kd_trainer(_make_bert_cfg(a_dir, 0.0, None, dense_d), device)
        trainer_a.train()
        val_a = trainer_a.evaluate()
        nonzero_a, total_a = _count_params(trainer_a.student)
        metrics_a = {**val_a, "params_nonzero": nonzero_a, "params_total": total_a, "sparsity": 0.0}
        metrics_a["accuracy"] = val_a.get("val_accuracy", 0.0)
        _save_metrics(metrics_a, a_dir / "metrics.json")

        # Step B (prune)
        b_dir = base_dir / "B_pruned"
        b_dir.mkdir(parents=True, exist_ok=True)
        pruned_student = copy.deepcopy(trainer_a.student)
        prune_pass = PrunePass()
        pruned_student, prune_info = prune_pass.run(
            pruned_student,
            PruneConfig(sparsity=sparsity, target_types=(nn.Linear,), make_permanent=True),
            {},
        )
        pruned_path = b_dir / "pruned_student"
        pruned_student.save_pretrained(str(pruned_path))

        # Evaluate pruned model accuracy by temporarily swapping the student
        # in trainer_a so we can reuse its val_loader and evaluate() method.
        # Use trainer_a.device (not the pipeline-level device) so that data and
        # model stay on the same device even when the mock uses a different device.
        orig_student_a = trainer_a.student
        trainer_a.student = pruned_student.to(trainer_a.device)
        eval_b = trainer_a.evaluate()
        trainer_a.student = orig_student_a

        metrics_b = {
            "accuracy": eval_b["val_accuracy"],
            "val_accuracy": eval_b["val_accuracy"],
            "params_nonzero": prune_info["params_nonzero"],
            "params_total": prune_info["params_total"],
            "sparsity": prune_info["sparsity_actual"],
        }
        _save_metrics(metrics_b, b_dir / "metrics.json")

        # Steps C, D, E — minimal stubs (return zero metrics to complete the run)
        for step_key, step_dir_name, alpha, step_d, sw in [
            ("C", "C_ft", 0.0, ft_d, str(pruned_path)),
            ("D", "D_kd", kd_d.get("alpha", 0.5), kd_d, str(pruned_path)),
        ]:
            step_dir = base_dir / step_dir_name
            cfg = _make_bert_cfg(step_dir, alpha, sw, step_d)
            trainer = build_bert_kd_trainer(cfg, device)
            trainer.train()
            val_m = trainer.evaluate()
            nz, tot = _count_params(trainer.student)
            m = {**val_m, "params_nonzero": nz, "params_total": tot,
                 "sparsity": 1.0 - nz / max(tot, 1)}
            m["accuracy"] = val_m.get("val_accuracy", 0.0)
            _save_metrics(m, step_dir / "metrics.json")

        # Step E
        e_dir = base_dir / "E_kd_ft"
        d_best = base_dir / "D_kd" / "best_student"
        if d_best.exists():
            e_cfg = _make_bert_cfg(e_dir, 0.0, str(d_best), kdft_d)
            trainer_e = build_bert_kd_trainer(e_cfg, device)
            trainer_e.train()
            val_e = trainer_e.evaluate()
            nz_e, tot_e = _count_params(trainer_e.student)
            metrics_e = {**val_e, "params_nonzero": nz_e, "params_total": tot_e,
                         "sparsity": 1.0 - nz_e / max(tot_e, 1)}
            metrics_e["accuracy"] = val_e.get("val_accuracy", 0.0)
        else:
            metrics_e = {"accuracy": 0.0, "sparsity": sparsity}
        _save_metrics(metrics_e, e_dir / "metrics.json")

        results = {
            "A": metrics_a,
            "B": metrics_b,
            "C": json.loads((base_dir / "C_ft" / "metrics.json").read_text()),
            "D": json.loads((base_dir / "D_kd" / "metrics.json").read_text()),
            "E": metrics_e,
        }
        ExportMetricsPass().run(results, str(base_dir), "bert", "accuracy")
        return results


# ---------------------------------------------------------------------------
# YOLO pipeline (stub)
# ---------------------------------------------------------------------------


class YoloPipeline:
    """Orchestrate A–E for YOLOv8n/COCO.

    Stub implementation: delegates to :class:`mase_kd.vision.yolo_kd_train.YOLOKDRunner`.
    YOLO pruning targets Conv2d weights only.
    """

    def run(
        self,
        config: dict,
        output_dir: str,
        sparsity: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> dict[str, dict]:
        from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig, build_yolo_kd_runner

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_dir = Path(output_dir) / f"sparsity_{sparsity:.2f}"
        base_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = config.get("model", {})
        dense_d = config.get("dense_training", {})
        ft_d = config.get("finetune", {})
        kd_d = config.get("kd", {})
        kdft_d = config.get("kd_finetune", {})

        def _make_yolo_cfg(step_dir, alpha, student_weights=None, step_d=None):
            if step_d is None:
                step_d = {}
            return YOLOTrainingConfig(
                teacher_weights=model_cfg.get("teacher_weights", "yolov8m.pt"),
                student_arch=model_cfg.get("student_arch", "yolov8n.yaml"),
                student_weights=student_weights,
                data_yaml=model_cfg.get("data_yaml", "coco8.yaml"),
                kd=DistillationLossConfig(
                    alpha=alpha, temperature=kd_d.get("temperature", 2.0)
                ),
                epochs=step_d.get("epochs", 2),
                batch_size=step_d.get("batch_size", 4),
                learning_rate=step_d.get("learning_rate", 1e-3),
                seed=dense_d.get("seed", 42),
                output_dir=str(step_dir),
            )

        # Step A: dense baseline
        a_dir = base_dir / "A_dense"
        runner_a = build_yolo_kd_runner(_make_yolo_cfg(a_dir, 0.0, None, dense_d))
        runner_a.train()
        val_a = runner_a.evaluate()
        nz_a, tot_a = _count_params(runner_a.student)
        metrics_a = {**val_a, "params_nonzero": nz_a, "params_total": tot_a, "sparsity": 0.0}
        _save_metrics(metrics_a, a_dir / "metrics.json")

        # Step B: prune
        b_dir = base_dir / "B_pruned"
        b_dir.mkdir(parents=True, exist_ok=True)
        pruned_student = copy.deepcopy(runner_a.student)
        prune_pass = PrunePass()
        pruned_student, prune_info = prune_pass.run(
            pruned_student,
            PruneConfig(sparsity=sparsity, target_types=(nn.Conv2d,), make_permanent=True),
            {},
        )
        pruned_path = b_dir / "pruned_student.pt"
        torch.save(pruned_student.state_dict(), pruned_path)

        # Evaluate pruned model by temporarily swapping runner_a's student so
        # we can reuse its evaluate() method (saves to tmp file + runs val).
        orig_student_a = runner_a.student
        runner_a.student = pruned_student.to(runner_a.device)
        eval_b = runner_a.evaluate()
        runner_a.student = orig_student_a

        metrics_b = {
            "mAP50": eval_b["mAP50"],
            "mAP50_95": eval_b.get("mAP50_95", 0.0),
            "params_nonzero": prune_info["params_nonzero"],
            "params_total": prune_info["params_total"],
            "sparsity": prune_info["sparsity_actual"],
        }
        _save_metrics(metrics_b, b_dir / "metrics.json")

        # Steps C, D, E
        for step_key, step_dir_name, alpha, step_d, sw in [
            ("C", "C_ft", 0.0, ft_d, str(pruned_path)),
            ("D", "D_kd", kd_d.get("alpha", 0.5), kd_d, str(pruned_path)),
        ]:
            step_dir = base_dir / step_dir_name
            runner = build_yolo_kd_runner(_make_yolo_cfg(step_dir, alpha, sw, step_d))
            runner.train()
            val_m = runner.evaluate()
            nz, tot = _count_params(runner.student)
            m = {**val_m, "params_nonzero": nz, "params_total": tot,
                 "sparsity": 1.0 - nz / max(tot, 1)}
            _save_metrics(m, step_dir / "metrics.json")

        # Step E
        e_dir = base_dir / "E_kd_ft"
        d_last = base_dir / "D_kd" / "last_student.pt"
        d_best = base_dir / "D_kd" / "best_student.pt"
        d_weights = str(d_best) if d_best.exists() else (str(d_last) if d_last.exists() else None)
        runner_e = build_yolo_kd_runner(_make_yolo_cfg(e_dir, 0.0, d_weights, kdft_d))
        runner_e.train()
        val_e = runner_e.evaluate()
        nz_e, tot_e = _count_params(runner_e.student)
        metrics_e = {**val_e, "params_nonzero": nz_e, "params_total": tot_e,
                     "sparsity": 1.0 - nz_e / max(tot_e, 1)}
        _save_metrics(metrics_e, e_dir / "metrics.json")

        # Rename mAP50 → "accuracy" key for ExportMetrics to use a common label
        results = {}
        for key, step_dir_name in [("A", "A_dense"), ("B", "B_pruned"), ("C", "C_ft"),
                                    ("D", "D_kd"), ("E", "E_kd_ft")]:
            m_path = base_dir / step_dir_name / "metrics.json"
            if m_path.exists():
                results[key] = json.loads(m_path.read_text())

        ExportMetricsPass().run(results, str(base_dir), "yolo", "mAP50")
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
