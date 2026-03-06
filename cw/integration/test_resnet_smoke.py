"""Integration smoke tests for ResNet18/CIFAR-10 KD pipeline.

All tests are marked ``@pytest.mark.integration`` and use tiny synthetic data
(random tensors) so they run on CPU in ~10-30 s with no downloads required.

To run:
    pytest cw/integration/test_resnet_smoke.py -v
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.vision.resnet_kd import (
    ResNetKDConfig,
    ResNetKDTrainer,
    build_cifar_resnet18,
    build_resnet_kd_trainer,
)
from mase_kd.passes.prune_pass import PruneConfig, PrunePass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BATCH = 16
_N_CLASSES = 10


def _make_synthetic_loader(n: int = 64, batch_size: int = _BATCH) -> DataLoader:
    """Return a DataLoader with random 3×32×32 images and CIFAR-10 labels."""
    imgs = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, _N_CLASSES, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=False)


def _make_trainer(
    tmp_dir: str,
    alpha: float = 0.5,
    epochs: int = 1,
    teacher_provided: bool = True,
    student_weights: str | None = None,
) -> ResNetKDTrainer:
    train_loader = _make_synthetic_loader(64)
    val_loader = _make_synthetic_loader(32)
    test_loader = _make_synthetic_loader(32)
    device = torch.device("cpu")

    teacher = build_cifar_resnet18(num_classes=_N_CLASSES) if teacher_provided else None

    cfg = ResNetKDConfig(
        teacher_weights="",  # we inject teacher directly
        num_classes=_N_CLASSES,
        student_weights=student_weights,
        kd=DistillationLossConfig(alpha=alpha, temperature=4.0),
        epochs=epochs,
        batch_size=_BATCH,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        lr_schedule="cosine",
        seed=0,
        data_dir=tmp_dir,
        output_dir=tmp_dir,
    )
    student = build_cifar_resnet18(num_classes=_N_CLASSES, weights_path=student_weights)
    return ResNetKDTrainer(
        config=cfg,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResNetKDTrainerOneEpoch:
    def test_history_populated(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5, epochs=1)
        history = trainer.train()
        assert len(history) == 1
        assert "train_loss" in history[0]
        assert "val_accuracy" in history[0]

    def test_losses_positive(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5, epochs=1)
        history = trainer.train()
        assert history[0]["train_loss"] > 0
        assert history[0]["train_hard_loss"] > 0
        assert history[0]["train_soft_loss"] > 0

    def test_val_accuracy_in_range(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5, epochs=1)
        trainer.train()
        val_metrics = trainer.evaluate("val")
        assert 0.0 <= val_metrics["accuracy"] <= 1.0

    def test_best_checkpoint_saved(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5, epochs=1)
        trainer.train()
        assert (tmp_path / "best_student.pth").exists()

    def test_history_json_saved(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5, epochs=1)
        trainer.train()
        assert (tmp_path / "training_history.json").exists()


@pytest.mark.integration
class TestTeacherFrozen:
    def test_teacher_params_unchanged_after_epoch(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5, epochs=1)
        # Save teacher weights before training
        before = {n: p.clone() for n, p in trainer.teacher.named_parameters()}
        trainer.train()
        after = {n: p for n, p in trainer.teacher.named_parameters()}
        for name in before:
            assert torch.allclose(before[name], after[name]), \
                f"Teacher parameter {name} changed during training!"

    def test_teacher_requires_no_grad(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.5)
        for p in trainer.teacher.parameters():
            assert not p.requires_grad


@pytest.mark.integration
class TestBaselineMode:
    def test_alpha_zero_soft_loss_zero(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.0, epochs=1)
        history = trainer.train()
        # When alpha=0 the soft loss is 0
        assert history[0]["train_soft_loss"] == 0.0

    def test_alpha_zero_no_teacher_needed(self, tmp_path):
        """alpha=0 with teacher=None should work fine."""
        trainer = _make_trainer(str(tmp_path), alpha=0.0, teacher_provided=False, epochs=1)
        history = trainer.train()
        assert history[0]["train_loss"] > 0


@pytest.mark.integration
class TestCheckpointSaveLoad:
    def test_save_load_identical_predictions(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.0, epochs=1)
        trainer.train()

        ckpt_path = tmp_path / "best_student.pth"
        # Save initial state
        trainer.save_student(ckpt_path)

        # Load into a fresh model
        loaded = build_cifar_resnet18(num_classes=_N_CLASSES)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded.load_state_dict(state)

        dummy = torch.randn(4, 3, 32, 32)
        loaded.eval()
        trainer.student.eval()
        with torch.no_grad():
            out_orig = trainer.student(dummy)
            out_loaded = loaded(dummy)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6), \
            "Loaded model predictions differ from saved model"

    def test_load_student_method(self, tmp_path):
        trainer = _make_trainer(str(tmp_path), alpha=0.0, epochs=1)
        trainer.train()

        ckpt = tmp_path / "best_student.pth"
        # Corrupt student weights in memory, then reload
        for p in trainer.student.parameters():
            p.data.fill_(999.0)
        trainer.load_student(ckpt)

        dummy = torch.randn(2, 3, 32, 32)
        trainer.student.eval()
        out = trainer.student(dummy)
        # Should not output 999s after reload
        assert out.abs().max().item() < 1e4


@pytest.mark.integration
class TestPruneThenTrain:
    def test_prune_then_finetune_runs(self, tmp_path):
        """Prune a model then pass pruned weights to a new trainer (C-step)."""
        # Train one epoch to get initial weights
        trainer_a = _make_trainer(str(tmp_path / "A"), alpha=0.0, epochs=1)
        trainer_a.train()
        dense_path = tmp_path / "A" / "best_student.pth"

        # Prune
        pruned = build_cifar_resnet18(num_classes=_N_CLASSES, weights_path=str(dense_path))
        _, info = PrunePass().run(
            pruned, PruneConfig(sparsity=0.5, make_permanent=True), {}
        )
        pruned_path = tmp_path / "pruned.pth"
        torch.save(pruned.state_dict(), pruned_path)
        assert info["sparsity_actual"] > 0.3

        # Fine-tune from pruned weights
        trainer_c = _make_trainer(
            str(tmp_path / "C"),
            alpha=0.0,
            epochs=1,
            teacher_provided=False,
            student_weights=str(pruned_path),
        )
        history = trainer_c.train()
        assert history[0]["train_loss"] > 0
