"""Integration smoke tests for the YOLO KD pipeline.

Uses a tiny toy vision model (no Ultralytics dependency) to verify the core
distillation loop — loss computation, backward pass, and gradient flow.

For full YOLO smoke tests that require dataset files, see the experiment scripts.
"""

import pytest
import torch
import torch.nn as nn

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.vision.yolo_kd import YOLOLogitsDistiller, YOLOLogitsKDOutput


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------


class TinyVisionModel(nn.Module):
    """Minimal flat MLP that mimics a detection model forward pass."""

    def __init__(self, in_channels: int = 3, img_size: int = 8, num_classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# YOLOLogitsDistiller smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_yolo_logits_kd_single_train_step():
    """One train step should return positive loss values."""
    teacher = TinyVisionModel()
    student = TinyVisionModel()

    distiller = YOLOLogitsDistiller(
        teacher=teacher,
        student=student,
        kd_config=DistillationLossConfig(alpha=0.7, temperature=2.0),
    )

    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    batch = {
        "images": torch.randn(4, 3, 8, 8),
        "targets": torch.randint(0, 6, (4,)),
    }

    output = distiller.train_step(batch=batch, optimizer=optimizer)

    assert isinstance(output, YOLOLogitsKDOutput)
    assert output.total_loss > 0
    assert output.soft_loss >= 0


@pytest.mark.integration
def test_yolo_distiller_teacher_frozen():
    """Teacher weights should not change after a train step."""
    teacher = TinyVisionModel()
    student = TinyVisionModel()

    initial_params = {n: p.clone() for n, p in teacher.named_parameters()}

    distiller = YOLOLogitsDistiller(
        teacher=teacher,
        student=student,
        kd_config=DistillationLossConfig(alpha=0.5, temperature=2.0),
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    batch = {"images": torch.randn(2, 3, 8, 8)}
    distiller.train_step(batch=batch, optimizer=optimizer)

    for name, param in teacher.named_parameters():
        assert torch.allclose(param, initial_params[name]), (
            f"Teacher param '{name}' was modified — should be frozen."
        )


@pytest.mark.integration
def test_yolo_distiller_student_updates():
    """Student weights should change after at least one gradient step."""
    teacher = TinyVisionModel()
    student = TinyVisionModel()

    initial_params = {n: p.clone() for n, p in student.named_parameters()}

    distiller = YOLOLogitsDistiller(
        teacher=teacher,
        student=student,
        kd_config=DistillationLossConfig(alpha=0.5, temperature=2.0),
    )

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    batch = {"images": torch.randn(4, 3, 8, 8)}

    for _ in range(3):
        distiller.train_step(batch=batch, optimizer=optimizer)

    changed = any(
        not torch.allclose(p, initial_params[n])
        for n, p in student.named_parameters()
    )
    assert changed, "Student parameters did not change after training steps."


@pytest.mark.integration
def test_yolo_distiller_soft_only_mode():
    """alpha=1.0, targets=None should use only soft KD loss."""
    teacher = TinyVisionModel()
    student = TinyVisionModel()

    distiller = YOLOLogitsDistiller(
        teacher=teacher,
        student=student,
        kd_config=DistillationLossConfig(alpha=1.0, temperature=2.0),
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    batch = {"images": torch.randn(4, 3, 8, 8)}  # No targets
    output = distiller.train_step(batch=batch, optimizer=optimizer)

    assert output.hard_loss == 0.0
    assert output.soft_loss >= 0


@pytest.mark.integration
def test_yolo_distiller_mismatched_logit_dims():
    """Distiller should handle teacher/student with different output dims."""

    class WideModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 20)  # wider output

    class NarrowModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 10)  # narrower output

    distiller = YOLOLogitsDistiller(
        teacher=WideModel(),
        student=NarrowModel(),
        kd_config=DistillationLossConfig(alpha=0.5, temperature=2.0),
    )

    optimizer = torch.optim.SGD(NarrowModel().parameters(), lr=0.01)
    # Re-create with same student ref for optimizer
    narrow = NarrowModel()
    distiller = YOLOLogitsDistiller(
        teacher=WideModel(),
        student=narrow,
        kd_config=DistillationLossConfig(alpha=1.0, temperature=2.0),
    )
    optimizer = torch.optim.SGD(narrow.parameters(), lr=0.01)
    batch = {"images": torch.randn(4, 3, 8, 8)}
    output = distiller.train_step(batch=batch, optimizer=optimizer)
    assert output.total_loss >= 0


# ---------------------------------------------------------------------------
# Flatten/align helpers
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_flatten_logits_nested_list():
    """_flatten_logits should handle nested list of tensors."""
    distiller = YOLOLogitsDistiller(
        teacher=TinyVisionModel(),
        student=TinyVisionModel(),
        kd_config=DistillationLossConfig(),
    )

    # Nested list: 3 tensors of shape [B, C, H, W]
    nested = [torch.randn(2, 4, 4, 4), torch.randn(2, 4, 2, 2), torch.randn(2, 4, 1, 1)]
    flat = distiller._flatten_logits(nested)
    assert flat.shape[0] == 2
    assert flat.dim() == 2


@pytest.mark.integration
def test_flatten_logits_plain_tensor():
    distiller = YOLOLogitsDistiller(
        teacher=TinyVisionModel(),
        student=TinyVisionModel(),
        kd_config=DistillationLossConfig(),
    )

    t = torch.randn(4, 80)
    flat = distiller._flatten_logits(t)
    assert flat.shape == (4, 80)
