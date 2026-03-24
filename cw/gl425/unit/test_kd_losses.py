import torch

from mase_kd.core.losses import DistillationLossConfig, compute_distillation_loss


def test_distillation_loss_combines_hard_and_soft() -> None:
    student_logits = torch.tensor([[2.0, 0.5, -1.0]], dtype=torch.float32)
    teacher_logits = torch.tensor([[1.5, 0.2, -0.3]], dtype=torch.float32)
    targets = torch.tensor([0], dtype=torch.long)

    total, hard_loss, soft_loss = compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        targets=targets,
        config=DistillationLossConfig(alpha=0.4, temperature=2.0),
    )

    expected = 0.6 * hard_loss + 0.4 * soft_loss
    assert torch.isclose(total, expected)
    assert hard_loss.item() > 0
    assert soft_loss.item() >= 0


def test_distillation_loss_supports_soft_only() -> None:
    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)

    total, hard_loss, soft_loss = compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        targets=None,
        config=DistillationLossConfig(alpha=1.0, temperature=3.0),
    )

    assert hard_loss.item() == 0.0
    assert torch.isclose(total, soft_loss)
