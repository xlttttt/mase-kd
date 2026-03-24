"""Reusable hard/soft distillation losses."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DistillationLossConfig:
    """Hyperparameters for logits distillation objective."""

    alpha: float = 0.5
    temperature: float = 2.0

    def validate(self) -> None:
        """Ensure `alpha` and `temperature` are valid."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


def soft_logit_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute temperature-scaled KL divergence between student and teacher logits."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature

    loss = F.kl_div(
        F.log_softmax(student_scaled, dim=-1),
        F.softmax(teacher_scaled, dim=-1),
        reduction="batchmean",
    )
    return loss * (temperature**2)


def hard_label_ce_loss(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute hard-label cross entropy for the student logits."""
    return F.cross_entropy(student_logits, targets)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor | None,
    config: DistillationLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total distillation loss and its hard/soft components."""
    config.validate()

    soft_loss = soft_logit_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=config.temperature,
    )

    if targets is None:
        hard_loss = torch.zeros_like(soft_loss)
    else:
        hard_loss = hard_label_ce_loss(student_logits, targets)

    total = (1.0 - config.alpha) * hard_loss + config.alpha * soft_loss
    return total, hard_loss, soft_loss
