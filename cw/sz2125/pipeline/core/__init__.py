"""Core KD components."""

from .losses import DistillationLossConfig, compute_distillation_loss

__all__ = ["DistillationLossConfig", "compute_distillation_loss"]
