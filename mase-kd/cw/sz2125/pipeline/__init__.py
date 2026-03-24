"""Coursework KD extensions for MASE."""

from .core.losses import DistillationLossConfig, compute_distillation_loss

__all__ = ["DistillationLossConfig", "compute_distillation_loss"]
