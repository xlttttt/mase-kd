"""
Knowledge Distillation Core Module.
Contains loss functions and mapping utilities for Teacher-Student distillation.
"""

from .losses import (
    attention_distillation_loss,
    HiddenDistillationLoss,
    prediction_distillation_loss
)
from .mapping import generate_layer_mapping

__all__ = [
    "attention_distillation_loss",
    "HiddenDistillationLoss",
    "prediction_distillation_loss",
    "generate_layer_mapping"
]