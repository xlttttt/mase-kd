<<<<<<< HEAD:mase-kd/src/mase_kd/nlp/distillation_Bert/__init__.py
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
from .kd_pass import KnowledgeDistillationPass

__all__ = [
    "attention_distillation_loss",
    "HiddenDistillationLoss",
    "prediction_distillation_loss",
    "KnowledgeDistillationPass",
    "generate_layer_mapping"
]
=======
"""Distillation module (reserved for future use)."""
>>>>>>> b6619a167f198f32a3e768af2adfe6bfbd2e08b1:src/mase_kd/distillation/__init__.py
