"""Dataclass schemas for coursework KD experiment configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class VisionKDConfig:
    """Configuration for first-pass vision distillation runs."""

    alpha: float = 0.5
    temperature: float = 2.0
    learning_rate: float = 1e-4

    def validate(self) -> None:
        """Validate numerical ranges for KD hyperparameters."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")


@dataclass
class ResNetKDConfig:
    """Configuration schema for ResNet18/CIFAR-10 KD experiments.

    This is a lightweight re-export of
    :class:`mase_kd.vision.resnet_kd.ResNetKDConfig` so that the schema
    module provides a single source of truth for config validation.
    """

    # Teacher
    teacher_weights: str = ""       # path to .pth; empty → no teacher (alpha=0)
    teacher_arch: str = "resnet18"  # "resnet18" or "resnet34"

    # Student
    student_weights: Optional[str] = None
    num_classes: int = 10

    # KD
    alpha: float = 0.5
    temperature: float = 4.0

    # Training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42

    # Data
    data_dir: str = "datasets/cifar10"
    val_split: float = 0.1
    subset_size: Optional[int] = None

    # I/O
    output_dir: str = "outputs/resnet_kd"

    def validate(self) -> None:
        """Validate all fields."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError("val_split must be in (0, 1)")
        if self.teacher_weights and self.alpha > 0 and not self.teacher_weights:
            raise ValueError("teacher_weights must be set when alpha > 0")
