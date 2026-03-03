"""Dataclass schemas for coursework KD experiment configuration."""

from dataclasses import dataclass


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
