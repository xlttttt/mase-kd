import pytest

from mase_kd.config.schema import VisionKDConfig


def test_valid_vision_kd_config() -> None:
    config = VisionKDConfig(alpha=0.3, temperature=2.5, learning_rate=1e-4)
    config.validate()


def test_invalid_alpha_raises() -> None:
    config = VisionKDConfig(alpha=1.3, temperature=2.0, learning_rate=1e-4)
    with pytest.raises(ValueError, match="alpha"):
        config.validate()
