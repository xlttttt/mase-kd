"""Unit tests for config schema validation."""

import pytest

from mase_kd.config.schema import ResNetKDConfig, VisionKDConfig
from mase_kd.core.losses import DistillationLossConfig


# ---------------------------------------------------------------------------
# VisionKDConfig
# ---------------------------------------------------------------------------


class TestVisionKDConfig:
    def test_valid_config(self):
        VisionKDConfig(alpha=0.3, temperature=2.5, learning_rate=1e-4).validate()

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            VisionKDConfig(alpha=1.3, temperature=2.0, learning_rate=1e-4).validate()

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            VisionKDConfig(alpha=-0.1, temperature=2.0, learning_rate=1e-4).validate()

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            VisionKDConfig(alpha=0.5, temperature=-1.0, learning_rate=1e-4).validate()

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            VisionKDConfig(alpha=0.5, temperature=2.0, learning_rate=0.0).validate()

    def test_boundary_alpha_zero(self):
        VisionKDConfig(alpha=0.0, temperature=1.0, learning_rate=1e-5).validate()

    def test_boundary_alpha_one(self):
        VisionKDConfig(alpha=1.0, temperature=1.0, learning_rate=1e-5).validate()



# ---------------------------------------------------------------------------
# ResNetKDConfig
# ---------------------------------------------------------------------------


class TestResNetKDConfig:
    def test_default_config_valid(self):
        cfg = ResNetKDConfig()
        cfg.validate()

    def test_invalid_alpha_raises(self):
        cfg = ResNetKDConfig(alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            cfg.validate()

    def test_invalid_alpha_negative(self):
        cfg = ResNetKDConfig(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            cfg.validate()

    def test_invalid_temperature_raises(self):
        cfg = ResNetKDConfig(temperature=-1.0)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_temperature_zero_raises(self):
        cfg = ResNetKDConfig(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_invalid_learning_rate(self):
        cfg = ResNetKDConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            cfg.validate()

    def test_invalid_epochs(self):
        cfg = ResNetKDConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs"):
            cfg.validate()

    def test_invalid_batch_size(self):
        cfg = ResNetKDConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_invalid_val_split_zero(self):
        cfg = ResNetKDConfig(val_split=0.0)
        with pytest.raises(ValueError, match="val_split"):
            cfg.validate()

    def test_invalid_val_split_one(self):
        cfg = ResNetKDConfig(val_split=1.0)
        with pytest.raises(ValueError, match="val_split"):
            cfg.validate()

    def test_boundary_alpha_zero(self):
        ResNetKDConfig(alpha=0.0).validate()

    def test_boundary_alpha_one(self):
        ResNetKDConfig(alpha=1.0).validate()

    def test_custom_fields(self):
        cfg = ResNetKDConfig(
            teacher_weights="outputs/dense.pth",
            teacher_arch="resnet34",
            num_classes=10,
            alpha=0.5,
            temperature=4.0,
            epochs=50,
            batch_size=128,
            learning_rate=0.1,
        )
        cfg.validate()
        assert cfg.teacher_arch == "resnet34"
        assert cfg.temperature == 4.0

    def test_subset_size_none_valid(self):
        cfg = ResNetKDConfig(subset_size=None)
        cfg.validate()

    def test_subset_size_integer_valid(self):
        cfg = ResNetKDConfig(subset_size=500)
        cfg.validate()
