"""Unit tests for config schema and BertKDConfig validation."""

import pytest

from mase_kd.config.schema import VisionKDConfig
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
# BertKDConfig
# ---------------------------------------------------------------------------


class TestBertKDConfig:
    def test_default_config_valid(self):
        from mase_kd.nlp.bert_kd import BertKDConfig

        cfg = BertKDConfig()
        cfg.validate()

    def test_invalid_lr(self):
        from mase_kd.nlp.bert_kd import BertKDConfig

        cfg = BertKDConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            cfg.validate()

    def test_invalid_epochs(self):
        from mase_kd.nlp.bert_kd import BertKDConfig

        cfg = BertKDConfig(num_epochs=0)
        with pytest.raises(ValueError, match="num_epochs"):
            cfg.validate()

    def test_kd_config_propagated(self):
        from mase_kd.nlp.bert_kd import BertKDConfig

        cfg = BertKDConfig(kd=DistillationLossConfig(alpha=0.6, temperature=3.0))
        cfg.validate()
        assert cfg.kd.alpha == 0.6
        assert cfg.kd.temperature == 3.0

    def test_student_config_fields(self):
        from mase_kd.nlp.bert_kd import BertKDConfig, BertStudentConfig

        student_cfg = BertStudentConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            intermediate_size=512,
        )
        cfg = BertKDConfig(student=student_cfg)
        cfg.validate()
        assert cfg.student.num_hidden_layers == 2


# ---------------------------------------------------------------------------
# YOLOTrainingConfig
# ---------------------------------------------------------------------------


class TestYOLOTrainingConfig:
    def test_default_config_valid(self):
        from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig

        cfg = YOLOTrainingConfig()
        cfg.validate()

    def test_invalid_epochs(self):
        from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig

        cfg = YOLOTrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs"):
            cfg.validate()

    def test_invalid_batch_size(self):
        from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig

        cfg = YOLOTrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_kd_defaults_populated(self):
        from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig

        cfg = YOLOTrainingConfig()
        assert cfg.kd is not None
        assert 0.0 <= cfg.kd.alpha <= 1.0
        assert cfg.kd.temperature > 0
