"""Integration smoke test for BertPipeline A-E orchestration.

Uses toy BERT models and synthetic data so the test runs on CPU with no
network downloads.  The HuggingFace factory (``build_bert_kd_trainer``) is
patched so that each pipeline step gets a real ``BertKDTrainer`` backed by
tiny models and a dict-style DataLoader built from random tensors.

Run with:
    pytest cw/integration/test_bert_pipeline_smoke.py -v
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertForSequenceClassification

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.bert_kd import BertKDConfig, BertKDTrainer
from mase_kd.passes.pipeline import BertPipeline


# ---------------------------------------------------------------------------
# Pipeline config (mirrors a minimal YAML structure)
# ---------------------------------------------------------------------------

_PIPELINE_CFG = {
    "teacher": {"model_name": "toy-teacher"},
    "student": {
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 2,
        "intermediate_size": 64,
    },
    "dense_training": {"epochs": 1, "batch_size": 4, "learning_rate": 1e-4, "seed": 0},
    "pruning": {},
    "finetune": {"epochs": 1, "batch_size": 4, "learning_rate": 1e-4},
    "kd": {"alpha": 0.5, "temperature": 4.0, "epochs": 1, "batch_size": 4, "learning_rate": 1e-4},
    "kd_finetune": {"epochs": 1, "batch_size": 4, "learning_rate": 1e-4},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_bert_config() -> BertConfig:
    return BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        num_labels=2,
    )


class _DictLoader:
    """Wraps a TensorDataset loader into the dict format BertKDTrainer expects."""

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __iter__(self):
        for input_ids, attention_mask, labels in self._loader:
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self) -> int:
        return len(self._loader)


def _make_toy_loader(n: int = 16, seq: int = 16, batch: int = 4) -> _DictLoader:
    input_ids = torch.randint(0, 100, (n, seq))
    attention_mask = torch.ones(n, seq, dtype=torch.long)
    labels = torch.randint(0, 2, (n,))
    return _DictLoader(DataLoader(TensorDataset(input_ids, attention_mask, labels), batch_size=batch))


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------


def _make_trainer_factory(teacher: BertForSequenceClassification):
    """Return a drop-in replacement for ``build_bert_kd_trainer``.

    Each call receives the pipeline's real ``BertKDConfig`` (with the correct
    ``output_dir``) so checkpoints land in the expected locations.  Teacher and
    student are tiny toy models; data loaders use synthetic tensors.
    The ``dev`` argument (forwarded from the pipeline) is used so that the
    trainer's internal device matches the pipeline's device, avoiding any
    CPU/CUDA mismatch when the pruned student is moved in step B.
    """
    toy_loader = _make_toy_loader()

    def factory(cfg: BertKDConfig, dev=None):
        device = dev if dev is not None else torch.device("cpu")
        real_cfg = BertKDConfig(
            output_dir=cfg.output_dir,
            num_epochs=1,
            kd=cfg.kd,
        )
        student = BertForSequenceClassification(_tiny_bert_config())
        return BertKDTrainer(
            config=real_cfg,
            teacher=teacher,
            student=student,
            train_loader=toy_loader,
            val_loader=toy_loader,
            device=device,
        )

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBertPipelineSmoke:
    @pytest.fixture
    def toy_teacher(self):
        return BertForSequenceClassification(_tiny_bert_config())

    def test_pipeline_ae_completes(self, toy_teacher, tmp_path):
        """All five steps should run and return a dict with keys A-E."""
        factory = _make_trainer_factory(toy_teacher)
        with patch("mase_kd.nlp.bert_kd.build_bert_kd_trainer", side_effect=factory):
            results = BertPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.3)

        assert set(results.keys()) == {"A", "B", "C", "D", "E"}, (
            f"Expected keys A-E, got {set(results.keys())}"
        )

    def test_step_a_accuracy_in_range(self, toy_teacher, tmp_path):
        """Step A accuracy must be a valid probability."""
        factory = _make_trainer_factory(toy_teacher)
        with patch("mase_kd.nlp.bert_kd.build_bert_kd_trainer", side_effect=factory):
            results = BertPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.3)

        acc = results["A"]["accuracy"]
        assert 0.0 <= acc <= 1.0, f"Step A accuracy out of range: {acc}"

    def test_step_b_has_real_accuracy(self, toy_teacher, tmp_path):
        """Step B must not have the old hardcoded 0.0 accuracy."""
        factory = _make_trainer_factory(toy_teacher)
        with patch("mase_kd.nlp.bert_kd.build_bert_kd_trainer", side_effect=factory):
            results = BertPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.3)

        b = results["B"]
        assert "accuracy" in b
        assert 0.0 <= b["accuracy"] <= 1.0, f"Step B accuracy out of range: {b['accuracy']}"
        assert b["sparsity"] > 0.0, "Step B should report non-zero sparsity after pruning"

    def test_abcd_steps_have_params(self, toy_teacher, tmp_path):
        """Steps A-D must always contain sparsity and param-count keys.

        Step E may use the fallback metrics dict (no params) if D's best
        checkpoint was not saved (e.g., accuracy never improved above 0.0 on
        toy data), so it is checked separately.
        """
        factory = _make_trainer_factory(toy_teacher)
        with patch("mase_kd.nlp.bert_kd.build_bert_kd_trainer", side_effect=factory):
            results = BertPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.3)

        for step in "ABCD":
            m = results[step]
            assert "params_total" in m, f"Step {step} missing params_total"
            assert m["params_total"] > 0, f"Step {step} params_total should be > 0"

        # E is allowed to be the fallback {"accuracy": 0.0, "sparsity": ...}
        assert "accuracy" in results["E"]

    def test_comparison_table_created(self, toy_teacher, tmp_path):
        """ExportMetricsPass should produce comparison_table.md and trade_off_plot.png."""
        factory = _make_trainer_factory(toy_teacher)
        with patch("mase_kd.nlp.bert_kd.build_bert_kd_trainer", side_effect=factory):
            BertPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.3)

        sparsity_dir = tmp_path / "sparsity_0.30"
        assert (sparsity_dir / "comparison_table.md").exists()
        assert (sparsity_dir / "comparison_table.json").exists()
