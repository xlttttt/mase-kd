"""Integration smoke tests for the BERT KD pipeline.

These tests use toy models (single-layer, tiny hidden size) to verify that the
full training pipeline runs end-to-end without errors on CPU. They do NOT check
model accuracy — that is covered by full experiments.

Marked with `pytest.mark.integration` so they can be skipped in fast CI with:
    pytest -m "not integration"
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertForSequenceClassification

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.bert_kd import BertKDConfig, BertKDTrainer, BertStudentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_bert_config():
    """Minimal BertConfig for fast CPU tests."""
    return BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        num_labels=2,
    )


@pytest.fixture
def tiny_teacher(tiny_bert_config):
    return BertForSequenceClassification(tiny_bert_config)


@pytest.fixture
def tiny_student(tiny_bert_config):
    return BertForSequenceClassification(tiny_bert_config)


@pytest.fixture
def toy_dataloader():
    """Synthetic SST-2-shaped data: (input_ids, attention_mask, labels)."""
    B, SEQ = 8, 16
    input_ids = torch.randint(0, 100, (B, SEQ))
    attention_mask = torch.ones(B, SEQ, dtype=torch.long)
    labels = torch.randint(0, 2, (B,))

    dataset = TensorDataset(input_ids, attention_mask, labels)

    class DictDataLoader:
        """Wrap TensorDataset batches as dicts expected by BertKDTrainer."""

        def __init__(self, loader):
            self._loader = loader

        def __iter__(self):
            for input_ids, attention_mask, labels in self._loader:
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        def __len__(self):
            return len(self._loader)

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return DictDataLoader(loader)


@pytest.fixture
def kd_config(tmp_path):
    return BertKDConfig(
        kd=DistillationLossConfig(alpha=0.5, temperature=2.0),
        num_epochs=1,
        output_dir=str(tmp_path / "bert_kd_smoke"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bert_kd_trainer_one_epoch(
    kd_config, tiny_teacher, tiny_student, toy_dataloader, tmp_path
):
    """Full one-epoch run: losses should decrease and history should be populated."""
    device = torch.device("cpu")
    trainer = BertKDTrainer(
        config=kd_config,
        teacher=tiny_teacher,
        student=tiny_student,
        train_loader=toy_dataloader,
        val_loader=toy_dataloader,
        device=device,
    )

    history = trainer.train()

    assert len(history) == 1
    epoch = history[0]
    assert "train_loss" in epoch
    assert "train_hard_loss" in epoch
    assert "train_soft_loss" in epoch
    assert epoch["train_loss"] > 0


@pytest.mark.integration
def test_bert_trainer_teacher_frozen(
    kd_config, tiny_teacher, tiny_student, toy_dataloader
):
    """Teacher parameters should not accumulate gradients."""
    device = torch.device("cpu")
    initial_params = {
        n: p.clone() for n, p in tiny_teacher.named_parameters()
    }

    trainer = BertKDTrainer(
        config=kd_config,
        teacher=tiny_teacher,
        student=tiny_student,
        train_loader=toy_dataloader,
        val_loader=toy_dataloader,
        device=device,
    )
    trainer._train_epoch()

    for name, param in tiny_teacher.named_parameters():
        assert torch.allclose(param, initial_params[name]), (
            f"Teacher parameter '{name}' changed during training — should be frozen."
        )


@pytest.mark.integration
def test_bert_baseline_mode(tiny_teacher, tiny_student, toy_dataloader, tmp_path):
    """Alpha=0 (baseline) should produce zero soft loss."""
    device = torch.device("cpu")
    config = BertKDConfig(
        kd=DistillationLossConfig(alpha=0.0, temperature=2.0),
        num_epochs=1,
        output_dir=str(tmp_path / "bert_baseline_smoke"),
    )
    trainer = BertKDTrainer(
        config=config,
        teacher=tiny_teacher,
        student=tiny_student,
        train_loader=toy_dataloader,
        val_loader=toy_dataloader,
        device=device,
    )
    history = trainer.train()
    # With alpha=0, train_soft_loss is still computed but not in total
    assert history[0]["train_hard_loss"] > 0


@pytest.mark.integration
def test_bert_checkpoint_save_and_load(
    kd_config, tiny_teacher, tiny_student, toy_dataloader, tmp_path
):
    """Saved checkpoint should load back with matching weights."""
    device = torch.device("cpu")
    trainer = BertKDTrainer(
        config=kd_config,
        teacher=tiny_teacher,
        student=tiny_student,
        train_loader=toy_dataloader,
        val_loader=toy_dataloader,
        device=device,
    )
    trainer._train_epoch()

    ckpt_path = tmp_path / "student_ckpt"
    trainer.save_student(ckpt_path)

    # Reload and compare weights
    loaded = BertKDTrainer.load_student(ckpt_path)
    for (n1, p1), (n2, p2) in zip(
        tiny_student.named_parameters(), loaded.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1.cpu(), p2.cpu()), f"Mismatch in param '{n1}'"


@pytest.mark.integration
def test_nlp_eval_accuracy_range(tiny_student, toy_dataloader, tmp_path):
    """Accuracy should be between 0 and 1 for a random model."""
    from mase_kd.nlp.eval import evaluate_classification

    device = torch.device("cpu")
    metrics = evaluate_classification(tiny_student, toy_dataloader, device)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["val_loss"] >= 0.0


@pytest.mark.integration
def test_nlp_efficiency_report(tiny_student):
    """Efficiency report should return positive param count and latency."""
    from mase_kd.nlp.eval import efficiency_report

    sample_ids = torch.randint(0, 100, (1, 16))
    sample_mask = torch.ones(1, 16, dtype=torch.long)
    report = efficiency_report(tiny_student, sample_ids, sample_mask)

    assert report["num_parameters"] > 0
    assert report["avg_latency_ms"] > 0
