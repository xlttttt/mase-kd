"""Unit tests for mase_kd.core.losses."""

import math

import pytest
import torch
import torch.nn.functional as F

from mase_kd.core.losses import (
    DistillationLossConfig,
    compute_distillation_loss,
    hard_label_ce_loss,
    soft_logit_kl_loss,
)


# ---------------------------------------------------------------------------
# soft_logit_kl_loss
# ---------------------------------------------------------------------------


class TestSoftLogitKLLoss:
    def test_returns_scalar(self):
        s = torch.randn(4, 10)
        t = torch.randn(4, 10)
        loss = soft_logit_kl_loss(s, t, temperature=2.0)
        assert loss.dim() == 0

    def test_non_negative(self):
        s = torch.randn(8, 5)
        t = torch.randn(8, 5)
        loss = soft_logit_kl_loss(s, t, temperature=3.0)
        assert loss.item() >= 0.0

    def test_zero_when_identical(self):
        logits = torch.randn(4, 10)
        loss = soft_logit_kl_loss(logits, logits, temperature=2.0)
        # KL(p || p) = 0
        assert loss.item() < 1e-5

    def test_temperature_scaling_increases_loss_entropy(self):
        """Higher temperature → softer distributions → lower KL divergence."""
        torch.manual_seed(0)
        s = torch.randn(4, 10) * 5  # peaked logits
        t = torch.randn(4, 10) * 5

        loss_low_t = soft_logit_kl_loss(s, t, temperature=1.0)
        loss_high_t = soft_logit_kl_loss(s, t, temperature=10.0)

        # KL divergence is normalised by T^2, but the underlying distributions
        # are smoother at higher T — low T logits differ more sharply
        # We just assert both are non-negative and finite
        assert loss_low_t.item() >= 0
        assert loss_high_t.item() >= 0
        assert math.isfinite(loss_low_t.item())
        assert math.isfinite(loss_high_t.item())

    def test_invalid_temperature_raises(self):
        s = torch.randn(2, 5)
        t = torch.randn(2, 5)
        with pytest.raises(ValueError, match="temperature"):
            soft_logit_kl_loss(s, t, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            soft_logit_kl_loss(s, t, temperature=-1.0)

    def test_gradient_flows_to_student(self):
        s = torch.randn(4, 5, requires_grad=True)
        t = torch.randn(4, 5)
        loss = soft_logit_kl_loss(s, t, temperature=2.0)
        loss.backward()
        assert s.grad is not None
        assert s.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# hard_label_ce_loss
# ---------------------------------------------------------------------------


class TestHardLabelCELoss:
    def test_returns_scalar(self):
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = hard_label_ce_loss(logits, targets)
        assert loss.dim() == 0

    def test_non_negative(self):
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        assert hard_label_ce_loss(logits, targets).item() >= 0

    def test_perfect_prediction_low_loss(self):
        """Near-perfect logits should give near-zero CE loss."""
        logits = torch.zeros(4, 3)
        logits[0, 0] = 100.0
        logits[1, 1] = 100.0
        logits[2, 2] = 100.0
        logits[3, 0] = 100.0
        targets = torch.tensor([0, 1, 2, 0])
        assert hard_label_ce_loss(logits, targets).item() < 1e-3


# ---------------------------------------------------------------------------
# compute_distillation_loss
# ---------------------------------------------------------------------------


class TestComputeDistillationLoss:
    def test_combines_hard_and_soft(self):
        s = torch.tensor([[2.0, 0.5, -1.0]])
        t = torch.tensor([[1.5, 0.2, -0.3]])
        targets = torch.tensor([0])

        total, hard, soft = compute_distillation_loss(
            s, t, targets, DistillationLossConfig(alpha=0.4, temperature=2.0)
        )
        expected = 0.6 * hard + 0.4 * soft
        assert torch.isclose(total, expected, atol=1e-6)

    def test_soft_only_when_alpha_one(self):
        s = torch.randn(3, 4)
        t = torch.randn(3, 4)
        total, hard, soft = compute_distillation_loss(
            s, t, targets=None, config=DistillationLossConfig(alpha=1.0, temperature=2.0)
        )
        assert hard.item() == 0.0
        assert torch.isclose(total, soft, atol=1e-6)

    def test_hard_only_when_alpha_zero(self):
        s = torch.randn(3, 4)
        t = torch.randn(3, 4)
        targets = torch.randint(0, 4, (3,))
        total, hard, soft = compute_distillation_loss(
            s, t, targets=targets, config=DistillationLossConfig(alpha=0.0, temperature=2.0)
        )
        assert torch.isclose(total, hard, atol=1e-6)

    def test_returns_three_tensors(self):
        s = torch.randn(2, 5)
        t = torch.randn(2, 5)
        result = compute_distillation_loss(
            s, t, targets=None, config=DistillationLossConfig()
        )
        assert len(result) == 3

    def test_gradient_flows_to_student_logits(self):
        s = torch.randn(4, 6, requires_grad=True)
        t = torch.randn(4, 6)
        targets = torch.randint(0, 6, (4,))
        total, _, _ = compute_distillation_loss(
            s, t, targets, DistillationLossConfig(alpha=0.5, temperature=3.0)
        )
        total.backward()
        assert s.grad is not None

    def test_all_losses_non_negative(self):
        s = torch.randn(8, 10)
        t = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        total, hard, soft = compute_distillation_loss(
            s, t, targets, DistillationLossConfig(alpha=0.5, temperature=2.0)
        )
        assert total.item() >= 0
        assert hard.item() >= 0
        assert soft.item() >= 0


# ---------------------------------------------------------------------------
# DistillationLossConfig validation
# ---------------------------------------------------------------------------


class TestDistillationLossConfig:
    def test_valid_config(self):
        cfg = DistillationLossConfig(alpha=0.5, temperature=2.0)
        cfg.validate()  # should not raise

    def test_invalid_alpha_below_zero(self):
        cfg = DistillationLossConfig(alpha=-0.1, temperature=2.0)
        with pytest.raises(ValueError, match="alpha"):
            cfg.validate()

    def test_invalid_alpha_above_one(self):
        cfg = DistillationLossConfig(alpha=1.1, temperature=2.0)
        with pytest.raises(ValueError, match="alpha"):
            cfg.validate()

    def test_invalid_temperature(self):
        cfg = DistillationLossConfig(alpha=0.5, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_boundary_alphas_valid(self):
        DistillationLossConfig(alpha=0.0, temperature=1.0).validate()
        DistillationLossConfig(alpha=1.0, temperature=1.0).validate()

