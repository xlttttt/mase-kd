"""Unit tests for PrunePass and ExportMetricsPass.

All tests are CPU-only and require no external downloads.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mase_kd.passes.prune_pass import (
    PruneConfig,
    PrunePass,
    compute_model_sparsity,
    count_nonzero_params,
)
from mase_kd.passes.export_pass import ExportMetricsPass


# ---------------------------------------------------------------------------
# Tiny test models
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv(x).relu().flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# PrunePass tests
# ---------------------------------------------------------------------------


class TestPrunePass:
    def test_achieves_target_sparsity(self):
        model = TinyMLP()
        pass_args = PruneConfig(sparsity=0.5, make_permanent=True)
        pruned, info = PrunePass().run(model, pass_args, {})
        # Actual sparsity should be close to 50% (global unstructured may differ slightly)
        assert 0.40 <= info["sparsity_actual"] <= 0.60, \
            f"Expected ~50% sparsity, got {info['sparsity_actual']:.4f}"

    def test_achieves_high_sparsity(self):
        model = TinyMLP()
        pass_args = PruneConfig(sparsity=0.9, make_permanent=True)
        _, info = PrunePass().run(model, pass_args, {})
        assert 0.85 <= info["sparsity_actual"] <= 0.95

    def test_makes_permanent_no_weight_orig(self):
        """After make_permanent=True there should be no weight_orig parameters."""
        model = TinyMLP()
        PrunePass().run(model, PruneConfig(sparsity=0.5, make_permanent=True), {})
        param_names = [name for name, _ in model.named_parameters()]
        assert not any("weight_orig" in n for n in param_names), \
            "weight_orig still present after make_permanent=True"

    def test_without_make_permanent_has_weight_orig(self):
        """Before removal, weight_orig should exist."""
        model = TinyMLP()
        PrunePass().run(model, PruneConfig(sparsity=0.5, make_permanent=False), {})
        param_names = [name for name, _ in model.named_parameters()]
        assert any("weight_orig" in n for n in param_names), \
            "weight_orig should be present when make_permanent=False"

    def test_info_keys_populated(self):
        model = TinyMLP()
        _, info = PrunePass().run(model, PruneConfig(sparsity=0.3), {})
        assert "sparsity_actual" in info
        assert "params_nonzero" in info
        assert "params_total" in info
        assert info["params_nonzero"] <= info["params_total"]

    def test_merges_existing_info(self):
        model = TinyMLP()
        existing = {"prior_metric": 42.0}
        _, info = PrunePass().run(model, PruneConfig(sparsity=0.5), existing)
        assert info.get("prior_metric") == 42.0
        assert "sparsity_actual" in info

    def test_conv_only_pruning(self):
        model = TinyCNN()
        pass_args = PruneConfig(
            sparsity=0.5, target_types=(nn.Conv2d,), make_permanent=True
        )
        _, info = PrunePass().run(model, pass_args, {})
        # Conv params should be ~50% zero; fc should be untouched
        fc_params = model.fc.weight.data
        assert (fc_params == 0).float().mean() < 0.01, \
            "Linear layer should not be pruned when target_types=(Conv2d,)"

    def test_default_args(self):
        model = TinyMLP()
        _, info = PrunePass().run(model)
        assert info["sparsity_actual"] > 0

    def test_invalid_sparsity_raises(self):
        model = TinyMLP()
        with pytest.raises(ValueError, match="sparsity"):
            PrunePass().run(model, PruneConfig(sparsity=1.0))
        with pytest.raises(ValueError, match="sparsity"):
            PrunePass().run(model, PruneConfig(sparsity=0.0))

    def test_no_matching_layers_raises(self):
        """Pruning a model with no matching layer types should raise."""
        model = nn.Sequential(nn.BatchNorm1d(8))
        with pytest.raises(RuntimeError, match="No layers"):
            PrunePass().run(model, PruneConfig(sparsity=0.5, target_types=(nn.Linear,)))

    def test_state_dict_saveable_after_pruning(self):
        """Pruned model state_dict should be serialisable without errors."""
        import io
        model = TinyMLP()
        PrunePass().run(model, PruneConfig(sparsity=0.5, make_permanent=True), {})
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=True)
        assert "fc1.weight" in loaded


# ---------------------------------------------------------------------------
# compute_model_sparsity / count_nonzero_params tests
# ---------------------------------------------------------------------------


class TestSparsityHelpers:
    def test_all_zero_is_fully_sparse(self):
        model = TinyMLP()
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data.zero_()
        assert compute_model_sparsity(model) == 1.0

    def test_all_nonzero_is_zero_sparsity(self):
        model = TinyMLP()
        # Ensure no weights are accidentally zero
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data.fill_(1.0)
        sp = compute_model_sparsity(model)
        assert sp == 0.0

    def test_known_50_percent_sparsity(self):
        model = TinyMLP()
        for name, param in model.named_parameters():
            if "weight" in name:
                n = param.numel()
                flat = param.data.view(-1)
                flat[: n // 2] = 0.0
                flat[n // 2 :] = 1.0
        sp = compute_model_sparsity(model)
        # Should be very close to 0.5 (bias terms excluded)
        assert abs(sp - 0.5) < 0.01

    def test_count_nonzero_returns_tuple(self):
        model = TinyMLP()
        nonzero, total = count_nonzero_params(model)
        assert isinstance(nonzero, int)
        assert isinstance(total, int)
        assert 0 <= nonzero <= total


# ---------------------------------------------------------------------------
# ExportMetricsPass tests
# ---------------------------------------------------------------------------


_MOCK_RESULTS = {
    "A": {"accuracy": 0.82, "params_nonzero": 11_000_000, "params_total": 11_000_000, "sparsity": 0.0},
    "B": {"accuracy": 0.54, "params_nonzero": 5_500_000, "params_total": 11_000_000, "sparsity": 0.5},
    "C": {"accuracy": 0.78, "params_nonzero": 5_500_000, "params_total": 11_000_000, "sparsity": 0.5},
    "D": {"accuracy": 0.80, "params_nonzero": 5_500_000, "params_total": 11_000_000, "sparsity": 0.5},
    "E": {"accuracy": 0.81, "params_nonzero": 5_500_000, "params_total": 11_000_000, "sparsity": 0.5},
}


class TestExportMetricsPass:
    def test_creates_json_file(self, tmp_path):
        ExportMetricsPass().run(_MOCK_RESULTS, str(tmp_path), "test_model")
        assert (tmp_path / "comparison_table.json").exists()

    def test_creates_md_file(self, tmp_path):
        ExportMetricsPass().run(_MOCK_RESULTS, str(tmp_path), "test_model")
        assert (tmp_path / "comparison_table.md").exists()

    def test_creates_png_file(self, tmp_path):
        ExportMetricsPass().run(_MOCK_RESULTS, str(tmp_path), "test_model")
        assert (tmp_path / "trade_off_plot.png").exists()

    def test_json_contains_all_variants(self, tmp_path):
        ExportMetricsPass().run(_MOCK_RESULTS, str(tmp_path), "test_model")
        data = json.loads((tmp_path / "comparison_table.json").read_text())
        for key in ["A", "B", "C", "D", "E"]:
            assert key in data

    def test_delta_vs_dense_computed(self, tmp_path):
        ExportMetricsPass().run(_MOCK_RESULTS, str(tmp_path), "test_model")
        data = json.loads((tmp_path / "comparison_table.json").read_text())
        # B's delta should be negative (accuracy dropped after pruning)
        assert data["B"]["delta_vs_dense"] < 0
        # A's delta should be ~0
        assert abs(data["A"]["delta_vs_dense"]) < 1e-6

    def test_md_table_has_header(self, tmp_path):
        ExportMetricsPass().run(_MOCK_RESULTS, str(tmp_path), "test_model")
        content = (tmp_path / "comparison_table.md").read_text()
        assert "accuracy" in content
        assert "Dense" in content

    def test_missing_variant_skipped(self, tmp_path):
        partial = {k: v for k, v in _MOCK_RESULTS.items() if k != "E"}
        ExportMetricsPass().run(partial, str(tmp_path), "test_model")
        data = json.loads((tmp_path / "comparison_table.json").read_text())
        assert "E" not in data
        assert "D" in data

    def test_mAP50_primary_metric(self, tmp_path):
        yolo_results = {
            "A": {"mAP50": 0.45, "params_nonzero": 3_200_000, "params_total": 3_200_000, "sparsity": 0.0},
            "B": {"mAP50": 0.22, "params_nonzero": 1_600_000, "params_total": 3_200_000, "sparsity": 0.5},
        }
        ExportMetricsPass().run(yolo_results, str(tmp_path), "yolo", primary_metric="mAP50")
        content = (tmp_path / "comparison_table.md").read_text()
        assert "mAP50" in content


# ---------------------------------------------------------------------------
# Config loading smoke test
# ---------------------------------------------------------------------------


class TestPipelineConfigLoads:
    def test_smoke_yaml_loads(self):
        """resnet18_cifar10_smoke.yaml should parse without errors."""
        import yaml
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "experiments" / "configs" / "resnet18_cifar10_smoke.yaml"
        if not config_path.exists():
            pytest.skip("Smoke config not found (run from repo root)")
        with config_path.open() as fp:
            config = yaml.safe_load(fp)
        assert "dense_training" in config
        assert "pruning" in config
        assert config["pruning"]["sparsity"] == 0.5
        assert config["dense_training"]["epochs"] >= 1

    def test_full_yaml_loads(self):
        """resnet18_cifar10_full.yaml should parse without errors."""
        import yaml
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "experiments" / "configs" / "resnet18_cifar10_full.yaml"
        if not config_path.exists():
            pytest.skip("Full config not found")
        with config_path.open() as fp:
            config = yaml.safe_load(fp)
        assert config["dense_training"]["epochs"] >= 50
        assert config["kd"]["alpha"] == 0.5
        assert "seeds" in config
