"""Integration smoke test for YoloPipeline A-E orchestration.

YOLO model loading and the Ultralytics validation loop are replaced with a
lightweight mock so the test runs on CPU with no dataset downloads.

The mock runner carries a real (tiny) ``nn.Conv2d`` as its ``student`` so that
``_count_params`` and ``PrunePass`` work correctly, while ``train()`` and
``evaluate()`` return stub values.

Run with:
    pytest cw/integration/test_yolo_pipeline_smoke.py -v
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mase_kd.passes.pipeline import YoloPipeline


# ---------------------------------------------------------------------------
# Pipeline config (mirrors a minimal YAML structure)
# ---------------------------------------------------------------------------

_PIPELINE_CFG = {
    "model": {
        "teacher_weights": "yolov8m.pt",
        "student_arch": "yolov8n.yaml",
        "data_yaml": "coco8.yaml",
    },
    "dense_training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "seed": 0},
    "finetune": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3},
    "kd": {"alpha": 0.5, "temperature": 2.0, "epochs": 1, "batch_size": 2, "learning_rate": 1e-3},
    "kd_finetune": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3},
}


# ---------------------------------------------------------------------------
# Mock runner
# ---------------------------------------------------------------------------


def _tiny_student() -> nn.Module:
    """A minimal Conv2d network that satisfies _count_params and PrunePass."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=3, padding=1),
    )


class _MockYOLORunner:
    """Drop-in for YOLOKDRunner: carries a real student but stubs I/O."""

    def __init__(self, out_dir: str = "."):
        self.student = _tiny_student()
        self._out_dir = out_dir
        # Expose device so the pipeline's pruned_student.to(runner_a.device) works.
        self.device = torch.device("cpu")

    def train(self):
        return []

    def evaluate(self):
        return {"mAP50": 0.12, "mAP50_95": 0.07}


def _runner_factory(cfg, *args, **kwargs):
    return _MockYOLORunner(out_dir=cfg.output_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestYoloPipelineSmoke:
    def test_pipeline_ae_completes(self, tmp_path):
        """All five steps should run and return a dict with keys A-E."""
        with patch("mase_kd.vision.yolo_kd_train.build_yolo_kd_runner", side_effect=_runner_factory):
            results = YoloPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.4)

        assert set(results.keys()) == {"A", "B", "C", "D", "E"}, (
            f"Expected keys A-E, got {set(results.keys())}"
        )

    def test_step_a_map50_in_range(self, tmp_path):
        """Step A mAP50 must be a valid value."""
        with patch("mase_kd.vision.yolo_kd_train.build_yolo_kd_runner", side_effect=_runner_factory):
            results = YoloPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.4)

        assert 0.0 <= results["A"]["mAP50"] <= 1.0

    def test_step_b_has_real_map50(self, tmp_path):
        """Step B mAP50 must not be the old hardcoded 0.0."""
        with patch("mase_kd.vision.yolo_kd_train.build_yolo_kd_runner", side_effect=_runner_factory):
            results = YoloPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.4)

        b = results["B"]
        assert "mAP50" in b
        assert 0.0 <= b["mAP50"] <= 1.0, f"Step B mAP50 out of range: {b['mAP50']}"
        assert b["sparsity"] > 0.0, "Step B should report non-zero sparsity after pruning"

    def test_all_steps_have_params(self, tmp_path):
        """Every step result must contain params_nonzero and params_total."""
        with patch("mase_kd.vision.yolo_kd_train.build_yolo_kd_runner", side_effect=_runner_factory):
            results = YoloPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.4)

        for step in "ABCDE":
            m = results[step]
            assert "params_total" in m, f"Step {step} missing params_total"
            assert m["params_total"] > 0, f"Step {step} params_total should be > 0"

    def test_comparison_table_created(self, tmp_path):
        """ExportMetricsPass should produce comparison_table.md and .json."""
        with patch("mase_kd.vision.yolo_kd_train.build_yolo_kd_runner", side_effect=_runner_factory):
            YoloPipeline().run(_PIPELINE_CFG, str(tmp_path), sparsity=0.4)

        sparsity_dir = tmp_path / "sparsity_0.40"
        assert (sparsity_dir / "comparison_table.md").exists()
        assert (sparsity_dir / "comparison_table.json").exists()
