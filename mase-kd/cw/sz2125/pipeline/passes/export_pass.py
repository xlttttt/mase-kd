"""Metrics export pass.

Reads per-variant ``metrics.json`` files produced by the A–E pipeline,
generates a Markdown comparison table, a JSON summary, and a bar-chart PNG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered variant labels used in the comparison table
_VARIANT_ORDER = ["A", "B", "C", "D", "E"]
_VARIANT_LABELS = {
    "A": "A — Dense",
    "B": "B — Pruned",
    "C": "C — Pruned+FT",
    "D": "D — Pruned+KD",
    "E": "E — Pruned+KD+FT",
}


class ExportMetricsPass:
    """Aggregate A–E metrics, write comparison table and trade-off plot.

    Usage::

        results = {"A": {...}, "B": {...}, "C": {...}, "D": {...}, "E": {...}}
        ExportMetricsPass().run(results, output_dir="outputs/resnet_pipeline",
                                model_name="resnet18")
    """

    def run(
        self,
        results: dict[str, dict],
        output_dir: str,
        model_name: str = "model",
        primary_metric: str = "accuracy",
    ) -> None:
        """Write comparison_table.{md,json} and trade_off_plot.png.

        Args:
            results:        Dict mapping variant key ("A"–"E") to metrics dict.
            output_dir:     Directory where files are written.
            model_name:     Short name used in titles (e.g. "resnet18").
            primary_metric: Key for the quality metric column
                            ("accuracy" for BERT/ResNet, "mAP50" for YOLO).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Enrich results with delta_vs_dense
        # ------------------------------------------------------------------
        dense_metric = results.get("A", {}).get(primary_metric)
        enriched: dict[str, dict] = {}
        for key in _VARIANT_ORDER:
            if key not in results:
                continue
            m = dict(results[key])
            if dense_metric is not None and primary_metric in m:
                m["delta_vs_dense"] = round(m[primary_metric] - dense_metric, 6)
            enriched[key] = m

        # ------------------------------------------------------------------
        # JSON summary
        # ------------------------------------------------------------------
        json_path = out / "comparison_table.json"
        with json_path.open("w") as fp:
            json.dump(enriched, fp, indent=2)
        logger.info("Wrote %s", json_path)

        # ------------------------------------------------------------------
        # Markdown table
        # ------------------------------------------------------------------
        md_path = out / "comparison_table.md"
        self._write_markdown(enriched, md_path, model_name, primary_metric)
        logger.info("Wrote %s", md_path)

        # ------------------------------------------------------------------
        # Trade-off plot
        # ------------------------------------------------------------------
        plot_path = out / "trade_off_plot.png"
        self._write_plot(enriched, plot_path, model_name, primary_metric)
        logger.info("Wrote %s", plot_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_markdown(
        enriched: dict[str, dict],
        path: Path,
        model_name: str,
        primary_metric: str,
    ) -> None:
        lines = [
            f"# {model_name} — A–E Experimental Comparison",
            "",
            f"| Variant | {primary_metric} | delta_vs_dense | params_nonzero | params_total | sparsity |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key in _VARIANT_ORDER:
            if key not in enriched:
                continue
            m = enriched[key]
            label = _VARIANT_LABELS.get(key, key)
            pm = m.get(primary_metric, "—")
            pm_str = f"{pm:.4f}" if isinstance(pm, float) else str(pm)
            delta = m.get("delta_vs_dense", "—")
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
            pnz = m.get("params_nonzero", "—")
            ptot = m.get("params_total", "—")
            sp = m.get("sparsity", "—")
            sp_str = f"{sp:.4f}" if isinstance(sp, float) else str(sp)
            lines.append(
                f"| {label} | {pm_str} | {delta_str} | {pnz} | {ptot} | {sp_str} |"
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_plot(
        enriched: dict[str, dict],
        path: Path,
        model_name: str,
        primary_metric: str,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping trade_off_plot.png")
            return

        keys = [k for k in _VARIANT_ORDER if k in enriched]
        values = [enriched[k].get(primary_metric, 0.0) for k in keys]
        labels = [_VARIANT_LABELS.get(k, k) for k in keys]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
        bars = ax.bar(labels, values, color=colors[: len(keys)], edgecolor="black", linewidth=0.6)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylabel(primary_metric)
        ax.set_title(f"{model_name} — {primary_metric} across A–E variants")
        ax.set_ylim(0, max(values) * 1.12 if values else 1.0)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Utility: load metrics from a directory tree
# ---------------------------------------------------------------------------


def load_metrics_from_dir(output_dir: str) -> dict[str, dict]:
    """Read all ``*/metrics.json`` files in *output_dir*, keyed by parent dir name.

    Returns a mapping like ``{"A_dense": {...}, "B_pruned": {...}, ...}``.
    """
    base = Path(output_dir)
    result: dict[str, dict] = {}
    for metrics_file in sorted(base.glob("*/metrics.json")):
        key = metrics_file.parent.name
        with metrics_file.open() as fp:
            result[key] = json.load(fp)
    return result
