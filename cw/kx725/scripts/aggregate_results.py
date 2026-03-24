"""Aggregate ResNet18/CIFAR-10 results across multiple sparsity levels.

Reads ``comparison_table.json`` from each sparsity subdirectory and produces:

* ``report_ready_tables/comparison_table_sparsity_{s}.md`` — per-sparsity A–E table
* ``report_ready_tables/combined_table.md`` — sparsity × variant cross-table
* ``report_ready_tables/combined_table.json`` — same as JSON
* ``figures/accuracy_vs_variant.png`` — grouped bar chart (one group per sparsity)
* ``figures/accuracy_vs_sparsity.png`` — line chart (recovery delta vs sparsity)

Usage::

    python experiments/scripts/aggregate_results.py \\
        --model resnet18 --dataset cifar10 \\
        --sparsities 0.5 0.7 \\
        --output-dir outputs/resnet18/cifar10

All output files are written under *output_dir*.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aggregate_results")

_VARIANTS = ["A", "B", "C", "D", "E"]
_VARIANT_LABELS = {
    "A": "A — Dense",
    "B": "B — Pruned",
    "C": "C — Pruned+FT",
    "D": "D — Pruned+KD",
    "E": "E — Pruned+KD+FT",
}


def load_results(output_dir: str, sparsities: list[float]) -> dict[float, dict]:
    """Load comparison_table.json for each sparsity level.

    Falls back to reading individual metrics.json files if the table is absent.
    """
    base = Path(output_dir)
    all_results: dict[float, dict] = {}

    for s in sparsities:
        s_dir = base / f"sparsity_{s:.2f}"
        table_path = s_dir / "comparison_table.json"
        if table_path.exists():
            with table_path.open() as fp:
                all_results[s] = json.load(fp)
            logger.info("Loaded %s", table_path)
        else:
            # Build from individual metrics.json
            per_variant: dict[str, dict] = {}
            for variant_dir in sorted(s_dir.iterdir()):
                metrics_path = variant_dir / "metrics.json"
                if metrics_path.exists():
                    # Map directory names to variant keys
                    dn = variant_dir.name
                    key = dn[0].upper() if dn[0].upper() in _VARIANTS else None
                    if key:
                        with metrics_path.open() as fp:
                            per_variant[key] = json.load(fp)
            if per_variant:
                all_results[s] = per_variant
                logger.info("Assembled results from metrics.json files in %s", s_dir)
            else:
                logger.warning("No results found in %s — skipping sparsity %.2f", s_dir, s)

    return all_results


def write_per_sparsity_tables(
    all_results: dict[float, dict],
    out_dir: Path,
    primary_metric: str = "accuracy",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, results in sorted(all_results.items()):
        lines = [
            f"## Sparsity {s:.0%} — A–E Comparison",
            "",
            f"| Variant | {primary_metric} | delta_vs_dense | params_nonzero | sparsity |",
            "|---|---:|---:|---:|---:|",
        ]
        dense_val = results.get("A", {}).get(primary_metric)
        for key in _VARIANTS:
            if key not in results:
                continue
            m = results[key]
            label = _VARIANT_LABELS.get(key, key)
            pm = m.get(primary_metric, "—")
            pm_str = f"{pm:.4f}" if isinstance(pm, float) else str(pm)
            delta = m.get("delta_vs_dense")
            if delta is None and dense_val is not None and isinstance(pm, float):
                delta = pm - dense_val
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else "—"
            pnz = m.get("params_nonzero", "—")
            sp_val = m.get("sparsity", "—")
            sp_str = f"{sp_val:.4f}" if isinstance(sp_val, float) else str(sp_val)
            lines.append(f"| {label} | {pm_str} | {delta_str} | {pnz} | {sp_str} |")
        table_path = out_dir / f"comparison_table_sparsity_{s:.2f}.md"
        table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote %s", table_path)


def write_combined_table(
    all_results: dict[float, dict],
    out_dir: Path,
    primary_metric: str = "accuracy",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sparsities = sorted(all_results.keys())

    # Markdown
    header_cols = " | ".join(f"s={s:.2f}" for s in sparsities)
    md_lines = [
        f"## Combined: {primary_metric} by Sparsity × Variant",
        "",
        f"| Variant | {header_cols} |",
        "|---" + "|---:" * len(sparsities) + "|",
    ]
    combined_json: dict[str, dict] = {}
    for key in _VARIANTS:
        row_vals = []
        label = _VARIANT_LABELS.get(key, key)
        combined_json[key] = {}
        for s in sparsities:
            pm = all_results.get(s, {}).get(key, {}).get(primary_metric, None)
            combined_json[key][f"sparsity_{s:.2f}"] = pm
            row_vals.append(f"{pm:.4f}" if isinstance(pm, float) else "—")
        md_lines.append(f"| {label} | {' | '.join(row_vals)} |")

    md_path = out_dir / "combined_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", md_path)

    json_path = out_dir / "combined_table.json"
    with json_path.open("w") as fp:
        json.dump(combined_json, fp, indent=2)
    logger.info("Wrote %s", json_path)


def write_figures(
    all_results: dict[float, dict],
    figures_dir: Path,
    primary_metric: str = "accuracy",
    model_name: str = "resnet18",
    dataset_name: str = "cifar10",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping figures")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    sparsities = sorted(all_results.keys())
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    # Figure 1: accuracy vs variant (grouped by sparsity)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_variants = len(_VARIANTS)
    n_sparsities = len(sparsities)
    bar_width = 0.8 / n_sparsities
    x = np.arange(n_variants)

    for i, s in enumerate(sparsities):
        values = [
            all_results[s].get(k, {}).get(primary_metric, 0.0) or 0.0
            for k in _VARIANTS
        ]
        offset = (i - n_sparsities / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=f"sparsity={s:.2f}", color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([_VARIANT_LABELS[k] for k in _VARIANTS], rotation=10, ha="right")
    ax.set_ylabel(primary_metric)
    ax.set_title(f"{model_name.upper()} {dataset_name.upper()} — {primary_metric} across variants by sparsity")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(figures_dir / "accuracy_vs_variant.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", figures_dir / "accuracy_vs_variant.png")

    # Figure 2: recovery delta vs sparsity (line chart per variant)
    linestyles = ["--", "-.", ":", "-"]
    markers = ["s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(_VARIANTS[1:], start=1):  # skip Dense (delta=0 by def)
        deltas = []
        for s in sparsities:
            m = all_results.get(s, {})
            dense_val = m.get("A", {}).get(primary_metric)
            val = m.get(key, {}).get(primary_metric)
            if dense_val is not None and val is not None:
                deltas.append(val - dense_val)
            else:
                deltas.append(None)
        xs = [s for s, d in zip(sparsities, deltas) if d is not None]
        ys = [d for d in deltas if d is not None]
        if xs:
            idx = i - 1
            ax.plot(xs, ys,
                    marker=markers[idx % len(markers)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.8,
                    markersize=7,
                    label=_VARIANT_LABELS[key],
                    color=colors[i % len(colors)])
            # Annotate final point with delta value
            ax.annotate(f"{ys[-1]:+.4f}",
                        xy=(xs[-1], ys[-1]),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=colors[i % len(colors)],
                        va="center")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, label="Dense baseline")
    ax.set_xlabel("Sparsity")
    ax.set_ylabel(f"Δ {primary_metric} vs Dense")
    ax.set_title(f"{model_name.upper()} {dataset_name.upper()} — Recovery delta vs Sparsity")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(figures_dir / "accuracy_vs_sparsity.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", figures_dir / "accuracy_vs_sparsity.png")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate A-E results across sparsity levels."
    )
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.5, 0.7],
                        help="Sparsity levels to aggregate (space-separated, e.g. 0.5 0.7)")
    parser.add_argument("--output-dir", default=None,
                        help="Root results dir (default: outputs/{model}/{dataset})")
    parser.add_argument("--metric", default="accuracy",
                        help="Primary metric key (accuracy or mAP50)")

    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = f"outputs/{args.model}/{args.dataset}"

    logger.info("Aggregating %s/%s | sparsities=%s | dir=%s",
                args.model, args.dataset, args.sparsities, args.output_dir)

    all_results = load_results(args.output_dir, args.sparsities)
    if not all_results:
        logger.error("No results found. Run the pipeline first.")
        return

    out_base = Path(args.output_dir)
    tables_dir = out_base / "report_ready_tables"
    figures_dir = out_base / "figures"

    write_per_sparsity_tables(all_results, tables_dir, args.metric)
    write_combined_table(all_results, tables_dir, args.metric)
    write_figures(all_results, figures_dir, args.metric, args.model, args.dataset)

    logger.info("Aggregation complete.")
    logger.info("  Tables: %s", tables_dir)
    logger.info("  Figures: %s", figures_dir)

    # Print combined table to stdout
    combined_md = tables_dir / "combined_table.md"
    if combined_md.exists():
        print("\n" + combined_md.read_text())


if __name__ == "__main__":
    main()
