#!/usr/bin/env python3
"""Alpha / temperature ablation sweep for BERT and YOLO KD experiments.

Runs a grid of (alpha, temperature) combinations and saves per-run metrics
to a JSON summary file. Useful for generating the ablation table in the report.

Usage:
    # BERT ablation (fast, recommended for first run)
    python experiments/scripts/ablation_sweep.py --task bert

    # YOLO ablation
    python experiments/scripts/ablation_sweep.py --task yolo

    # Custom grid
    python experiments/scripts/ablation_sweep.py --task bert \\
        --alphas 0.3 0.5 0.7 \\
        --temperatures 2.0 4.0 6.0 \\
        --epochs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="KD ablation sweep over alpha and temperature")
    p.add_argument("--task", choices=["bert", "yolo"], required=True)
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help="KD alpha values to sweep (0=baseline, 1=soft-only)",
    )
    p.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 6.0],
        help="Temperature values to sweep",
    )
    # BERT-specific
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    # YOLO-specific
    p.add_argument("--data", default="coco8.yaml")
    p.add_argument("--yolo-epochs", type=int, default=30)
    # Output
    p.add_argument("--output-dir", default="outputs/ablation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# BERT sweep
# ---------------------------------------------------------------------------


def run_bert_point(alpha: float, temperature: float, args) -> dict:
    """Train one BERT KD point and return its best val metrics."""
    import torch
    from mase_kd.nlp.bert_kd import BertKDConfig, BertStudentConfig, build_bert_kd_trainer
    from mase_kd.core.losses import DistillationLossConfig

    out_dir = f"{args.output_dir}/bert_a{alpha:.2f}_t{temperature:.1f}"

    config = BertKDConfig(
        kd=DistillationLossConfig(alpha=alpha, temperature=temperature),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=out_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = build_bert_kd_trainer(config, device=device)
    history = trainer.train()

    best = max(history, key=lambda x: x["val_accuracy"])
    return {
        "alpha": alpha,
        "temperature": temperature,
        "best_epoch": best["epoch"],
        "val_accuracy": best["val_accuracy"],
        "val_f1": best["val_f1"],
        "train_loss": best["train_loss"],
    }


def sweep_bert(args) -> list[dict]:
    results = []
    total = len(args.alphas) * len(args.temperatures)
    idx = 0

    for alpha in args.alphas:
        for temperature in args.temperatures:
            idx += 1
            logger.info(
                "[%d/%d] BERT sweep: alpha=%.2f, T=%.1f",
                idx, total, alpha, temperature,
            )
            try:
                metrics = run_bert_point(alpha, temperature, args)
                results.append(metrics)
                logger.info(
                    "  => accuracy=%.4f, F1=%.4f",
                    metrics["val_accuracy"], metrics["val_f1"],
                )
            except Exception as e:
                logger.error("  => FAILED: %s", e)
                results.append({"alpha": alpha, "temperature": temperature, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# YOLO sweep
# ---------------------------------------------------------------------------


def run_yolo_point(alpha: float, temperature: float, args) -> dict:
    """Train one YOLO KD point and return its best mAP metrics."""
    from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig, YOLOKDRunner
    from mase_kd.core.losses import DistillationLossConfig

    out_dir = f"{args.output_dir}/yolo_a{alpha:.2f}_t{temperature:.1f}"

    config = YOLOTrainingConfig(
        data_yaml=args.data,
        kd=DistillationLossConfig(alpha=alpha, temperature=temperature),
        epochs=args.yolo_epochs,
        output_dir=out_dir,
    )

    runner = YOLOKDRunner(config)
    history = runner.train()

    best = max(history, key=lambda x: x["mAP50"])
    return {
        "alpha": alpha,
        "temperature": temperature,
        "best_epoch": best["epoch"],
        "mAP50": best["mAP50"],
        "mAP50_95": best["mAP50_95"],
    }


def sweep_yolo(args) -> list[dict]:
    results = []
    total = len(args.alphas) * len(args.temperatures)
    idx = 0

    for alpha in args.alphas:
        for temperature in args.temperatures:
            idx += 1
            logger.info(
                "[%d/%d] YOLO sweep: alpha=%.2f, T=%.1f",
                idx, total, alpha, temperature,
            )
            try:
                metrics = run_yolo_point(alpha, temperature, args)
                results.append(metrics)
                logger.info(
                    "  => mAP50=%.4f, mAP50:95=%.4f",
                    metrics["mAP50"], metrics["mAP50_95"],
                )
            except Exception as e:
                logger.error("  => FAILED: %s", e)
                results.append({"alpha": alpha, "temperature": temperature, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_table(results: list[dict], task: str) -> None:
    if task == "bert":
        metric_key = "val_accuracy"
        metric_name = "Accuracy"
    else:
        metric_key = "mAP50"
        metric_name = "mAP@50"

    print(f"\n=== {task.upper()} Ablation Results ===")
    header = f"{'Alpha':>8}  {'Temp':>6}  {metric_name:>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            val_str = "ERROR"
        else:
            val_str = f"{r[metric_key]:.4f}"
        print(f"{r['alpha']:>8.2f}  {r['temperature']:>6.1f}  {val_str:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Ablation sweep: task=%s | alphas=%s | temperatures=%s",
        args.task, args.alphas, args.temperatures,
    )

    if args.task == "bert":
        results = sweep_bert(args)
    else:
        results = sweep_yolo(args)

    # Save to JSON
    summary_path = out_dir / f"{args.task}_ablation_summary.json"
    with summary_path.open("w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Summary saved to: %s", summary_path)

    print_table(results, args.task)


if __name__ == "__main__":
    main()
