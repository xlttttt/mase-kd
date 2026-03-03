#!/usr/bin/env python3
"""Evaluate saved checkpoints and print a consolidated comparison table.

Loads saved student checkpoints from baseline and KD runs, computes all
metrics (quality + efficiency), and writes a combined results JSON and
Markdown table.

Usage:
    # Evaluate BERT checkpoints
    python experiments/scripts/evaluate_all.py --task bert \\
        --baseline-dir outputs/bert_baseline \\
        --kd-dir outputs/bert_kd

    # Evaluate YOLO checkpoints
    python experiments/scripts/evaluate_all.py --task yolo \\
        --baseline-dir outputs/yolo_baseline \\
        --kd-dir outputs/yolo_kd

    # Compare multiple KD runs (e.g. from ablation)
    python experiments/scripts/evaluate_all.py --task bert \\
        --baseline-dir outputs/bert_baseline \\
        --kd-dir outputs/ablation/bert_a0.50_t4.0
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
    p = argparse.ArgumentParser(description="Evaluate and compare KD checkpoints")
    p.add_argument("--task", choices=["bert", "yolo"], required=True)
    p.add_argument("--baseline-dir", required=True, help="Output dir of baseline run")
    p.add_argument("--kd-dir", required=True, help="Output dir of KD run")
    p.add_argument("--output", default="outputs/comparison_results.json")
    # BERT
    p.add_argument("--teacher", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    # YOLO
    p.add_argument("--data", default="coco8.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


# ---------------------------------------------------------------------------
# BERT evaluation
# ---------------------------------------------------------------------------


def eval_bert_checkpoint(ckpt_dir: str, args) -> dict:
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    from mase_kd.nlp.bert_kd import load_sst2_dataloaders
    from mase_kd.nlp.eval import evaluate_classification, efficiency_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_dir) / "best_student"

    logger.info("Evaluating BERT checkpoint: %s", ckpt_path)
    model = BertForSequenceClassification.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    _, val_loader = load_sst2_dataloaders(tokenizer, args.max_seq_len, args.batch_size)
    quality = evaluate_classification(model, val_loader, device)

    sample = next(iter(val_loader))
    eff = efficiency_report(
        model,
        sample["input_ids"][:1].to(device),
        sample["attention_mask"][:1].to(device),
    )

    return {**quality, **eff, "checkpoint": str(ckpt_path)}


# ---------------------------------------------------------------------------
# YOLO evaluation
# ---------------------------------------------------------------------------


def eval_yolo_checkpoint(ckpt_dir: str, args) -> dict:
    import torch
    from ultralytics import YOLO as UltralyticsYOLO
    from mase_kd.vision.eval import count_parameters, benchmark_forward_latency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_dir) / "best_student.pt"

    logger.info("Evaluating YOLO checkpoint: %s", ckpt_path)
    saved = torch.load(ckpt_path, map_location=device)
    arch = saved.get("arch", "yolov8n.yaml")

    yolo = UltralyticsYOLO(arch)
    yolo.model.load_state_dict(saved["model"])

    # mAP
    results = yolo.val(data=args.data, imgsz=args.imgsz, verbose=False)
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)

    # Efficiency
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    latency = benchmark_forward_latency(yolo.model, dummy)
    n_params = count_parameters(yolo.model)

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "num_parameters": n_params,
        **latency,
        "checkpoint": str(ckpt_path),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison(baseline: dict, kd: dict, task: str) -> None:
    if task == "bert":
        keys = ["accuracy", "f1", "num_parameters", "avg_latency_ms"]
        labels = ["Accuracy", "F1 (macro)", "Parameters", "Latency (ms)"]
        fmts = [".4f", ".4f", "d", ".2f"]
    else:
        keys = ["mAP50", "mAP50_95", "num_parameters", "avg_latency_ms"]
        labels = ["mAP@50", "mAP@50:95", "Parameters", "Latency (ms)"]
        fmts = [".4f", ".4f", "d", ".2f"]

    print(f"\n{'Metric':<20} {'Baseline':>12} {'KD Student':>12} {'Delta':>10}")
    print("-" * 56)
    for key, label, fmt in zip(keys, labels, fmts):
        b_val = baseline.get(key, 0)
        k_val = kd.get(key, 0)
        delta = k_val - b_val
        sign = "+" if delta >= 0 else ""
        if fmt == "d":
            print(f"{label:<20} {b_val:>12,d} {k_val:>12,d} {sign}{delta:>9,d}")
        else:
            print(f"{label:<20} {b_val:>12{fmt}} {k_val:>12{fmt}} {sign}{delta:>9{fmt}}")


def write_markdown_table(baseline: dict, kd: dict, task: str, out_path: Path) -> None:
    if task == "bert":
        keys = ["accuracy", "f1", "num_parameters", "avg_latency_ms"]
        labels = ["Accuracy", "F1 (macro)", "Parameters", "Latency (ms)"]
    else:
        keys = ["mAP50", "mAP50_95", "num_parameters", "avg_latency_ms"]
        labels = ["mAP@50", "mAP@50:95", "Parameters", "Latency (ms)"]

    rows = [
        "| Metric | Baseline | KD Student | Delta |",
        "|--------|---------|-----------|-------|",
    ]
    for key, label in zip(keys, labels):
        b = baseline.get(key, 0)
        k = kd.get(key, 0)
        delta = k - b
        sign = "+" if delta >= 0 else ""
        rows.append(f"| {label} | {b:.4g} | {k:.4g} | {sign}{delta:.4g} |")

    out_path.write_text("\n".join(rows) + "\n")
    logger.info("Markdown table saved to: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task == "bert":
        baseline = eval_bert_checkpoint(args.baseline_dir, args)
        kd = eval_bert_checkpoint(args.kd_dir, args)
    else:
        baseline = eval_yolo_checkpoint(args.baseline_dir, args)
        kd = eval_yolo_checkpoint(args.kd_dir, args)

    comparison = {"baseline": baseline, "kd": kd, "task": args.task}
    with out_path.open("w") as fp:
        json.dump(comparison, fp, indent=2)
    logger.info("Results saved to: %s", out_path)

    print_comparison(baseline, kd, args.task)

    md_path = out_path.with_suffix(".md")
    write_markdown_table(baseline, kd, args.task, md_path)


if __name__ == "__main__":
    main()
