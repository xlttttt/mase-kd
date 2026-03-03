#!/usr/bin/env python3
"""Train YOLOv8n student baseline (no KD) from the yolo_baseline.yaml config.

Usage:
    python experiments/scripts/run_yolo_baseline.py
    python experiments/scripts/run_yolo_baseline.py --config experiments/configs/yolo_baseline.yaml
    python experiments/scripts/run_yolo_baseline.py --data coco.yaml --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="YOLO baseline (no KD) training")
    p.add_argument("--config", default="experiments/configs/yolo_baseline.yaml")
    p.add_argument("--data", default=None, help="Override data_yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data is not None:
        cfg_dict["data_yaml"] = args.data
    if args.epochs is not None:
        cfg_dict["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg_dict["training"]["batch_size"] = args.batch_size
    if args.imgsz is not None:
        cfg_dict["training"]["imgsz"] = args.imgsz
    if args.output_dir is not None:
        cfg_dict["output_dir"] = args.output_dir

    from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig, YOLOKDRunner
    from mase_kd.core.losses import DistillationLossConfig

    kd_dict = cfg_dict.get("kd", {})
    training_dict = cfg_dict.get("training", {})

    config = YOLOTrainingConfig(
        teacher_weights=cfg_dict["teacher_weights"],
        student_arch=cfg_dict["student_arch"],
        student_weights=cfg_dict.get("student_weights"),
        data_yaml=cfg_dict["data_yaml"],
        kd=DistillationLossConfig(
            alpha=kd_dict.get("alpha", 0.0),  # 0.0 = no KD
            temperature=kd_dict.get("temperature", 2.0),
        ),
        epochs=training_dict.get("epochs", 50),
        batch_size=training_dict.get("batch_size", 16),
        imgsz=training_dict.get("imgsz", 640),
        learning_rate=training_dict.get("learning_rate", 1e-3),
        weight_decay=training_dict.get("weight_decay", 5e-4),
        warmup_epochs=training_dict.get("warmup_epochs", 3.0),
        use_amp=training_dict.get("use_amp", True),
        workers=training_dict.get("workers", 8),
        seed=training_dict.get("seed", 42),
        output_dir=cfg_dict.get("output_dir", "outputs/yolo_baseline"),
    )

    runner = YOLOKDRunner(config)
    history = runner.train()

    best = max(history, key=lambda x: x["mAP50"])
    print("\n=== YOLO Baseline Results ===")
    print(f"Best epoch:   {best['epoch']}")
    print(f"mAP@50:       {best['mAP50']:.4f}")
    print(f"mAP@50:95:    {best['mAP50_95']:.4f}")
    print(f"Outputs:      {config.output_dir}")


if __name__ == "__main__":
    main()
