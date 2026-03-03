"""CLI entry point for YOLO knowledge distillation experiments."""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLO object-detection KD experiment"
    )

    # Models
    p.add_argument("--teacher", default="yolov8m.pt", help="Teacher model checkpoint")
    p.add_argument("--student-arch", default="yolov8n.yaml", help="Student architecture YAML")
    p.add_argument("--student-weights", default=None, help="Optional student pretrained weights")

    # Dataset
    p.add_argument("--data", default="coco8.yaml", help="Ultralytics dataset YAML")

    # KD hyperparameters
    p.add_argument("--alpha", type=float, default=0.5, help="KD mixing weight (0=hard only, 1=soft only)")
    p.add_argument("--temperature", type=float, default=2.0, help="Softmax temperature for KD")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training")

    # I/O
    p.add_argument("--output-dir", default="outputs/yolo_kd")
    p.add_argument("--baseline", action="store_true", help="Train without KD (alpha=0)")

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    from mase_kd.vision.yolo_kd_train import YOLOTrainingConfig, YOLOKDRunner
    from mase_kd.core.losses import DistillationLossConfig

    alpha = 0.0 if args.baseline else args.alpha

    config = YOLOTrainingConfig(
        teacher_weights=args.teacher,
        student_arch=args.student_arch,
        student_weights=args.student_weights,
        data_yaml=args.data,
        kd=DistillationLossConfig(alpha=alpha, temperature=args.temperature),
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        learning_rate=args.lr,
        workers=args.workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        output_dir=args.output_dir,
    )

    logger.info(
        "Starting %s experiment | alpha=%.2f | T=%.1f | epochs=%d",
        "BASELINE" if args.baseline else "KD",
        alpha,
        args.temperature,
        args.epochs,
    )

    runner = YOLOKDRunner(config)
    history = runner.train()

    best = max(history, key=lambda x: x["mAP50"])
    logger.info("=== Final Results ===")
    logger.info("Best epoch:  %d", best["epoch"])
    logger.info("mAP@50:      %.4f", best["mAP50"])
    logger.info("mAP@50:95:   %.4f", best["mAP50_95"])


if __name__ == "__main__":
    main()
