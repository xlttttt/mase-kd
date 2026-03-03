"""CLI entrypoint for simple vision KD configuration checks."""

import argparse

from mase_kd.core.losses import DistillationLossConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for vision KD runs."""
    parser = argparse.ArgumentParser(description="Run vision KD experiments")
    parser.add_argument("--teacher", type=str, default="yolov8m.pt")
    parser.add_argument("--student", type=str, default="yolov8n.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Validate and echo a basic YOLO KD run configuration."""
    args = parse_args()
    config = DistillationLossConfig(alpha=args.alpha, temperature=args.temperature)
    config.validate()
    print(
        f"Prepared YOLO logits KD config: teacher={args.teacher}, student={args.student}, "
        f"alpha={args.alpha}, temperature={args.temperature}"
    )


if __name__ == "__main__":
    main()
