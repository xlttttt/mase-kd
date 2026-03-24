"""Unified one-command pipeline runner for the A-E experimental matrix.

Usage (smoke)::

    python -m mase_kd.runners.run_pipeline \\
        --model resnet18 --dataset cifar10 \\
        --sparsity 0.5 --profile smoke

Usage (full, seed 0)::

    python -m mase_kd.runners.run_pipeline \\
        --model resnet18 --dataset cifar10 \\
        --sparsity 0.5 --profile full --seed 0

The runner:
1. Loads ``cw/kx725/configs/{model}_{dataset}_{profile}.yaml``
2. Applies any CLI overrides (--sparsity, --output-dir, --seed, --alpha, --temperature)
3. Dispatches to ``ResNetPipeline``
4. Calls ``ExportMetricsPass`` to write comparison_table.{md,json} + trade_off_plot.png
5. Prints the Markdown comparison table to stdout
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

# Repo root (three levels up from src/mase_kd/runners/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_DIR = _REPO_ROOT / "cw" / "kx725" / "configs"


def _load_config(model: str, dataset: str, profile: str) -> dict:
    config_path = _CONFIGS_DIR / f"{model}_{dataset}_{profile}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Expected file: cw/kx725/configs/{model}_{dataset}_{profile}.yaml"
        )
    with config_path.open() as fp:
        return yaml.safe_load(fp)


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides to the loaded config dict in-place."""
    # sparsity
    if args.sparsity is not None:
        config.setdefault("pruning", {})["sparsity"] = args.sparsity

    # seed: propagates into all training sub-configs
    if args.seed is not None:
        for section in ("dense_training", "finetune", "kd", "kd_finetune"):
            config.setdefault(section, {})["seed"] = args.seed
        config.setdefault("dense_training", {})["seed"] = args.seed

    # alpha / temperature: propagate into kd section
    if args.alpha is not None:
        config.setdefault("kd", {})["alpha"] = args.alpha
    if args.temperature is not None:
        config.setdefault("kd", {})["temperature"] = args.temperature

    # output_dir override
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir

    return config


def _print_table(output_dir: str) -> None:
    """Print the Markdown comparison table to stdout if it exists."""
    table_path = Path(output_dir) / "comparison_table.md"
    # Walk into sparsity subdirs if needed
    if not table_path.exists():
        candidates = list(Path(output_dir).glob("sparsity_*/comparison_table.md"))
        if candidates:
            table_path = sorted(candidates)[-1]  # most recent
    if table_path.exists():
        print("\n" + table_path.read_text())
    else:
        logger.warning("comparison_table.md not found in %s", output_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the A-E knowledge distillation pipeline for a single model."
    )
    parser.add_argument("--model", required=True,
                        choices=["resnet18"],
                        help="Model identifier")
    parser.add_argument("--dataset", required=True,
                        choices=["cifar10", "cifar100"],
                        help="Dataset identifier")
    parser.add_argument("--profile", default="smoke",
                        choices=["smoke", "full"],
                        help="Config profile: 'smoke' for quick validation, 'full' for report-ready run")
    parser.add_argument("--sparsity", type=float, default=None,
                        help="Pruning sparsity [0,1). Overrides config value.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory from config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Overrides config value.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="KD alpha weight for soft loss. Overrides config value.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="KD temperature. Overrides config value.")
    parser.add_argument("--config", default=None,
                        help="Explicit path to YAML config (bypasses auto-detection).")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    if args.config:
        config_path = Path(args.config)
        with config_path.open() as fp:
            config = yaml.safe_load(fp)
        logger.info("Loaded config: %s", config_path)
    else:
        # Map (model, dataset) to config filename prefix
        prefix_map = {
            ("resnet18", "cifar10"):  "resnet18_cifar10",
            ("resnet18", "cifar100"): "resnet18_cifar100",
        }
        prefix = prefix_map.get((args.model, args.dataset))
        if prefix is None:
            logger.error("No config mapping for model=%s dataset=%s", args.model, args.dataset)
            sys.exit(1)
        config_path = _CONFIGS_DIR / f"{prefix}_{args.profile}.yaml"
        if not config_path.exists():
            # Try the nested config dirs from the original plan (bert/pipeline.toml etc.)
            logger.error("Config not found: %s", config_path)
            sys.exit(1)
        with config_path.open() as fp:
            config = yaml.safe_load(fp)
        logger.info("Loaded config: %s", config_path)

    config = _apply_overrides(config, args)

    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    out_dir = config.get("output", {}).get("dir", f"cw/kx725/outputs/{args.model}/{args.dataset}")
    sparsity = config.get("pruning", {}).get("sparsity", 0.5)
    logger.info("Output dir: %s | sparsity=%.2f", out_dir, sparsity)

    # ------------------------------------------------------------------
    # Dispatch to pipeline
    # ------------------------------------------------------------------
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.model == "resnet18" and args.dataset in ("cifar10", "cifar100"):
        from mase_kd.passes.pipeline import ResNetPipeline
        results = ResNetPipeline().run(config, out_dir, sparsity, device)
        primary_metric = "accuracy"

    else:
        logger.error("Unsupported combination: model=%s dataset=%s", args.model, args.dataset)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    sparsity_dir = str(Path(out_dir) / f"sparsity_{sparsity:.2f}")
    _print_table(sparsity_dir)

    logger.info("Pipeline complete. Outputs: %s", sparsity_dir)


if __name__ == "__main__":
    main()
