import argparse
import csv
import json
import logging
from itertools import product
from pathlib import Path

import torch
import yaml

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.gpt2_imdb_kd import (
    GPT2IMDbKDConfig,
    GPT2StudentConfig,
    build_gpt2_imdb_kd_trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "experiments/configs/distilgpt2_imdb_grid_search.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_cfg(raw: dict, alpha: float, temperature: float) -> GPT2IMDbKDConfig:
    run_dir = Path(raw["output"]["dir"]) / f"alpha_{alpha:.1f}_temp_{temperature:.1f}"

    return GPT2IMDbKDConfig(
        teacher_model_name=raw["teacher"]["model_name"],
        student=GPT2StudentConfig(
            model_name=raw["student"]["model_name"],
            num_labels=raw["student"].get("num_labels", 2),
        ),
        kd=DistillationLossConfig(
            alpha=alpha,
            temperature=temperature,
        ),
        max_seq_length=raw["training"].get("max_seq_length", 256),
        batch_size=raw["training"].get("batch_size", 8),
        learning_rate=raw["training"].get("learning_rate", 2e-5),
        num_epochs=raw["training"].get("num_epochs", 2),
        warmup_ratio=raw["training"].get("warmup_ratio", 0.06),
        weight_decay=raw["training"].get("weight_decay", 0.01),
        seed=raw["training"].get("seed", 42),
        train_subset=raw["training"].get("train_subset"),
        test_subset=raw["training"].get("test_subset"),
        output_dir=str(run_dir),
    )


def run_grid_search(config_path: str):
    raw = load_yaml(config_path)

    alphas = raw["grid"]["alpha"]
    temperatures = raw["grid"]["temperature"]

    sweep_dir = Path(raw["output"]["dir"])
    sweep_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for alpha, temperature in product(alphas, temperatures):
        logger.info("Running alpha=%.1f temperature=%.1f", alpha, temperature)

        cfg = make_cfg(raw, alpha, temperature)
        trainer = build_gpt2_imdb_kd_trainer(cfg, device=device)
        trainer.train()
        metrics = trainer.evaluate()

        row = {
            "alpha": alpha,
            "temperature": temperature,
            "val_loss": float(metrics["val_loss"]),
            "val_accuracy": float(metrics["val_accuracy"]),
            "val_f1": float(metrics["val_f1"]),
            "output_dir": cfg.output_dir,
        }
        results.append(row)

        with open(sweep_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(sweep_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "alpha",
                    "temperature",
                    "val_loss",
                    "val_accuracy",
                    "val_f1",
                    "output_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

    best = max(results, key=lambda x: x["val_accuracy"])
    with open(sweep_dir / "best.json", "w") as f:
        json.dump(best, f, indent=2)

    print("BEST =", best)
    return results


def main():
    args = parse_args()
    run_grid_search(args.config)


if __name__ == "__main__":
    main()
