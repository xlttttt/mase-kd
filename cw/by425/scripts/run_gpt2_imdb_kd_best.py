import json
from pathlib import Path

import torch
import yaml

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.gpt2_imdb_kd import (
    GPT2IMDbKDConfig,
    GPT2StudentConfig,
    build_gpt2_imdb_kd_trainer,
)

CONFIG_PATH = "experiments/configs/distilgpt2_imdb_kd_best.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    cfg = GPT2IMDbKDConfig(
        teacher_model_name=raw["teacher"]["model_name"],
        student=GPT2StudentConfig(
            model_name=raw["student"]["model_name"],
            num_labels=raw["student"].get("num_labels", 2),
        ),
        kd=DistillationLossConfig(
            alpha=raw["kd"]["alpha"],
            temperature=raw["kd"]["temperature"],
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
        output_dir=raw["output"]["dir"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = build_gpt2_imdb_kd_trainer(cfg, device=device)
    trainer.train()
    metrics = trainer.evaluate()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "alpha": float(cfg.kd.alpha),
        "temperature": float(cfg.kd.temperature),
        "val_loss": float(metrics["val_loss"]),
        "val_accuracy": float(metrics["val_accuracy"]),
        "val_f1": float(metrics["val_f1"]),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
