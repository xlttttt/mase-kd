import logging

import torch
import yaml

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.gpt2_imdb_kd import (
    GPT2IMDbKDConfig,
    GPT2StudentConfig,
    build_gpt2_imdb_kd_trainer,
)

logging.basicConfig(level=logging.INFO)

with open("experiments/configs/distilgpt2_imdb_kd_full.yaml", "r") as f:
    raw = yaml.safe_load(f)

cfg = GPT2IMDbKDConfig(
    teacher_model_name=raw["teacher"]["model_name"],
    student=GPT2StudentConfig(
        model_name=raw["student"]["model_name"],
        num_labels=raw["student"].get("num_labels", 2),
    ),
    kd=DistillationLossConfig(
        alpha=raw["kd"].get("alpha", 0.5),
        temperature=raw["kd"].get("temperature", 4.0),
    ),
    max_seq_length=raw["training"].get("max_seq_length", 256),
    batch_size=raw["training"].get("batch_size", 8),
    learning_rate=raw["training"].get("learning_rate", 0.00002),
    num_epochs=raw["training"].get("num_epochs", 2),
    warmup_ratio=raw["training"].get("warmup_ratio", 0.06),
    weight_decay=raw["training"].get("weight_decay", 0.01),
    seed=raw["training"].get("seed", 42),
    train_subset=raw["training"].get("train_subset", None),
    test_subset=raw["training"].get("test_subset", None),
    output_dir=raw["output"]["dir"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = build_gpt2_imdb_kd_trainer(cfg, device=device)
trainer.train()

metrics = trainer.evaluate()
print(metrics)
