import logging
import torch

from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.gpt2_imdb_kd import (
    GPT2IMDbKDConfig,
    GPT2StudentConfig,
    build_gpt2_imdb_kd_trainer,
)

logging.basicConfig(level=logging.INFO)

cfg = GPT2IMDbKDConfig(
    teacher_model_name="mnoukhov/gpt2-imdb-sentiment-classifier",
    student=GPT2StudentConfig(
        model_name="distilgpt2",
        num_labels=2,
    ),
    kd=DistillationLossConfig(
        alpha=0.5,
        temperature=4.0,
    ),
    max_seq_length=256,
    batch_size=4,
    learning_rate=0.00002,
    num_epochs=1,
    train_subset=1024,
    test_subset=1000,
    output_dir="outputs/gpt2_imdb_kd_smoke",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = build_gpt2_imdb_kd_trainer(cfg, device=device)
trainer.train()

metrics = trainer.evaluate()
print(metrics)
