#!/usr/bin/env python3
"""Train BERT student with knowledge distillation from the bert_kd.toml config.

Usage:
    python experiments/scripts/run_bert_kd.py
    python experiments/scripts/run_bert_kd.py --config experiments/configs/bert_kd.toml
    python experiments/scripts/run_bert_kd.py --alpha 0.7 --temperature 6.0
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

import toml


def parse_args():
    p = argparse.ArgumentParser(description="BERT KD training")
    p.add_argument("--config", default="experiments/configs/bert_kd.toml")
    p.add_argument("--alpha", type=float, default=None, help="Override kd.alpha")
    p.add_argument("--temperature", type=float, default=None, help="Override kd.temperature")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dict = toml.load(args.config)

    # Apply CLI overrides
    if args.alpha is not None:
        cfg_dict["kd"]["alpha"] = args.alpha
    if args.temperature is not None:
        cfg_dict["kd"]["temperature"] = args.temperature
    if args.epochs is not None:
        cfg_dict["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg_dict["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_dict["training"]["learning_rate"] = args.lr
    if args.output_dir is not None:
        cfg_dict["output"]["dir"] = args.output_dir

    from mase_kd.nlp.bert_kd import (
        BertKDConfig,
        BertStudentConfig,
        build_bert_kd_trainer,
    )
    from mase_kd.core.losses import DistillationLossConfig

    config = BertKDConfig(
        teacher_model_name=cfg_dict["teacher"]["model_name"],
        student=BertStudentConfig(
            num_hidden_layers=cfg_dict["student"]["num_hidden_layers"],
            hidden_size=cfg_dict["student"]["hidden_size"],
            num_attention_heads=cfg_dict["student"]["num_attention_heads"],
            intermediate_size=cfg_dict["student"]["intermediate_size"],
            num_labels=cfg_dict["student"]["num_labels"],
            pretrained_name=cfg_dict["student"].get("pretrained_name"),
        ),
        kd=DistillationLossConfig(
            alpha=cfg_dict["kd"]["alpha"],
            temperature=cfg_dict["kd"]["temperature"],
        ),
        max_seq_length=cfg_dict["training"]["max_seq_length"],
        batch_size=cfg_dict["training"]["batch_size"],
        learning_rate=cfg_dict["training"]["learning_rate"],
        num_epochs=cfg_dict["training"]["num_epochs"],
        warmup_ratio=cfg_dict["training"]["warmup_ratio"],
        weight_decay=cfg_dict["training"]["weight_decay"],
        seed=cfg_dict["training"]["seed"],
        output_dir=cfg_dict["output"]["dir"],
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = build_bert_kd_trainer(config, device=device)
    history = trainer.train()

    best = max(history, key=lambda x: x["val_accuracy"])
    print("\n=== BERT KD Results ===")
    print(f"KD alpha:       {config.kd.alpha}")
    print(f"Temperature:    {config.kd.temperature}")
    print(f"Best epoch:     {best['epoch']}")
    print(f"Accuracy:       {best['val_accuracy']:.4f}")
    print(f"F1 (macro):     {best['val_f1']:.4f}")
    print(f"Outputs:        {config.output_dir}")


if __name__ == "__main__":
    main()
