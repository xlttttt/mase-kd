"""CLI entry point for BERT knowledge distillation experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BERT sequence-classification KD experiment"
    )

    # Teacher / student
    p.add_argument(
        "--teacher",
        default="textattack/bert-base-uncased-SST-2",
        help="HuggingFace model name for the teacher (must be already fine-tuned on SST-2)",
    )
    p.add_argument(
        "--student-layers", type=int, default=4, help="Number of hidden layers in student"
    )
    p.add_argument(
        "--student-hidden", type=int, default=256, help="Hidden size of student"
    )
    p.add_argument(
        "--student-heads", type=int, default=4, help="Attention heads in student"
    )
    p.add_argument(
        "--student-intermediate",
        type=int,
        default=1024,
        help="Intermediate (FFN) size of student",
    )
    p.add_argument(
        "--student-pretrained",
        default=None,
        help="Optional: init student from this HuggingFace checkpoint",
    )

    # KD hyperparameters
    p.add_argument("--alpha", type=float, default=0.5, help="KD mixing weight (0=hard only, 1=soft only)")
    p.add_argument("--temperature", type=float, default=2.0, help="Softmax temperature for KD")

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument("--output-dir", default="outputs/bert_kd", help="Where to save checkpoints and history")
    p.add_argument("--baseline", action="store_true", help="Train without KD (alpha=0, pure hard-label CE)")
    p.add_argument("--eval-only", default=None, metavar="CKPT", help="Path to a saved student checkpoint — evaluate only, no training")

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    import torch
    from mase_kd.nlp.bert_kd import (
        BertKDConfig,
        BertStudentConfig,
        build_bert_kd_trainer,
    )
    from mase_kd.core.losses import DistillationLossConfig

    # Override alpha to 0 in baseline mode (pure hard-label CE)
    alpha = 0.0 if args.baseline else args.alpha

    config = BertKDConfig(
        teacher_model_name=args.teacher,
        student=BertStudentConfig(
            num_hidden_layers=args.student_layers,
            hidden_size=args.student_hidden,
            num_attention_heads=args.student_heads,
            intermediate_size=args.student_intermediate,
            pretrained_name=args.student_pretrained,
        ),
        kd=DistillationLossConfig(alpha=alpha, temperature=args.temperature),
        max_seq_length=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        _eval_only(args.eval_only, config, device)
        return

    logger.info(
        "Starting %s experiment | alpha=%.2f | T=%.1f",
        "BASELINE" if args.baseline else "KD",
        alpha,
        args.temperature,
    )

    trainer = build_bert_kd_trainer(config, device=device)
    history = trainer.train()

    # Print final summary
    best = max(history, key=lambda x: x["val_accuracy"])
    logger.info("=== Final Results ===")
    logger.info("Best epoch: %d", best["epoch"])
    logger.info("Best val accuracy: %.4f", best["val_accuracy"])
    logger.info("Best val F1:       %.4f", best["val_f1"])


def _eval_only(ckpt_path: str, config, device) -> None:
    """Load a saved student checkpoint and run evaluation."""
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    from mase_kd.nlp.bert_kd import load_sst2_dataloaders
    from mase_kd.nlp.eval import evaluate_classification, efficiency_report

    logger.info("Evaluation-only mode: loading checkpoint from %s", ckpt_path)
    student = BertForSequenceClassification.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    _, val_loader = load_sst2_dataloaders(tokenizer, config.max_seq_length, config.batch_size)

    metrics = evaluate_classification(student, val_loader, device)
    logger.info("Accuracy: %.4f | F1: %.4f", metrics["accuracy"], metrics["f1"])

    # Efficiency
    sample = next(iter(val_loader))
    eff = efficiency_report(
        student,
        sample["input_ids"][:1].to(device),
        sample["attention_mask"][:1].to(device),
    )
    logger.info("Parameters: %d | Latency: %.2f ms", eff["num_parameters"], eff["avg_latency_ms"])

    out = {**metrics, **eff}
    out_path = Path(config.output_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
