"""BERT knowledge distillation trainer for sequence classification."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from mase_kd.core.losses import DistillationLossConfig, compute_distillation_loss

logger = logging.getLogger(__name__)


@dataclass
class BertStudentConfig:
    """Small-BERT architecture for sequence classification."""

    num_hidden_layers: int = 4
    hidden_size: int = 256
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    num_labels: int = 2
    pretrained_name: Optional[str] = None


@dataclass
class BertKDConfig:
    """All hyperparameters for one BERT KD experiment."""

    teacher_model_name: str = "textattack/bert-base-uncased-SST-2"
    student: BertStudentConfig = field(default_factory=BertStudentConfig)
    kd: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    max_seq_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 42

    output_dir: str = "outputs/bert_kd"

    # NEW: load an already-trained / already-pruned student checkpoint
    student_weights_path: Optional[str] = None

    def validate(self) -> None:
        self.kd.validate()
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_student_model(cfg: BertStudentConfig) -> BertForSequenceClassification:
    """Construct a small BERT from scratch or load from a HuggingFace checkpoint."""
    if cfg.pretrained_name:
        logger.info("Loading student from pretrained checkpoint: %s", cfg.pretrained_name)
        return BertForSequenceClassification.from_pretrained(
            cfg.pretrained_name,
            num_labels=cfg.num_labels,
        )

    bert_cfg = BertConfig(
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        num_labels=cfg.num_labels,
    )
    logger.info(
        "Building student from scratch: layers=%d, hidden=%d, heads=%d",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
    )
    return BertForSequenceClassification(bert_cfg)


def load_sst2_dataloaders(
    tokenizer,
    max_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Load and tokenise the GLUE SST-2 dataset; return train/validation loaders."""
    from datasets import load_dataset

    raw = load_dataset("glue", "sst2")

    def tokenise(examples):
        return tokenizer(
            examples["sentence"],
            max_length=max_length,
            truncation=True,
        )

    tokenised = raw.map(tokenise, batched=True)
    tokenised = tokenised.rename_column("label", "labels")
    tokenised.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenised["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        tokenised["validation"],
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


class BertKDTrainer:
    """Knowledge distillation trainer for BERT sequence classification."""

    def __init__(
        self,
        config: BertKDConfig,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        config.validate()
        self.config = config
        self.device = device

        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        no_decay = {"bias", "LayerNorm.weight"}
        grouped_params = [
            {
                "params": [
                    p for n, p in student.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(grouped_params, lr=config.learning_rate)

        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.history: list[dict] = []

    def _train_epoch(self) -> dict[str, float]:
        self.student.train()
        total_loss = 0.0
        hard_loss_sum = 0.0
        soft_loss_sum = 0.0
        n_batches = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }

            student_out = self.student(**model_inputs)

            with torch.no_grad():
                teacher_out = self.teacher(**model_inputs)

            loss, hard, soft = compute_distillation_loss(
                student_logits=student_out.logits,
                teacher_logits=teacher_out.logits,
                targets=batch["labels"],
                config=self.config.kd,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            hard_loss_sum += hard.item()
            soft_loss_sum += soft.item()
            n_batches += 1

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_hard_loss": hard_loss_sum / max(n_batches, 1),
            "train_soft_loss": soft_loss_sum / max(n_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        from evaluate import load as load_metric

        self.student.eval()
        metric_acc = load_metric("accuracy")
        metric_f1 = load_metric("f1")

        val_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            out = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            preds = out.logits.argmax(dim=-1)
            metric_acc.add_batch(
                predictions=preds.cpu(),
                references=batch["labels"].cpu(),
            )
            metric_f1.add_batch(
                predictions=preds.cpu(),
                references=batch["labels"].cpu(),
            )

            val_loss += out.loss.item()
            n_batches += 1

        acc = metric_acc.compute()["accuracy"]
        f1 = metric_f1.compute(average="macro")["f1"]

        return {
            "val_loss": val_loss / max(n_batches, 1),
            "val_accuracy": acc,
            "val_f1": f1,
        }

    def train(self) -> list[dict]:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        best_acc = 0.0

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._train_epoch()
            val_metrics = self.evaluate()

            epoch_log = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(epoch_log)

            logger.info(
                "Epoch %d/%d | loss=%.4f | val_acc=%.4f | val_f1=%.4f",
                epoch,
                self.config.num_epochs,
                train_metrics["train_loss"],
                val_metrics["val_accuracy"],
                val_metrics["val_f1"],
            )

            if val_metrics["val_accuracy"] > best_acc:
                best_acc = val_metrics["val_accuracy"]
                self.save_student(out_dir / "best_student")
                logger.info("=> New best: %.4f — checkpoint saved.", best_acc)

        with (out_dir / "training_history.json").open("w") as fp:
            json.dump(self.history, fp, indent=2)

        logger.info("Training complete. Best val accuracy: %.4f", best_acc)
        return self.history

    def save_student(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(path)

    @classmethod
    def load_student(
        cls,
        path: str | Path,
        num_labels: int = 2,
    ) -> BertForSequenceClassification:
        return BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)


def build_bert_kd_trainer(
    config: BertKDConfig,
    device: Optional[torch.device] = None,
) -> BertKDTrainer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(config.seed)
    logger.info("Device: %s", device)
    logger.info("Loading teacher: %s", config.teacher_model_name)

    teacher = BertForSequenceClassification.from_pretrained(config.teacher_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)

    if config.student_weights_path:
        logger.info("Loading student weights from: %s", config.student_weights_path)
        student = BertForSequenceClassification.from_pretrained(
            config.student_weights_path,
            num_labels=config.student.num_labels,
        )
    else:
        student = build_student_model(config.student)

    logger.info("Loading dataset: glue/sst2")
    train_loader, val_loader = load_sst2_dataloaders(
        tokenizer,
        config.max_seq_length,
        config.batch_size,
    )

    return BertKDTrainer(
        config=config,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
