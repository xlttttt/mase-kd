"""IMDb logits-only KD trainer using GPT-2 style sequence classification."""

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
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from mase_kd.core.losses import DistillationLossConfig, compute_distillation_loss

logger = logging.getLogger(__name__)


@dataclass
class GPT2StudentConfig:
    model_name: str = "distilgpt2"
    num_labels: int = 2


@dataclass
class GPT2IMDbKDConfig:
    teacher_model_name: str = "mnoukhov/gpt2-imdb-sentiment-classifier"
    student: GPT2StudentConfig = field(default_factory=GPT2StudentConfig)
    kd: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    max_seq_length: int = 256
    batch_size: int = 8
    learning_rate: float = 0.00002
    num_epochs: int = 1
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 42

    train_subset: Optional[int] = 512
    test_subset: Optional[int] = 256

    output_dir: str = "outputs/gpt2_imdb_kd_smoke"
    student_weights_path: Optional[str] = None

    def validate(self) -> None:
        self.kd.validate()
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_gpt2_student(cfg: GPT2StudentConfig):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )


def load_imdb_dataloaders(
    tokenizer,
    max_length: int,
    batch_size: int,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    seed: int = 42,
):
    from datasets import load_dataset

    raw = load_dataset("imdb")
    train_ds = raw["train"]
    test_ds = raw["test"]

    if train_subset is not None:
        train_ds = train_ds.shuffle(seed=seed).select(
            range(min(train_subset, len(train_ds)))
        )
    if test_subset is not None:
        test_ds = test_ds.shuffle(seed=seed).select(
            range(min(test_subset, len(test_ds)))
        )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    if "text" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["text"])
    if "text" in test_ds.column_names:
        test_ds = test_ds.remove_columns(["text"])

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep_cols)
    test_ds.set_format(type="torch", columns=keep_cols)

    def collate_fn(features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


class GPT2IMDbKDTrainer:
    def __init__(
        self,
        config: GPT2IMDbKDConfig,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
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
        self.test_loader = test_loader

        self.zero_masks: dict[str, torch.Tensor] = {}
        for name, param in self.student.named_parameters():
            if "weight" in name:
                self.zero_masks[name] = (param.data == 0)

        no_decay = {"bias", "LayerNorm.weight", "ln_f.weight", "ln_1.weight", "ln_2.weight"}
        grouped_params = [
            {
                "params": [
                    p for n, p in self.student.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.student.named_parameters()
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

    def _enforce_zero_mask_on_grads(self) -> None:
        for name, param in self.student.named_parameters():
            if param.grad is None:
                continue
            mask = self.zero_masks.get(name)
            if mask is not None and mask.any():
                param.grad.data[mask] = 0

    def _enforce_zero_mask_on_weights(self) -> None:
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                mask = self.zero_masks.get(name)
                if mask is not None and mask.any():
                    param.data[mask] = 0

    def _train_epoch(self):
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
            self._enforce_zero_mask_on_grads()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self._enforce_zero_mask_on_weights()
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
    def evaluate(self):
        from evaluate import load as load_metric

        self.student.eval()
        metric_acc = load_metric("accuracy")
        metric_f1 = load_metric("f1")

        val_loss = 0.0
        n_batches = 0

        for batch in self.test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            preds = out.logits.argmax(dim=-1)
            metric_acc.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
            metric_f1.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
            val_loss += out.loss.item()
            n_batches += 1

        acc = metric_acc.compute()["accuracy"]
        f1 = metric_f1.compute(average="macro")["f1"]

        return {
            "val_loss": val_loss / max(n_batches, 1),
            "val_accuracy": acc,
            "val_f1": f1,
        }

    def train(self):
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
                self.student.save_pretrained(out_dir / "best_student")
                logger.info("=> New best: %.4f — checkpoint saved.", best_acc)

        with (out_dir / "training_history.json").open("w") as fp:
            json.dump(self.history, fp, indent=2)

        logger.info("Training complete. Best val accuracy: %.4f", best_acc)
        return self.history


def build_gpt2_imdb_kd_trainer(
    config: GPT2IMDbKDConfig,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(config.seed)
    logger.info("Device: %s", device)
    logger.info("Loading teacher: %s", config.teacher_model_name)

    teacher = AutoModelForSequenceClassification.from_pretrained(config.teacher_model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.student.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.student_weights_path:
        logger.info("Loading student weights from: %s", config.student_weights_path)
        student = AutoModelForSequenceClassification.from_pretrained(
            config.student_weights_path,
            num_labels=config.student.num_labels,
        )
    else:
        student = build_gpt2_student(config.student)

    student.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Loading dataset: imdb")
    train_loader, test_loader = load_imdb_dataloaders(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        batch_size=config.batch_size,
        train_subset=config.train_subset,
        test_subset=config.test_subset,
        seed=config.seed,
    )

    return GPT2IMDbKDTrainer(
        config=config,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )
