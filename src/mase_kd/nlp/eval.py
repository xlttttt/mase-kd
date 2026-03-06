"""NLP evaluation utilities: accuracy, F1, latency, and model size."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_all(model: nn.Module) -> int:
    """Return total parameter count (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int = 2,
) -> dict[str, float]:
    """Compute accuracy and macro-F1 on a classification dataloader.

    Args:
        model: A HuggingFace BertForSequenceClassification (or compatible).
        dataloader: Yields batches with keys 'input_ids', 'attention_mask', 'labels'.
        device: Target device.
        num_labels: Number of output classes.

    Returns:
        Dict with 'accuracy', 'f1', 'val_loss'.
    """
    from evaluate import load as load_metric

    model.eval()
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        preds = out.logits.argmax(dim=-1)
        metric_acc.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
        metric_f1.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
        total_loss += out.loss.item()
        n_batches += 1

    return {
        "accuracy": metric_acc.compute()["accuracy"],
        "f1": metric_f1.compute(average="macro")["f1"],
        "val_loss": total_loss / max(n_batches, 1),
    }


def benchmark_inference_latency(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    sample_attention_mask: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 50,
) -> dict[str, Any]:
    """Measure average forward-pass latency in milliseconds.

    Args:
        model: Any nn.Module with signature forward(input_ids, attention_mask).
        sample_input_ids: Shape [1, seq_len].
        sample_attention_mask: Shape [1, seq_len].
        warmup_steps: Steps to discard before timing.
        measure_steps: Steps to average over.

    Returns:
        Dict with 'avg_latency_ms' and 'steps'.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = sample_input_ids.to(device)
    attention_mask = sample_attention_mask.to(device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            model(input_ids=input_ids, attention_mask=attention_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(measure_steps):
            model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / measure_steps) * 1000.0
    return {"avg_latency_ms": avg_ms, "steps": measure_steps}


def efficiency_report(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    sample_attention_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return a consolidated efficiency snapshot for a model.

    Includes parameter count and inference latency.
    """
    latency = benchmark_inference_latency(model, sample_input_ids, sample_attention_mask)
    return {
        "num_parameters": count_parameters_all(model),
        "trainable_parameters": count_parameters(model),
        **latency,
    }
