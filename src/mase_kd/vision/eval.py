"""Vision evaluation helpers for efficiency metrics."""

import time
from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Return total parameter count for a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def benchmark_forward_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup_steps: int = 3,
    measure_steps: int = 20,
) -> dict[str, Any]:
    """Measure average forward-pass latency in milliseconds."""
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(sample_input)

        start = time.perf_counter()
        for _ in range(measure_steps):
            _ = model(sample_input)
        end = time.perf_counter()

    total = end - start
    avg_ms = (total / measure_steps) * 1000
    return {"avg_latency_ms": avg_ms, "steps": measure_steps}
