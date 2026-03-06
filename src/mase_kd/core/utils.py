"""Utility helpers for reproducibility and metrics I/O."""

import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_metrics_json(metrics: Mapping[str, object], output_path: str | Path) -> None:
    """Persist metrics to a JSON file, creating parent directories when needed."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fp:
        json.dump(dict(metrics), fp, indent=2, sort_keys=True)
