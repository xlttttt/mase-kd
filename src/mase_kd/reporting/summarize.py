"""Utilities for aggregating JSON metric artifacts."""

import json
from pathlib import Path


def summarize_metric_files(metrics_dir: str | Path) -> dict[str, float]:
    """Collect scalar values from JSON metric artifacts into a flat summary."""
    metrics_path = Path(metrics_dir)
    summary: dict[str, float] = {}

    for file_path in sorted(metrics_path.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        for key, value in data.items():
            if isinstance(value, (int, float)):
                summary[f"{file_path.stem}.{key}"] = float(value)

    return summary
