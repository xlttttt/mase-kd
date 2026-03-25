import json
from pathlib import Path

import experiments.scripts.run_gpt2_imdb_kd_best as best


def test_gpt2_imdb_kd_best_smoke(tmp_path, monkeypatch):
    config_path = tmp_path / "distilgpt2_imdb_kd_best.yaml"
    output_dir = tmp_path / "outputs"

    config_path.write_text(
        f"""
teacher:
  model_name: mnoukhov/gpt2-imdb-sentiment-classifier

student:
  model_name: distilgpt2
  num_labels: 2

training:
  batch_size: 2
  learning_rate: 0.00002
  num_epochs: 1
  warmup_ratio: 0.0
  weight_decay: 0.0
  seed: 42
  max_seq_length: 128
  train_subset: 32
  test_subset: 32

kd:
  alpha: 0.5
  temperature: 2.0

output:
  dir: {output_dir}
""".strip()
    )

    monkeypatch.setattr(best, "CONFIG_PATH", str(config_path))
    best.main()

    metrics_path = output_dir / "metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["alpha"] == 0.5
    assert metrics["temperature"] == 2.0
    assert "val_accuracy" in metrics
    assert "val_f1" in metrics
    assert "val_loss" in metrics
