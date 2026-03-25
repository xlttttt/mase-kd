from pathlib import Path

import experiments.scripts.run_gpt2_imdb_grid_search as gs


def test_gpt2_imdb_grid_search_smoke(tmp_path, monkeypatch):
    config_path = tmp_path / "distilgpt2_imdb_grid_search.yaml"
    config_path.write_text(
        """
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

grid:
  alpha: [0.5]
  temperature: [2.0]

output:
  dir: PLACEHOLDER
""".strip()
    )

    out_dir = tmp_path / "outputs"
    text = config_path.read_text().replace("PLACEHOLDER", str(out_dir))
    config_path.write_text(text)

    monkeypatch.setattr(gs, "CONFIG_PATH", str(config_path))

    gs.main()

    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "best.json").exists()

    run_dir = out_dir / "alpha_0.5_temp_2.0"
    assert run_dir.exists()
