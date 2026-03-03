import json

from mase_kd.core.utils import dump_metrics_json


def test_dump_metrics_json(tmp_path) -> None:
    output_file = tmp_path / "metrics" / "run_001.json"
    dump_metrics_json({"map50": 0.42, "latency_ms": 11.3}, output_file)

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["map50"] == 0.42
