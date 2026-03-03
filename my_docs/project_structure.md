# Project Folder Structure

## Placement rules

- Original MASE source code remains in `./src/chop` (no relocation).
- All **new implementation source code** for this project goes in `./src/mase_kd`.
- All **test code** for coursework goes in `./cw`.

## Proposed structure

```text
.
├── src/
│   ├── chop/                         # Existing MASE codebase (original)
│   └── mase_kd/                      # New coursework source code only
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py             # Experiment/KD config schema
│       ├── core/
│       │   ├── __init__.py
│       │   ├── losses.py             # KD losses (hard + soft)
│       │   └── utils.py              # Shared helpers (seed, logging, io)
│       ├── vision/
│       │   ├── __init__.py
│       │   ├── yolo_kd.py            # YOLO teacher-student KD pipeline
│       │   └── eval.py               # mAP + efficiency evaluation
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── bert_kd.py            # BERT teacher-student KD pipeline
│       │   └── eval.py               # Accuracy/F1 + efficiency evaluation
│       ├── runners/
│       │   ├── __init__.py
│       │   ├── run_vision.py         # Entry point for YOLO runs
│       │   └── run_nlp.py            # Entry point for BERT runs
│       └── reporting/
│           ├── __init__.py
│           └── summarize.py          # Aggregate metrics/tables
├── cw/                               # Coursework tests only
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_kd_losses.py
│   │   └── test_config_schema.py
│   ├── integration/
│   │   ├── test_vision_smoke.py
│   │   └── test_nlp_smoke.py
│   └── regression/
│       └── test_metrics_artifacts.py
└── my_docs/
	├── project_plan.md
	└── project_structure.md
```

## Notes

- Keep imports one-way: `mase_kd` may call into `chop`, but avoid modifying `chop` unless strictly necessary.
- Keep experiment outputs (logs/checkpoints/metrics) outside source directories (e.g., a runtime `outputs/` path).
