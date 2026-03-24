# ResNet18 A-E KD Pipeline — Usage Guide (kx725)

## Overview

This document explains every file in the kx725 workspace and the supporting `src/mase_kd/` library, then shows exactly how to run smoke tests and full experiments.

The pipeline trains ResNet18 on CIFAR-10 or CIFAR-100 under five experimental conditions (A–E) to measure the effect of pruning and knowledge distillation on accuracy and model size.

---

## File Map

### `cw/kx725/` — Test scripts, configs, scripts, and outputs

```
cw/kx725/
├── .gitignore                        ← excludes outputs/ from version control
├── configs/                          ← experiment YAML configs (canonical location)
│   ├── resnet18_cifar10_smoke.yaml
│   ├── resnet18_cifar10_full.yaml
│   ├── resnet18_cifar100_smoke.yaml
│   └── resnet18_cifar100_full.yaml
├── scripts/
│   └── aggregate_results.py          ← post-processing: combine multi-sparsity results
├── unit/                             ← pytest unit tests (fast, no GPU)
│   ├── test_passes.py                ← 56 tests for PrunePass + ExportMetricsPass
│   ├── test_kd_losses.py             ← 17 tests for core KD loss functions
│   ├── test_config_schema.py         ← 21 tests for ResNetKDConfig + VisionKDConfig
│   └── test_metrics_artifacts.py     ← 1 regression test for metrics JSON I/O
├── integration/                      ← pytest integration tests (require CPU/GPU)
│   └── test_resnet_smoke.py          ← 12 end-to-end tests with synthetic CIFAR data
└── outputs/                          ← generated at runtime (gitignored)
    └── resnet18/
        ├── cifar10/sparsity_<s>/     ← one dir per sparsity level
        └── cifar100/sparsity_<s>/
```

#### `configs/resnet18_cifar10_smoke.yaml`
Fast validation config for CIFAR-10. Uses a 5 000-sample subset, 2 dense epochs, and 1 epoch each for fine-tune/KD steps. Completes in ~1–3 min on CPU. Intended to confirm the full A–E pipeline runs without errors, not to produce publishable numbers.

#### `configs/resnet18_cifar10_full.yaml`
Full training config for CIFAR-10. 100 dense epochs, 40 fine-tune/KD epochs, full 50 000-sample dataset, 3 random seeds. Produces the accuracy numbers reported in the project.

#### `configs/resnet18_cifar100_smoke.yaml`
Same structure as the CIFAR-10 smoke config but targets CIFAR-100 (100 classes). Uses a 5 000-sample subset and 3 dense epochs.

#### `configs/resnet18_cifar100_full.yaml`
Full training config for CIFAR-100. 120 dense epochs (longer due to harder task), 40 fine-tune/KD epochs.

#### `scripts/aggregate_results.py`
Post-processing script that reads per-sparsity `comparison_table.json` files from `cw/kx725/outputs/` and produces:
- `comparison_table_sparsity_<s>.md` — per-sparsity variant table
- `combined_table.{md,json}` — all sparsities × all variants in one table
- `accuracy_vs_variant.png` and `accuracy_vs_sparsity.png` — matplotlib charts

Usage:
```bash
PYTHONPATH=src python3 cw/kx725/scripts/aggregate_results.py \
    --model resnet18 --dataset cifar10 --sparsities 0.5 0.7 0.85
```

#### `unit/test_passes.py`
Unit tests for the pruning and export infrastructure:
- `TestPrunePass` — verifies sparsity targets are met, `make_permanent` zeros out masks, parameter counts are correct, and edge cases (s=0.01, s=0.99) are handled
- `TestSparsityHelpers` — tests `count_nonzero_params()` utility
- `TestExportMetricsPass` — verifies JSON/MD/PNG artifacts are written, delta columns are computed correctly, and multi-variant aggregation works
- `TestPipelineConfigLoads` — smoke-loads `resnet18_cifar10_{smoke,full}.yaml` and asserts key fields are present

#### `unit/test_kd_losses.py`
Unit tests for `mase_kd.core.losses`:
- `TestSoftLogitKLLoss` — scalar output, non-negativity, zero when identical, gradient flows
- `TestHardLabelCELoss` — scalar output, near-zero loss on perfect predictions
- `TestComputeDistillationLoss` — alpha=0 → pure hard label, alpha=1 → pure soft, alpha=0.4 → correct linear combination
- `TestDistillationLossConfig` — validate() rejects bad alpha/temperature values

#### `unit/test_config_schema.py`
Unit tests for config dataclasses:
- `TestVisionKDConfig` — generic vision config (alpha, temperature, lr bounds)
- `TestResNetKDConfig` — ResNet-specific config (all fields, boundary conditions, custom teacher_arch)

#### `unit/test_metrics_artifacts.py`
Single regression test: calls `dump_metrics_json()` and verifies the file is written and parseable.

#### `integration/test_resnet_smoke.py`
End-to-end integration tests using a synthetic 32×32 dataset (no real CIFAR download needed):
- `TestResNetKDTrainerOneEpoch` — train 1 epoch with alpha=0.5, assert history populated, loss > 0, accuracy in [0,1], checkpoint saved
- `TestTeacherFrozen` — assert teacher weights unchanged after training step
- `TestBaselineMode` — alpha=0 mode works with and without providing teacher
- `TestCheckpointSaveLoad` — saved/loaded student produces identical eval accuracy
- `TestPruneThenTrain` — prune then fine-tune (simulates C-step): sparsity achieved, training converges

---

### `my_docs/kx725/` — Documentation

#### `experiment_log.md`
Scientific record of experiment sessions:
- Session 2026-03-06: CIFAR-10 catastrophic forgetting fix (LR 0.01→0.001 for fine-tune/KD), CIFAR-100 extension, full A–E results at sparsity 0.5/0.7/0.85
- Session 2026-03-08: BERT experiment results (logits KD vs hidden KD) and pipeline fixes — recorded here because kx725 implemented those pipeline fixes

#### `resnet_usage.md` *(this file)*
File-by-file explanation and run commands.

---

### `src/mase_kd/` — Shared source library

The source lives here and is shared across all team members' work. kx725's ResNet contribution spans `vision/`, `passes/`, `runners/`, `core/`, and `config/`.

```
src/mase_kd/
├── __init__.py                    ← package root; exports DistillationLossConfig
├── core/
│   ├── losses.py                  ← foundational KD loss functions (used by ResNet + NLP + Vision)
│   └── utils.py                   ← set_seed(), dump_metrics_json()
├── config/
│   └── schema.py                  ← ResNetKDConfig, VisionKDConfig dataclasses
├── vision/
│   ├── resnet_kd.py               ← ResNetKDTrainer + data loaders (main ResNet implementation)
│   └── eval.py                    ← count_parameters(), benchmark_forward_latency() (generic)
├── passes/
│   ├── pipeline.py                ← ResNetPipeline: orchestrates steps A→E
│   ├── prune_pass.py              ← PrunePass: global L1 unstructured pruning
│   └── export_pass.py             ← ExportMetricsPass: writes comparison_table.{md,json} + plot
├── runners/
│   └── run_pipeline.py            ← CLI entry point: python -m mase_kd.runners.run_pipeline
├── reporting/
│   └── summarize.py               ← summarize_metric_files(): aggregates scalar JSON metrics
├── distillation/
│   └── __init__.py                ← reserved for future distillation modules
└── nlp/
    └── __init__.py                ← reserved for other team members (BERT)
```

#### `src/mase_kd/core/losses.py`
The foundational KD loss module used by all pipelines.

Key symbols:
- `DistillationLossConfig` — dataclass holding `alpha` (soft-loss weight) and `temperature`
- `soft_logit_kl_loss(student, teacher, temperature)` — temperature-scaled KL divergence, multiplied by T² to keep gradient magnitude consistent across temperatures
- `hard_label_ce_loss(logits, targets)` — standard cross-entropy
- `compute_distillation_loss(student, teacher, targets, config)` → `(total, hard, soft)` — combines hard and soft losses: `total = (1-alpha)*hard + alpha*soft`

#### `src/mase_kd/core/utils.py`
Infrastructure utilities:
- `set_seed(seed)` — seeds Python `random`, NumPy, and PyTorch (including CUDA) for reproducible runs
- `dump_metrics_json(metrics, path)` — writes a dict to JSON, creating parent directories automatically

#### `src/mase_kd/config/schema.py`
Config dataclasses with `validate()` methods:
- `VisionKDConfig` — generic vision KD config (alpha, temperature, learning_rate); used for any vision model
- `ResNetKDConfig` — full ResNet pipeline config: teacher/student weights, num_classes, dataset, all training hyperparameters (epochs, batch_size, lr, momentum, weight_decay, lr_schedule), data settings (data_dir, val_split, subset_size), and output_dir

#### `src/mase_kd/vision/resnet_kd.py`
The core ResNet KD implementation.

Key symbols:
- `ResNetKDConfig` — (also defined here; `schema.py` provides the canonical version)
- `build_cifar_resnet18(num_classes)` — constructs ResNet18 with a CIFAR-friendly first conv (3×3, stride 1, no maxpool). Preserves 32×32 spatial resolution; `num_classes` controls only the final FC layer
- `load_cifar10_dataloaders(data_dir, batch_size, val_split, subset_size, seed)` — returns `(train_loader, val_loader, test_loader)` for CIFAR-10
- `load_cifar100_dataloaders(...)` — same for CIFAR-100
- `ResNetKDTrainer` — the main training class:
  - `train()` — runs full training loop, saves `best_student.pth` and `training_history.json`
  - `evaluate(split)` — evaluates student on `"val"` or `"test"` split, returns `{"accuracy": float}`
  - `load_student(path)` — restores a checkpoint into the student model
  - Teacher always runs in `eval()` mode with `torch.no_grad()`; when `alpha=0` the teacher forward pass is skipped entirely

#### `src/mase_kd/passes/pipeline.py`
Orchestrates the five A–E experimental steps for ResNet18.

`ResNetPipeline.run(config, output_dir, sparsity, device)`:
| Step | Dir | What happens |
|---|---|---|
| A | `A_dense/` | Train ResNet18 from scratch (alpha=0); checkpoint becomes KD teacher |
| B | `B_pruned/` | Deep-copy A's student, apply `PrunePass`, evaluate immediately |
| C | `C_ft/` | Load B's pruned weights, fine-tune with hard labels (alpha=0) |
| D | `D_kd/` | Load B's pruned weights, distil from A's checkpoint (alpha=0.5 by default) |
| E | `E_kd_ft/` | Load D's best checkpoint, fine-tune with hard labels (alpha=0) |

Each step writes `metrics.json` to its subdirectory. After all steps, `ExportMetricsPass` writes `comparison_table.{md,json}` and `trade_off_plot.png` to the sparsity root.

#### `src/mase_kd/passes/prune_pass.py`
Global magnitude pruning.

- `PruneConfig(sparsity, target_types, make_permanent)` — configuration; `target_types` defaults to `(Conv2d, Linear)`
- `PrunePass.run(model, config, metadata)` → `(pruned_model, info_dict)` — applies `torch.nn.utils.prune.global_unstructured` with `L1Unstructured`, then calls `make_permanent()` to fuse the mask into the weight tensor (so `state_dict()` directly gives sparse weights, no mask reinstatement needed)
- `count_nonzero_params(model)` → `(nonzero, total)` — used by pipeline to report sparsity

#### `src/mase_kd/passes/export_pass.py`
Artifact generation after each pipeline run.

- `ExportMetricsPass.run(results, output_dir, model_name, primary_metric)` — given the `{A: {...}, B: {...}, ...}` dict, writes:
  - `comparison_table.md` — Markdown table with accuracy, Δ-vs-A, parameter count, sparsity for each variant
  - `comparison_table.json` — same data in machine-readable form
  - `trade_off_plot.png` — accuracy vs sparsity bar chart (requires matplotlib)
- `load_metrics_from_dir(output_dir)` — reconstructs the results dict from individual `metrics.json` files

#### `src/mase_kd/runners/run_pipeline.py`
The CLI entry point. Reads a YAML config from `cw/kx725/configs/`, applies any CLI overrides, dispatches to `ResNetPipeline`, and prints the Markdown comparison table.

Usage: see **Running experiments** below.

#### `src/mase_kd/vision/eval.py`
Generic evaluation utilities (model-agnostic):
- `count_parameters(model)` — counts total trainable parameters
- `benchmark_forward_latency(model, sample_input, warmup_steps, measure_steps)` — measures mean/std forward latency in milliseconds

#### `src/mase_kd/reporting/summarize.py`
- `summarize_metric_files(metrics_dir)` — walks a directory, loads all `metrics.json` files, and flattens them into a single dict. Used for programmatic result inspection.

---

## Running experiments

All commands are run from the repo root with `PYTHONPATH=src`.

### Smoke test (validates pipeline, ~1–3 min CPU)

```bash
PYTHONPATH=src python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 --profile smoke --sparsity 0.5
```

Output lands in `cw/kx725/outputs/resnet18/cifar10/sparsity_0.50/`.

### Full run (single sparsity + seed)

```bash
PYTHONPATH=src python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 --profile full --sparsity 0.85 --seed 0
```

### CIFAR-100

```bash
PYTHONPATH=src python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar100 --profile full --sparsity 0.7 --seed 0
```

### Aggregate multi-sparsity results

```bash
PYTHONPATH=src python3 cw/kx725/scripts/aggregate_results.py \
    --model resnet18 --dataset cifar10 --sparsities 0.5 0.7 0.85 \
    --output-dir cw/kx725/outputs/resnet18/cifar10
```

### Via Docker

```bash
docker exec -e PYTHONPATH=/workspace/src mase-dev-kd \
    python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 --profile smoke --sparsity 0.5
```

---

## Running tests

```bash
# All kx725 tests
pytest cw/kx725/ -v

# Unit tests only (fast, no GPU)
pytest cw/kx725/unit/ -v

# Integration tests only
pytest cw/kx725/integration/ -v

# Single test file
pytest cw/kx725/unit/test_passes.py -v
```

`cw/conftest.py` (at the parent level) automatically adds `src/` to `sys.path` for all tests under `cw/`.

---

## Output directory structure

After a run at `sparsity=0.5`:

```
cw/kx725/outputs/resnet18/cifar10/
└── sparsity_0.50/
    ├── A_dense/
    │   ├── best_student.pth          ← checkpoint reused as teacher for D/E
    │   ├── training_history.json     ← per-epoch loss + accuracy
    │   └── metrics.json              ← final test accuracy, params, sparsity
    ├── B_pruned/
    │   ├── pruned_student.pth
    │   └── metrics.json
    ├── C_ft/
    │   ├── best_student.pth
    │   ├── training_history.json
    │   └── metrics.json
    ├── D_kd/
    │   ├── best_student.pth
    │   ├── training_history.json
    │   └── metrics.json
    ├── E_kd_ft/
    │   ├── best_student.pth
    │   ├── training_history.json
    │   └── metrics.json
    ├── comparison_table.md           ← human-readable results summary
    ├── comparison_table.json         ← machine-readable results
    └── trade_off_plot.png            ← accuracy vs variant bar chart
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| CIFAR first conv: 3×3 stride-1, no maxpool | Preserves 32×32 spatial resolution; standard 7×7 stride-2 halves it to 4×4 after pooling |
| Self-KD: teacher = A checkpoint | No external teacher download required; dense model provides upper-bound signal |
| `make_permanent=True` in PrunePass | Fuses mask into weights so `state_dict()` is directly sparse; SGD may fill zeros in C/D/E (intentional — no mask reinstatement needed) |
| Fine-tune LR = 0.001, not 0.01 | After 100 dense epochs weights have converged; lr=0.01 causes catastrophic forgetting (C < B) |
| `alpha=0` for C and E steps | C and E are recovery steps using only hard labels; KD is only applied in D |
