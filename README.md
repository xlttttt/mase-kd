# MASE-KD: Knowledge Distillation Extension for MASE

> **ADLS 2026 coursework project** — knowledge distillation (KD) pipelines for
> ResNet18/CIFAR-10, BERT/SST-2, and YOLOv8/COCO, built on top of the MASE
> framework from Imperial College London's DeepWok Lab.

---

## Overview

MASE-KD adds a structured A–E experimental matrix to the MASE compiler stack:

```
A — Dense baseline          (student trained from scratch, no KD)
B — Pruned                  (L1 global magnitude pruning, no recovery)
C — Pruned + Finetune       (CE fine-tuning of pruned model)
D — Pruned + KD             (knowledge distillation from frozen teacher)
E — Pruned + KD + Finetune  (fine-tune the KD checkpoint with CE)
```

```
┌───────────────────────────────────────────────────────────┐
│  CLI: python -m mase_kd.runners.run_pipeline              │
│       --model resnet18 --dataset cifar10 --sparsity 0.5   │
└─────────────────────────────┬─────────────────────────────┘
                              │
         ┌──────┬─────────────┼───────────┬────────────┐
         ▼      ▼             ▼           ▼            ▼
       A Dense  B Prune    C Prune+FT  D Prune+KD  E KD+FT
         │      │PrunePass    │           │            │
         │      └─────────────┴───────────┘            │
         │           pruned_student.pth                 │
         └────────────────────── teacher ───────────────┘
                    (dense A checkpoint)
                              │
               ExportMetricsPass → comparison_table.md
                                    trade_off_plot.png
```

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.11.9. CUDA is auto-detected. PyYAML is required for
YAML configs (`pip install pyyaml`).

---

## Quick Smoke Tests (validate pipeline, ~1–3 min CPU)

```bash
# ResNet18 / CIFAR-10 — runs A-E with 5 000 samples, 2/1/1/1 epochs
python -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 \
    --sparsity 0.5 --profile smoke

# Output: outputs/resnet18/cifar10/sparsity_0.50/
#   A_dense/   best_student.pth, training_history.json, metrics.json
#   B_pruned/  pruned_student.pth, metrics.json
#   C_ft/      best_student.pth, training_history.json, metrics.json
#   D_kd/      best_student.pth, training_history.json, metrics.json
#   E_kd_ft/   best_student.pth, training_history.json, metrics.json
#   comparison_table.md, comparison_table.json, trade_off_plot.png
```

---

## Full Report-Ready Experiments (ResNet18/CIFAR-10)

### Two sparsity levels

```bash
# sparsity = 0.50, seed 0  (~2–4 h on a single GPU)
python -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 \
    --profile full --sparsity 0.5 --seed 0

# sparsity = 0.70, seed 0
python -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 \
    --profile full --sparsity 0.7 --seed 0
```

Full config defaults (see `experiments/configs/resnet18_cifar10_full.yaml`):
- Dense training: 100 epochs, SGD+cosine, lr=0.1
- Finetune (C): 30 epochs, lr=0.01
- KD (D): 30 epochs, alpha=0.5, T=4.0, lr=0.01
- KD+FT (E): 10 epochs, lr=0.001

### Aggregate results across sparsities

```bash
python experiments/scripts/aggregate_results.py \
    --model resnet18 --dataset cifar10 \
    --sparsities 0.5 0.7 \
    --output-dir outputs/resnet18/cifar10
```

Generates:
- `outputs/resnet18/cifar10/report_ready_tables/comparison_table_sparsity_0.50.md`
- `outputs/resnet18/cifar10/report_ready_tables/comparison_table_sparsity_0.70.md`
- `outputs/resnet18/cifar10/report_ready_tables/combined_table.md`
- `outputs/resnet18/cifar10/figures/accuracy_vs_variant.png`
- `outputs/resnet18/cifar10/figures/accuracy_vs_sparsity.png`

### Resource requirements

| Profile | GPU | Approx. time | VRAM |
|---|---|---|---|
| smoke (5 k samples) | optional | 1–3 min CPU | — |
| full (50 k, 100 ep) | recommended | 2–4 h (GPU) | ~2 GB |
| full × 3 seeds | recommended | 6–12 h (GPU) | ~2 GB |

---

## BERT / SST-2 Pipeline

```bash
# Standalone runs (existing)
python3 experiments/scripts/run_bert_baseline.py --epochs 5 --output-dir outputs/bert_baseline
python3 experiments/scripts/run_bert_kd.py --epochs 5 --output-dir outputs/bert_kd

# A-E pipeline (smoke)
python -m mase_kd.runners.run_pipeline \
    --model bert --dataset sst2 --profile smoke --sparsity 0.5
```

## YOLO / COCO Pipeline

```bash
# Standalone runs (existing)
python3 experiments/scripts/run_yolo_baseline.py --epochs 50 --data coco.yaml --output-dir outputs/yolo_baseline
python3 experiments/scripts/run_yolo_kd.py --epochs 50 --data coco.yaml --output-dir outputs/yolo_kd

# A-E pipeline smoke (coco8)
python -m mase_kd.runners.run_pipeline \
    --model yolo --dataset coco --profile smoke --sparsity 0.5
```

---

## Output Directory Layout

```
outputs/resnet18/cifar10/
└── sparsity_0.50/
    ├── A_dense/
    │   ├── best_student.pth
    │   ├── training_history.json
    │   └── metrics.json          # {accuracy, params_nonzero, params_total, sparsity}
    ├── B_pruned/
    │   ├── pruned_student.pth
    │   └── metrics.json          # {accuracy, sparsity_actual, params_nonzero, ...}
    ├── C_ft/   (same as A_dense)
    ├── D_kd/   (same as A_dense)
    ├── E_kd_ft/(same as A_dense)
    ├── comparison_table.md
    ├── comparison_table.json
    └── trade_off_plot.png
```

---

## Config Reference

| Key | Smoke | Full | Description |
|---|---|---|---|
| `dense_training.epochs` | 2 | 100 | Dense baseline training epochs |
| `finetune.epochs` | 1 | 30 | Pruned+FT epochs |
| `kd.epochs` | 1 | 30 | Pruned+KD epochs |
| `kd_finetune.epochs` | 1 | 10 | KD+FT epochs |
| `kd.alpha` | 0.5 | 0.5 | KD soft loss weight |
| `kd.temperature` | 4.0 | 4.0 | KD temperature |
| `pruning.sparsity` | 0.5 | 0.5/0.7 | Global L1 pruning sparsity |
| `data.subset_size` | 5000 | null | Training subset (null = full) |

---

## Running Tests

```bash
# Unit tests only (no downloads, CPU, ~10 s)
pytest cw/unit/ -v

# Integration smoke (ResNet synthetic data, ~30 s CPU)
pytest cw/integration/test_resnet_smoke.py -v

# All integration tests (may download CIFAR-10/SST-2)
pytest cw/integration/ -v -m integration

# All tests
pytest cw/ -v
```

---

## Docker Usage

```bash
# BERT KD (GPU)
docker run --rm --gpus all --ipc=host \
    -v /home/xukun/projects/mase-kd:/workspace \
    -w /workspace \
    -e TOKENIZERS_PARALLELISM=false \
    deepwok/mase-docker-cpu:latest \
    python3 -m mase_kd.runners.run_pipeline \
        --model bert --dataset sst2 --profile smoke --sparsity 0.5

# ResNet KD smoke (CPU container)
docker run --rm --ipc=host \
    -v $(pwd):/workspace -w /workspace \
    deepwok/mase-docker-cpu:latest \
    python3 -m mase_kd.runners.run_pipeline \
        --model resnet18 --dataset cifar10 --profile smoke --sparsity 0.5
```

Key flags: `--ipc=host` is **required** (avoids DataLoader shared-memory crashes);
`python3` not `python` (no symlink in image); run containers sequentially on 8 GB GPUs.

---

## Architecture Notes

- **Teacher (ResNet)**: Dense ResNet18 trained in Step A; reused as KD teacher for D/E.
  No external pretrained teacher required. Set `teacher.arch: resnet34` in the full
  config if a separate larger teacher is desired.
- **Pruning**: Global L1 unstructured (`torch.nn.utils.prune.global_unstructured`);
  masks made permanent via `prune.remove()` before checkpointing.
- **KD loss**: `L = (1−α)·L_hard + α·T²·KL(student‖teacher)`.

---

# Machine-Learning Accelerator System Exploration Tools

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Doc][doc-shield]][doc-url]

[contributors-shield]: https://img.shields.io/github/contributors/DeepWok/mase.svg?style=flat
[contributors-url]: https://github.com/DeepWok/mase/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/DeepWok/mase.svg?style=flat
[forks-url]: https://github.com/DeepWok/mase/network/members
[stars-shield]: https://img.shields.io/github/stars/DeepWok/mase.svg?style=flat
[stars-url]: https://github.com/DeepWok/mase/stargazers
[issues-shield]: https://img.shields.io/github/issues/DeepWok/mase.svg?style=flat
[issues-url]: https://github.com/DeepWok/mase/issues
[license-shield]: https://img.shields.io/github/license/DeepWok/mase.svg?style=flat
[license-url]: https://github.com/DeepWok/mase/blob/master/LICENSE.txt
[issues-shield]: https://img.shields.io/github/issues/DeepWok/mase.svg?style=flat
[issues-url]: https://github.com/DeepWok/mase/issues
[doc-shield]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[doc-url]: https://deepwok.github.io/mase/

## Overview

Mase is a Machine Learning compiler based on PyTorch FX, maintained by researchers at Imperial College London. We provide a set of tools for inference and training optimization of state-of-the-art language and vision models. The following features are supported, among others:

- Efficient AI Optimization: 
  MASE provides a set of composable tools for optimizing AI models. The tools are designed to be modular and can be used in a variety of ways to optimize models for different hardware targets. The tools can be used to optimize models for inference, training, or both. We support features such as the following:

  - Quantization Search: mixed-precision quantization of any PyTorch model. We support microscaling and other numerical formats, at various granularities.
  - Quantization-Aware Training (QAT): finetuning quantized models to minimize accuracy loss.
  - And more!

- Hardware Generation: automatic generation of high-performance FPGA accelerators for arbitrary Pytorch models, through the Emit Verilog flow.

- Distributed Deployment (Beta): Automatic parallelization of models across distributed GPU clusters, based on the Alpa algorithm.

For more details, refer to the Tutorials. If you enjoy using the framework, you can support us by starring the repository on GitHub!


## MASE Publications

* Fast Prototyping Next-Generation Accelerators for New ML Models using MASE: ML Accelerator System Exploration, [link](https://arxiv.org/abs/2307.15517)
  ```
  @article{cheng2023fast,
  title={Fast prototyping next-generation accelerators for new ml models using mase: Ml accelerator system exploration},
  author={Cheng, Jianyi and Zhang, Cheng and Yu, Zhewen and Montgomerie-Corcoran, Alex and Xiao, Can and Bouganis, Christos-Savvas and Zhao, Yiren},
  journal={arXiv preprint arXiv:2307.15517},
  year={2023}}
  ```
* MASE: An Efficient Representation for Software-Defined ML Hardware System Exploration, [link](https://openreview.net/forum?id=Z7v6mxNVdU)
  ```
  @article{zhangmase,
  title={MASE: An Efficient Representation for Software-Defined ML Hardware System Exploration},
  author={Zhang, Cheng and Cheng, Jianyi and Yu, Zhewen and Zhao, Yiren}}
  ```
### Repository structure

This repo contains the following directories:
* `src/chop` - MASE's software stack
* `src/mase_components` - Internal hardware library
* `src/mase_cocotb` - Internal hardware testing flow
* `src/mase_hls` - HLS component of MASE
* `scripts` - Run and test scripts  
* `test` - Unit testing 
* `docs` - Documentation
* `mlir-air` - MLIR AIR for ACAP devices
* `setup.py` - Installation entry point
* `Docker` - Docker container configurations

## MASE Dev Meetings

* Direct [Google Meet link](meet.google.com/fke-zvii-tgv)
* Join the [Mase Slack](https://join.slack.com/t/mase-tools/shared_invite/zt-2gl60pvur-pktLLLAsYEJTxvYFgffCog)
* If you want to discuss anything in future meetings, please add them as comments in the [meeting agenda](https://docs.google.com/document/d/12m96h7gOhhmikniXIu44FJ0sZ2mSxg9SqyX-Uu3s-tc/edit?usp=sharing) so we can review and add them.

## Donation  

If you think MASE is helpful, please [donate](https://www.buymeacoffee.com/mase_tools) for our work, we appreciate your support!

<img src='./docs/imgs/bmc_qr.png' width='250'>


## easy start

cd ~
docker ps -a      to check container's name
docker start -ai mase     start container


