# MASE-KD Project Plan

## 1. Motivation & Requirements Mapping

The project extends MASE (Machine-Learning Accelerator System Exploration) with a
knowledge distillation (KD) framework covering three model/dataset pairs and a
five-step experimental matrix (A–E) that evaluates the interaction between pruning,
fine-tuning, and knowledge distillation.

| Requirement | Implementation |
|---|---|
| Logits-level KD loss | `src/mase_kd/core/losses.py` — `compute_distillation_loss` |
| Intermediate feature KD (TinyBERT-style) | `src/mase_kd/distillation/losses.py` |
| Layer alignment for feature KD | `src/mase_kd/distillation/mapping.py` — `generate_layer_mapping` |
| BERT/SST-2 KD pipeline | `src/mase_kd/nlp/bert_kd.py` — `BertKDTrainer` |
| YOLO/COCO KD pipeline | `src/mase_kd/vision/yolo_kd_train.py` — `YOLOKDRunner` |
| ResNet18/CIFAR-10 KD pipeline | `src/mase_kd/vision/resnet_kd.py` — `ResNetKDTrainer` |
| Global magnitude pruning | `src/mase_kd/passes/prune_pass.py` — `PrunePass` |
| A–E pipeline orchestration | `src/mase_kd/passes/pipeline.py` — `BertPipeline`, `ResNetPipeline`, `YoloPipeline` |
| Metrics export + comparison table | `src/mase_kd/passes/export_pass.py` — `ExportMetricsPass` |
| One-command pipeline scripts | `experiments/scripts/run_{bert,resnet,yolo,all}_pipeline.py` |
| Unit/integration tests | `cw/unit/test_passes.py`, `cw/integration/test_resnet_smoke.py` |

---

## 2. System Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              Pipeline Orchestrator               │
                    │  BertPipeline / ResNetPipeline / YoloPipeline   │
                    └────────────────────┬────────────────────────────┘
                                         │
              ┌──────────┬───────────────┼──────────────┬──────────────┐
              ▼          ▼               ▼              ▼              ▼
          Pass A      Pass B          Pass C         Pass D         Pass E
         (Dense)    (Pruned)        (Pruned        (Pruned        (Pruned
         Train        PrunePass      +FT)           +KD)          +KD+FT)
           │            │           Train           Train KD       Train CE
           │            │           CE only         alpha>0        alpha=0
           ▼            ▼               ▼              ▼              ▼
       metrics.json  metrics.json  metrics.json  metrics.json  metrics.json

                    ┌────────────────────────────────────────────────┐
                    │              ExportMetricsPass                  │
                    │  comparison_table.md/.json + trade_off_plot.png │
                    └────────────────────────────────────────────────┘

Data flow through each pass:
  (model, pass_args, info_dict) → pass.run() → (model', info_dict')
```

---

## 3. Model Specifications

| Model | Dataset | Teacher | Student | Primary Metric |
|---|---|---|---|---|
| BERT | GLUE SST-2 | `textattack/bert-base-uncased-SST-2` (12-layer, 768-hidden) | 4-layer, 256-hidden BertConfig (~13M params) | Accuracy |
| ResNet18 | CIFAR-10 | Dense ResNet18 trained in Step A (same arch) | ResNet18 with CIFAR-10 first-conv (kernel=3, stride=1) | Accuracy |
| YOLOv8n | COCO | `yolov8m.pt` (25.9M params) | `yolov8n.yaml` (3.2M params, from scratch) | mAP50 |

---

## 4. Pass Design Specification

| Pass | Input | Output | Args | Artifacts |
|---|---|---|---|---|
| Train (A) | fresh student | trained weights | trainer config | `best_student.*`, `training_history.json`, `metrics.json` |
| PrunePass (B) | dense weights | pruned weights | `PruneConfig(sparsity=0.5)` | `metrics.json` (sparsity stats) |
| FinetuneCE (C) | pruned weights | fine-tuned weights | trainer config (alpha=0) | `best_student.*`, `metrics.json` |
| TrainKD (D) | pruned weights | KD-trained weights | trainer config (alpha>0, teacher) | `best_student.*`, `metrics.json` |
| FinetuneCE (E) | KD weights | fine-tuned weights | trainer config (alpha=0, fewer epochs) | `best_student.*`, `metrics.json` |
| ExportMetrics | `{A:…,B:…,C:…,D:…,E:…}` | None (writes files) | output_dir, model_name | `comparison_table.{md,json}`, `trade_off_plot.png` |

---

## 5. Experimental Matrix

| Step | BERT/SST-2 | ResNet18/CIFAR-10 | YOLOv8n/COCO |
|---|---|---|---|
| A — Dense | Train student from scratch, alpha=0 | Train ResNet18 from scratch, alpha=0 | Train yolov8n from scratch, alpha=0 |
| B — Pruned | L1 global pruning (sparsity=0.5) on A's Linear weights | L1 global pruning (sparsity=0.5) on A's Conv2d+Linear | L1 global pruning (sparsity=0.5) on A's Conv2d |
| C — Pruned+FT | Fine-tune B's weights with CE (alpha=0) | Fine-tune B's weights with CE (alpha=0) | Fine-tune B's weights with task loss (alpha=0) |
| D — Pruned+KD | KD from B's weights with teacher (alpha=0.5) | KD from B's weights with A's dense model as teacher (alpha=0.5) | KD from B's weights with yolov8m teacher (alpha=0.5) |
| E — Pruned+KD+FT | Fine-tune D's weights with CE (alpha=0, fewer epochs) | Fine-tune D's weights with CE (alpha=0, fewer epochs) | Fine-tune D's weights with task loss (alpha=0, fewer epochs) |

---

## 6. Metrics & Evaluation Criteria

All `metrics.json` files use a consistent key schema:

```json
{
  "accuracy": 0.82,            // BERT/ResNet (val set)
  "mAP50": 0.45,               // YOLO only
  "mAP50_95": 0.31,            // YOLO only
  "params_total": 13000000,
  "params_nonzero": 6500000,
  "sparsity": 0.5,
  "delta_vs_dense": -0.02      // filled by ExportMetricsPass
}
```

Success criteria:
- **D ≥ C**: Pruned+KD should outperform Pruned+FT (demonstrates KD benefit)
- **E ≥ D**: Pruned+KD+FT should match or exceed Pruned+KD
- **sparsity ≈ 0.5**: PrunePass achieves target within ±2%
- **A ≈ teacher** (within ~5%): validates the teacher is a meaningful upper bound

---

## 7. Milestones & Implementation Order

1. `project_plan.md` — deliverable, no code dependencies ✅
2. `src/mase_kd/vision/resnet_kd.py` — ResNet18/CIFAR-10 KD trainer
3. `src/mase_kd/passes/prune_pass.py` — global L1 pruning wrapper
4. `src/mase_kd/passes/export_pass.py` — metrics aggregation + plots
5. `src/mase_kd/passes/pipeline.py` — A–E orchestration for all 3 models
6. `src/mase_kd/passes/__init__.py` — re-exports
7. `src/mase_kd/config/schema.py` — add `ResNetKDConfig`
8. Experiment configs (TOML/YAML) for bert, resnet, yolo
9. Experiment scripts (pipeline runners)
10. Unit tests: `cw/unit/test_passes.py`
11. Integration tests: `cw/integration/test_resnet_smoke.py`
12. Extend `cw/unit/test_config_schema.py`
13. `README.md` rewrite

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| YOLO `student_weights` path not found after pruning | Save pruned state_dict as `.pt`, load via `_load_yolo_model` wrapper which handles state_dict loading |
| ResNet CIFAR-10 spatial collapse (32×32 → 8×8) | Replace first conv (kernel=7,stride=2) with (kernel=3,stride=1) and remove maxpool |
| `v8DetectionLoss` returns shape-[3] vector | Always call `.sum()` before combining with KD loss |
| Teacher not available for ResNet (no external pretrained) | Use dense A checkpoint as teacher for D and E; saves separately as `dense_teacher.pth` |
| `tomllib` only reads TOML (no write) | Configs are written manually; pipeline reads via `tomllib.loads()` |
| Docker DataLoader worker OOM | `--ipc=host` flag; DataLoader `num_workers` tuned per model |
| Pruning BERT teacher accidentally | Only deep-copy and prune the **student**; teacher instance is never touched |
| `prune.remove()` incompatible with HF `save_pretrained` | Use `torch.save(model.state_dict())` for non-HF models; BERT student still uses `save_pretrained` after removing masks |
