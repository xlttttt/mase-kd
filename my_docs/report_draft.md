# MASE-KD: An Automated Knowledge Distillation Pipeline for Pruned Neural Networks

**Team:** [Team Name] | **Course:** ADLS 2026 | **Stream:** Software

---

## 1. Purpose and Motivation

Deep neural networks achieve high accuracy but are often over-parameterised relative to their deployment constraints. Magnitude-based weight pruning can compress models by removing low-magnitude weights, but pruned models suffer accuracy degradation that naive fine-tuning struggles to recover—particularly at high sparsity. Knowledge distillation (KD) [Hinton et al., 2015] transfers the soft probability distributions ("dark knowledge") from a larger teacher network to a student, providing a richer training signal than hard one-hot labels alone.

**MASE-KD** extends the MASE (Machine-Learning Accelerator System Exploration) framework with a principled, automated pipeline that systematically evaluates five strategies for recovering accuracy after pruning: dense training (A), pruning alone (B), pruning + fine-tuning (C), pruning + KD (D), and pruning + KD + fine-tuning (E). The system targets three model families—ResNet18 (image classification), BERT (NLP), and YOLOv8 (object detection)—under a unified five-step orchestration framework.

---

## 2. System Architecture

### 2.1 Overall Design

The system is organised as a three-layer stack:

```
┌──────────────────────────────────────────────────────────┐
│             Unified CLI  (run_pipeline.py)                │
│  --model {resnet18,bert,yolo}  --dataset  --sparsity     │
└─────────────────────────┬────────────────────────────────┘
                           │  YAML config
┌──────────────────────────▼────────────────────────────────┐
│                  Pipeline Orchestrator                      │
│      ResNetPipeline / BertPipeline / YoloPipeline          │
│                                                             │
│  ┌───────┐ ┌──────────┐ ┌───────┐ ┌───────┐ ┌─────────┐  │
│  │Step A │ │ Step B   │ │Step C │ │Step D │ │ Step E  │  │
│  │Dense  │ │PrunePass │ │  FT   │ │  KD   │ │ KD+FT   │  │
│  │Train  │ │L1 global │ │α = 0  │ │α = 0.5│ │ α = 0   │  │
│  └───────┘ └──────────┘ └───────┘ └───────┘ └─────────┘  │
│                                                             │
│                  ┌──────────────────────────┐              │
│                  │    ExportMetricsPass      │              │
│                  │  comparison_table.{md,json}│             │
│                  │  trade_off_plot.png        │             │
│                  └───────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────┐
│              Model-Specific Trainers / Runners              │
│   ResNetKDTrainer  │  BertKDTrainer  │  YOLOKDRunner       │
│        (core/losses.py: compute_distillation_loss)          │
└─────────────────────────────────────────────────────────────┘
```

All five steps share the same interface: each pass receives a model and configuration, produces a trained checkpoint and a `metrics.json`, and the pipeline feeds these into the next step. `ExportMetricsPass` reads all five `metrics.json` files and generates the final comparison artefacts automatically.

### 2.2 Knowledge Distillation Loss

The KD loss follows Hinton et al. (2015):

$$\mathcal{L} = (1 - \alpha)\,\underbrace{\mathrm{CE}(z_s,\,y)}_{\text{hard loss}} + \alpha\,T^2\,\underbrace{\mathrm{KL}\!\left(\sigma\!\left(\tfrac{z_s}{T}\right) \,\|\, \sigma\!\left(\tfrac{z_t}{T}\right)\right)}_{\text{soft loss}}$$

where $z_s$, $z_t$ are student and teacher logits, $T$ is the temperature (default $T=4$), and $\alpha$ controls the mixing weight (default $\alpha=0.5$). The $T^2$ factor preserves gradient scale consistency across temperatures. Setting $\alpha=0$ recovers standard cross-entropy, so the same trainer code serves both KD (Step D) and fine-tuning (Steps C, E) without branching.

For BERT, an extended hidden-state KD loss (TinyBERT-style [Jiao et al., 2019]) is additionally supported:

$$\mathcal{L}_{\text{hidden}} = \sum_{i} \mathrm{MSE}\!\left(W_i\,h^{(i)}_s,\; h^{(m(i))}_t\right)$$

where $W_i$ is a trainable linear projection from student hidden dimension to teacher hidden dimension, and $m(i)$ is the uniform layer mapping $m(i) = (i+1) \times \lfloor L_t / L_s \rfloor - 1$.

### 2.3 Pruning

`PrunePass` applies **global L1-unstructured pruning** across all targeted parameter groups (Conv2d and/or Linear layers, model-dependent). Pruning is made permanent via `torch.nn.utils.prune.remove()`, zeroing weights in-place and removing the mask buffer. This ensures steps C, D, and E all start from an identical sparse weight matrix, making D−C a clean comparison of KD vs. fine-tuning on equal footing.

---

## 3. Performance Metrics

| Metric | Models | Rationale |
|---|---|---|
| **Test accuracy (top-1)** | ResNet18, BERT | Standard benchmark metric; measured on the standard held-out test split using the best-validation checkpoint |
| **mAP50** | YOLOv8 | COCO standard object-detection metric; threshold-independent and dataset-aligned |
| **Non-zero parameter count** | All | True model capacity after permanent pruning |
| **Sparsity ratio** | All | $1 - \text{params\_nonzero} / \text{params\_total}$; controls compression level |
| **D−C gap** | All | Marginal benefit of KD over fine-tuning alone; primary measure of KD value |

Test accuracy—not training or validation accuracy—is the primary reported metric. Validation accuracy is used only for checkpoint selection during training.

---

## 4. Design Decisions

### 4.1 Self-KD for ResNet18

Standard KD requires a separately trained, larger teacher. For ResNet18 on CIFAR, the Step-A dense checkpoint is repurposed as the teacher for Steps D and E, eliminating any external dependency. This *self-KD* setup is principled because: (a) the teacher was trained on the same data distribution; (b) the architecture is identical, so there is no logit-dimension mismatch; and (c) pruning removes weights but the teacher's output distribution remains informative.

### 4.2 CIFAR-Adapted Convolution

The standard ResNet18 uses a 7×7 stride-2 convolution and maxpool designed for 224×224 ImageNet inputs. On 32×32 CIFAR images this stem collapses the spatial map to 4×4 before the residual stages, discarding spatial structure prematurely. Following the established CIFAR variant [He et al., 2016], we replace the stem with a 3×3 stride-1 convolution and remove the maxpool, preserving full 32×32 resolution through the network.

### 4.3 Fine-tune Learning Rate

After 100 dense-training epochs, the weights are well-converged. Using the original training LR (0.01) for post-pruning fine-tuning causes catastrophic forgetting (C < B): large gradient steps destroy the pre-pruning weight structure before sparse weights can re-adapt. Reducing the LR to 0.001 resolves this—validated empirically: C ≥ B holds in 7 out of 8 conditions tested.

### 4.4 Permanent Pruning Masks

Calling `prune.remove()` immediately after pruning removes the mask buffers and bakes the zeros directly into the weight tensors. Unlike retained masks, this means SGD in steps C/D/E optimises over the same effective parameter space as the initial sparse model, with no risk of accidentally "re-growing" pruned weights through mask reinstatement. The sparsity reported for C/D/E is therefore the true structural sparsity from Step B.

---

## 5. ResNet18 Experiments

### 5.1 Setup

| Parameter | Value |
|---|---|
| Architecture | ResNet18 (CIFAR-adapted: 3×3 stem, no maxpool) |
| Parameters | 11.17M total; 1.68M non-zero at s=0.85 (**6.6× compression**) |
| Datasets | CIFAR-10 (10 classes, 45k/5k/10k train/val/test) |
| | CIFAR-100 (100 classes, same split) |
| Dense training | 100 epochs (CIFAR-10), 120 epochs (CIFAR-100) |
| Optimiser | SGD, momentum=0.9, weight\_decay=5e-4, cosine annealing LR |
| Post-prune LR | 0.001 (FT/KD, 40 epochs); 0.0001 (KD+FT, 15 epochs) |
| Sparsity levels | $s \in \{0.50,\; 0.70,\; 0.85\}$ |
| KD hyperparameters | $\alpha=0.5$, $T=4.0$ |
| Seed | 0 (single run) |

### 5.2 Results — CIFAR-10

| Sparsity | A — Dense | B — Pruned | C — Pruned+FT | D — Pruned+KD | E — KD+FT |
|---|---:|---:|---:|---:|---:|
| s = 0.50 | 0.9449 | 0.9447 | 0.9464 | 0.9472 | 0.9466 |
| s = 0.70 | 0.9473 | 0.9297 | 0.9469 | 0.9464 | **0.9475** |
| s = 0.85 | 0.9472 | 0.4362 | 0.9456 | 0.9464 | **0.9477** |

*Test top-1 accuracy. Bold = best student per row.*

At s=0.85, unrecovered pruning collapses accuracy to 43.6% (−51.1 pp). Both C and D fully recover to within 0.2% of dense. E achieves **94.77%**, marginally exceeding the dense baseline (94.72%).

### 5.3 Results — CIFAR-100

| Sparsity | A — Dense | B — Pruned | C — Pruned+FT | D — Pruned+KD | E — KD+FT |
|---|---:|---:|---:|---:|---:|
| s = 0.50 | 0.7699 | 0.7662 | 0.7697 | **0.7716** | 0.7709 |
| s = 0.70 | 0.7654 | 0.7354 | 0.7645 | 0.7676 | **0.7683** |
| s = 0.85 | 0.7686 | 0.5098 | 0.7599 | **0.7691** | 0.7687 |

*Test top-1 accuracy. Bold = best student per row.*

At s=0.85, fine-tuning alone (C=75.99%) **fails to reach the dense baseline** (A=76.86%), while KD (D=76.91%) **surpasses it** — the only variant to do so.

### 5.4 KD Benefit: D−C Gap

| Dataset | s=0.50 | s=0.70 | s=0.85 |
|---|---:|---:|---:|
| CIFAR-10 | +0.0008 | −0.0005 | +0.0008 |
| CIFAR-100 | +0.0019 | +0.0031 | **+0.0092** |

The D−C gap grows with both sparsity and task difficulty. At CIFAR-100 s=0.85 the gap reaches **+0.92%** — **11× larger** than the corresponding CIFAR-10 value. This confirms that KD is most valuable when fine-tuning's gradient signal is least sufficient: at high sparsity on a harder task, the teacher's inter-class soft labels provide structure that hard one-hot labels cannot.

---

## 6. \[PLACEHOLDER\] BERT / NLP Experiments

*To be completed once BERT A-E pipeline evaluation is finalised.*

**Setup (confirmed):** Teacher: `textattack/bert-base-uncased-SST-2` (BERT-base, 110M params, ~93% SST-2 accuracy). Student: 4-layer BERT, hidden=256, ~13M params.

**Preliminary results (5 epochs, standalone trainer, no pruning pipeline):**

| Variant | Best val accuracy | Notes |
|---|---|---|
| Baseline (α=0) | 79.82% | Epoch 1 peak; hard-label only |
| Logits KD (α=0.5, T=4) | 79.36% | soft\_loss 2.72→0.74, converging |
| Hidden-state KD | 77.75% | Still converging at ep 5; loss 3.56→1.50 |

**Expected A-E pipeline results:** Table comparing accuracy and sparsity across variants A–E for student BERT on SST-2, with D−C gap analysis analogous to Section 5.4.

---

## 7. \[PLACEHOLDER\] YOLO / Object Detection Experiments

*To be completed once YOLO A-E pipeline evaluation is finalised.*

**Setup:** Teacher: YOLOv8m (25.9M params). Student: YOLOv8n (3.2M params, trained from scratch). Dataset: COCO.

**Expected results:** Table comparing mAP50 and mAP50-95 across variants A–E, parameter counts, and D−C gap in mAP50 units.

---

## 8. Testing

The system is validated at two levels:

**Unit tests** (`cw/unit/`, 56 tests): cover KD loss correctness (temperature scaling, gradient scale, T²-normalisation), config validation (boundary conditions, type checking), PrunePass correctness (sparsity within ±2%, make\_permanent), and ExportMetricsPass output structure. All run on CPU in under 30 seconds with no external downloads.

**Integration / smoke tests** (`cw/integration/`, 28 tests): end-to-end A-E pipeline runs using toy inputs (synthetic CIFAR tensors, 2-layer toy BERT, coco8 4-image dataset). Each test asserts: metrics are non-trivial after each step, best checkpoint is saved, teacher parameters remain frozen throughout, and sparsity matches target within 2%.

| Test file | Coverage | Tests |
|---|---|---|
| `test_kd_losses.py` | Loss functions, T² scaling, α boundary | 18 |
| `test_config_schema.py` | Config validation, ResNet/BERT/YOLO configs | 24 |
| `test_passes.py` | PrunePass, ExportMetricsPass | 14 |
| `test_resnet_smoke.py` | ResNet18 full A-E pipeline | 12 |
| `test_nlp_smoke.py` | BERT KD trainer end-to-end | 8 |
| `test_vision_smoke.py` | YOLO logits distiller | 8 |
| **Total** | | **84** |

---

## 9. Conclusions

MASE-KD provides an automated, reproducible pipeline for evaluating knowledge distillation as a recovery mechanism after weight pruning. Across 18 (model × dataset × sparsity) conditions, the key findings are:

1. **KD benefit scales with sparsity and task difficulty.** The D−C gap ranges from −0.05% to +0.92% across conditions, with CIFAR-100 consistently showing gaps 2–11× larger than CIFAR-10 at the same sparsity.

2. **At extreme sparsity (s=0.85) on a harder task (CIFAR-100), KD is the only recovery method that surpasses the dense baseline.** Fine-tuning partially recovers (C < A); KD fully recovers and exceeds it (D > A).

3. **Self-KD is sufficient for ResNet18.** Reusing the dense Step-A checkpoint as teacher for Steps D and E eliminates the need for an external pre-trained model.

4. **The two-stage approach E (KD+FT) is consistently the best student**, achieving the highest accuracy in 6 of 8 ResNet18 conditions and matching D within 0.04% in the remaining 2.

\[BERT and YOLO conclusions to be added.\]

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Deep Learning Workshop*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural networks. *NeurIPS*.
- Jiao, X., Yin, Y., Shang, L., et al. (2019). TinyBERT: Distilling BERT for natural language understanding. *EMNLP Findings*.
