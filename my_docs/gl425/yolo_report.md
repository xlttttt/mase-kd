# Logits-Based Knowledge Distillation for YOLOv8 Classification Models

## 1. Purpose

This system implements **logits-based knowledge distillation (KD)** for YOLOv8 classification models, following Hinton et al. (2015). The goal is to transfer the "dark knowledge" encoded in a large teacher network's soft output distribution to a smaller student network, enabling the student to recover—or exceed—the accuracy it would achieve through standard cross-entropy (CE) fine-tuning alone. We investigate whether KD provides a consistent benefit across datasets of varying classification complexity (CIFAR-10 with 10 classes, CIFAR-100 with 100 classes), and identify which hyperparameter settings ($\alpha$, $T$) maximise this benefit.

## 2. System Architecture

The system is built as a modular extension to the MASE (Machine-Learning Accelerator System Exploration) framework and consists of three layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Experiment Scripts                            │
│   P2_grid_search.py (CIFAR-100)  │  P3_grid_search.py (CIFAR-10)   │
│   - Load teacher & build student seed via MaseGraph                 │
│   - Grid sweep over (α, T); CE-only baseline (α = 0)               │
│   - Per-experiment: reset student → train → checkpoint → evaluate   │
│   - Append per-run metrics to CSV                                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼───────────────────────────────────────────┐
│                    YOLOLogitsDistiller                                │
│                 (src/mase_kd/vision/yolo_kd.py)                      │
│                                                                      │
│  ┌────────────┐   train_step()   ┌──────────────┐                   │
│  │  Teacher    │ ──(no_grad)───► │  KD Loss      │                  │
│  │  (frozen)   │   raw logits    │  Combiner     │                  │
│  └────────────┘                  │               │                  │
│  ┌────────────┐   forward +      │  α·T²·KL(σ   │                  │
│  │  Student    │ ──backprop────► │  (zs/T)‖σ    │                  │
│  │  (trained)  │   raw logits    │  (zt/T))     │                  │
│  └────────────┘                  │ +(1−α)·CE    │                  │
│                                  └──────┬───────┘                   │
│  train(): epoch loop, val tracking, best-epoch checkpointing        │
│  evaluate(): top-1, top-5, CE loss, forward latency, val KD loss    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ calls
┌──────────────────────────▼───────────────────────────────────────────┐
│                      Loss Functions                                  │
│               (src/mase_kd/core/losses.py)                           │
│                                                                      │
│  soft_logit_kl_loss(zs, zt, T)  →  T²·KL(log σ(zs/T) ‖ σ(zt/T))  │
│  hard_label_ce_loss(zs, y)      →  CE(zs, y)                        │
│  compute_distillation_loss(...)  →  (1−α)·CE + α·KL                 │
│  DistillationLossConfig(α, T)   →  validated dataclass               │
└──────────────────────────────────────────────────────────────────────┘
```

**Teacher–student pair.** The teacher is a fine-tuned `yolov8x-cls` (71.4M parameters); the student is an ImageNet-pretrained `yolov8m-cls` (18.9M parameters) — a ~3.8× capacity gap. Both are loaded via the MASE `MaseYoloClassificationModel` wrapper with `patch_yolo` to expose a standard `nn.Module` forward interface. The teacher is frozen (`requires_grad=False`, `eval()` mode for parameters but temporarily switched to `train()` mode during forward to obtain raw logits instead of the ultralytics Classify head's default softmax output).

## 3. Performance Metrics

| Metric | Rationale |
|--------|-----------|
| **Top-1 accuracy** (primary) | The standard classification metric; directly measures whether the student recovers teacher-level predictions. |
| **Top-5 accuracy** | Captures whether the student preserves the teacher's ranking of plausible classes—particularly useful on CIFAR-100 where multiple classes are visually similar. |
| **KD gain vs CE-only** ($\Delta$Acc) | Isolates the contribution of soft-target transfer by subtracting the CE-only baseline from the KD result. A positive $\Delta$ confirms that dark knowledge improves over hard-label training. |
| **Cross-entropy loss** | Monitors calibration: a model with high accuracy but high CE loss is over-confident on wrong predictions. |
| **Validation KD loss** | The temperature-scaled KL divergence on the held-out set; tracks how closely the student's distribution aligns with the teacher's. |

Top-1 accuracy is the primary selection criterion for best-epoch checkpointing. Top-5 accuracy is reported alongside to verify that the student preserves the teacher's inter-class similarity structure—the core promise of logits-based KD.

## 4. Design Decisions

1. **Raw-logits extraction via train-mode trick.** The ultralytics `Classify` head returns `x.softmax(1)` in eval mode. Applying `F.softmax(softmax/T)` downstream would produce a double-softmax, destroying the dark-knowledge signal. The fix is to temporarily switch the teacher to `train()` inside a `torch.no_grad()` block so the head returns raw logits. This was a critical bug fix that unblocked correct KD training.

2. **Strict shape assertion instead of silent truncation.** `_align_logits` raises a `ValueError` on any student–teacher shape mismatch rather than silently truncating. This was introduced after discovering that the `Classify` head returns different tensor formats (tuple vs single tensor) depending on train/eval mode, causing a 2× dimension mismatch that was previously masked.

3. **Epoch-based best-model checkpointing.** `train()` saves the student's `state_dict()` whenever validation top-1 accuracy improves, and the caller restores best weights before final evaluation. `strict=False` is required during `load_state_dict` because MASE's pruning pass registers sparsity masks as non-persistent buffers.

4. **Grid search over $\alpha$ and $T$.** Five soft-loss weights $\alpha \in \{0.3, 0.5, 0.7, 0.9, 1.0\}$ and five temperatures $T \in \{1, 2, 4, 8, 16\}$ yield 25 KD configurations plus 1 CE-only baseline per dataset. This covers the full range from a CE-dominated loss to the pure soft-target regime ($\alpha = 1.0$) recommended by Hinton et al.

5. **Shared infrastructure across experiments.** Each grid-search script loads the teacher and builds the student once, then deep-copies and resets the student for each experiment. This ensures identical starting weights across all 26 runs and reduces GPU memory churn.

## 5. Testing Approach and Results

**Protocol.** For each dataset (CIFAR-10, CIFAR-100), we run 26 experiments: 1 CE-only baseline ($\alpha = 0$) and 25 KD configurations. All share: AdamW optimiser, $lr = 5 \times 10^{-4}$, weight decay 0.05, batch size 128, 60 epochs, seed 42. Four conditions are compared per run: teacher, untrained student, CE-only fine-tuned student, and KD-distilled student.

**Results summary (CIFAR-100, P2):**

| Condition | Best Top-1 | Config |
|-----------|--------:|--------|
| Teacher (yolov8x-cls) | 68.33% | — |
| Untrained student | 0.93% | — |
| CE-only fine-tuned | 50.99% | $\alpha = 0$ |
| **Best KD** | **60.46%** | $\alpha = 1.0, T = 16$ |
| KD gain over CE | **+9.47 pp** | |

On CIFAR-100, KD consistently outperforms CE-only fine-tuning across all 25 configurations (minimum gain +0.48 pp at $\alpha = 0.5, T = 1$; maximum +9.47 pp at $\alpha = 1.0, T = 16$). Higher temperatures and higher $\alpha$ values generally produce larger gains, confirming Hinton et al.'s prediction that soft targets carry more useful signal when the temperature is high enough to spread probability mass across semantically related classes in a 100-class problem.

**Results summary (CIFAR-10, P3):**

| Condition | Best Top-1 | Config |
|-----------|--------:|--------|
| Teacher (yolov8x-cls) | 91.25% | — |
| Untrained student | 10.64% | — |
| CE-only fine-tuned | 83.04% | $\alpha = 0$ |
| **Best KD** | **84.93%** | $\alpha = 0.9, T = 8$ |
| KD gain over CE | **+1.88 pp** | |

On CIFAR-10, KD again improves over CE-only in 24 of 25 configurations, but the gains are smaller (maximum +1.88 pp). This is expected: with only 10 classes the teacher's soft distribution carries less inter-class similarity information. One configuration ($\alpha = 0.3, T = 1$) shows a marginal regression of $-0.16$ pp, consistent with $T = 1$ being a degenerate case where soft targets approximate hard labels.

**Key finding.** KD benefit scales with classification complexity: the improvement is roughly $5\times$ larger on CIFAR-100 (100 classes) than on CIFAR-10 (10 classes), consistent with the theory that dark knowledge is richer when there are more inter-class relationships to encode.
