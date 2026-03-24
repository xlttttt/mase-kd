# MASE-KD: Knowledge Distillation Extension for MASE

> **ADLS 2026 coursework project** — knowledge distillation (KD) pipelines for
> ResNet18, YOLOv8, BERT, GPT2 built on top of the MASE
> framework from Imperial College London's DeepWok Lab.

---

## Folder Structure

```
mase-kd/
│
├── cw/             # Test scripts
│   ├── gl425/      # YOLOv8
│   ├── kx725/      # RESNET18
│   ├── by425/      # GPT2
│   └── sz2125/     # BERT
│
├── my_docs/        # Documentations
│   ├── gl425/
│   ├── kx725/
│   ├── by425/
│   └── sz2125/
│
├── src/            # Source code
│   ├── chop/       # Original mase source code
│   ├── mase-kd/    # Distillation library
│   │   ├── core/   # Utils for distillation pipeline (i.e. loss function)
│   │   ├── nlp/    # Code for BERT and GPT2 distillation pipeline
│   │   └── vision/ # Code for RESNET18 and YOLOv8 distillation pipeline
```