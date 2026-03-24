# MASE-Based Team Project Plan (Software Stream)

## Project Objective

Build an automated, principled optimization workflow based on MASE by integrating **Knowledge Distillation (KD)** for:

- **Vision**: YOLO (object detection)

The system will demonstrate optimization gains and trade-offs under practical resource limits (**8–12GB GPU**), with reproducible experiments and testing.

## Scope and Design Choices

### Why this scope is feasible in one month

- Keep one core optimization method (KD) on vision.
- Reuse existing MASE model support for YOLO.
- Use a **separate KD module folder** to avoid risky refactors to core source.
- Prioritize logits-based KD first; treat advanced KD (e.g., feature KD) as stretch.

### Integration strategy (based on current MASE capabilities)

- **YOLO track**:
	- Use YOLO training/distillation workflow through Ultralytics-compatible code.
	- Feed trained models into MASE for graph-level optimization/analysis/export.
	- This keeps the project “based on MASE” while avoiding high-risk detection-training refactors.


## System Architecture (High Level)

1. **Experiment Config Layer**
	 - YAML/TOML/JSON configs for teacher-student pairs, KD hyperparameters, runtime settings.

2. **KD Core Layer (new module)**
	 - Shared KD loss implementation:
		 - Hard-label loss (task CE)
		 - Soft-label distillation loss (KL divergence with temperature)
		 - Combined objective: `L = (1 - alpha) * L_hard + alpha * T^2 * L_soft`

3. **Task Pipelines**
	 - Vision pipeline (YOLO): baseline train → KD train → MASE optimization/eval.

4. **Evaluation & Reporting Layer**
	 - Unified metrics logger and artifact exporter (tables/plots/checkpoints).

## Metrics and Success Criteria

### Quality metrics

- **YOLO**: mAP@50 and mAP@50:95

### Efficiency metrics

- Peak GPU memory (GB)
- Training time per epoch
- Inference latency / throughput
- Model size / parameter count

### Optimization success definition

- KD student should significantly improve over non-KD student on quality.
- KD student should retain meaningful efficiency advantage over teacher.
- Results must include clear trade-off discussion (quality vs cost).

## One-Month Execution Plan

### Week 1 — Setup and Baselines

- Create project structure and KD module scaffold.
- Implement reproducible run setup (seed, logging, artifact folders).
- Run baseline (non-KD) training:
	- YOLO teacher + student baselines
- Verify GPU-fit configurations for 8–12GB.

**Deliverables**: baseline scripts run end-to-end; baseline metrics table v1.

### Week 2 — KD Implementation

- Implement logits KD core (temperature + alpha).
- Integrate KD into YOLO training flow.
- Add smoke tests for loss correctness and one-batch train step.

**Deliverables**: first KD training runs on YOLO; smoke tests passing.

### Week 3 — Experiments and Ablations

- Full baseline vs KD experiments for YOLO.
- Ablation studies:
	- `alpha` sweep
	- `temperature` sweep
- Run MASE-side optimization/evaluation on distilled checkpoints.

**Deliverables**: complete result tables, preliminary plots, ablation summary.

### Week 4 — Engineering Hardening and Report

- Improve test coverage for core KD modules.
- Finalize reproducibility documentation and run instructions.
- Create architecture diagram and final report figures.
- Write final 4-page report aligned with course rubric.

**Deliverables**: final report PDF, repository/PR link, test evidence, reproducible instructions.

## Resource Plan (8–12GB GPU)

### Vision (YOLO)

- Prefer small/medium YOLO teacher-student pair to stay in memory.
- Use mixed precision where stable.
- Tune image size and batch size conservatively.

## Testing and Engineering Standards

### Testing plan

- Unit tests:
	- KD loss value/shape checks
	- Temperature scaling and alpha mixing checks
- Integration smoke tests:
	- One mini YOLO training run
- Regression checks:
	- Ensure metrics/log artifacts are generated consistently

### Engineering quality plan

- Modular code structure and clear interfaces.
- Config-driven experiments (no hard-coded hyperparameters).
- README instructions for setup, training, evaluation, and reproduction.

## Risks and Mitigations

1. **Risk**: YOLO full native training integration into MASE is high effort.
	 - **Mitigation**: use Ultralytics-compatible training for KD, then apply MASE optimization/evaluation flow.

2. **Risk**: Memory overflow on 8GB runs.
	 - **Mitigation**: reduce batch/image size/sequence length; use fp16; use gradient accumulation.

3. **Risk**: One month is short for feature-rich KD variants.
	 - **Mitigation**: lock MVP to logits KD; treat feature KD as optional stretch.

4. **Risk**: Reproducibility drift.
	 - **Mitigation**: fixed seeds, fixed config snapshots, explicit environment notes.

## Final Deliverables Checklist

- Working automated KD workflow for YOLO.
- Quantitative baseline vs KD comparisons with trade-off analysis.
- Tests and engineering documentation.
- Architecture diagram and final 4-page report.
- Code submission via PR/repository URL.
