# by425 — Testing Notes

## Purpose

The GPT-2 IMDb line is tested at multiple levels so that the contribution satisfies the project requirement for thorough testing and clear engineering practice. The requirement explicitly emphasises detailed READMEs, testing, and documentation, and the report must describe the testing approach and results. 

## Test Layers

### 1. Smoke KD test
File:
- `cw/by425/tests/test_gpt2_imdb_kd_smoke.py`

Purpose:
- verify that the minimum GPT-2 KD path runs end-to-end
- check teacher/student loading
- confirm IMDb subset sampling works
- catch tokenizer / teacher mismatch issues early

Why it matters:
During development, the workflow exposed a concrete failure mode where GPT-2 tokenisation was paired with a BERT teacher, causing invalid teacher inputs. This smoke test exists to catch that class of integration problem immediately. :contentReference[oaicite:10]{index=10}

### 2. Full KD test
File:
- `cw/by425/tests/test_gpt2_imdb_kd_full.py`

Purpose:
- run a larger KD experiment
- validate that the full training/evaluation path works on the intended teacher/student pair
- produce report-usable headline metrics

Observed result:
- val_loss: `0.329678921431303`
- validation accuracy: `0.8870`
- validation macro-F1: `0.8869981919710715`
### 3. A–E smoke pipeline test
File:
- `cw/by425/tests/test_gpt2_imdb_pipeline_smoke.py`

Purpose:
- verify the full pruning-recovery workflow:
  - A Dense
  - B Pruned
  - C Pruned+FT
  - D Pruned+KD
  - E Pruned+KD+FT
- confirm that pruning actually affects GPT-2
- check that later recovery stages improve over the pruned baseline

Why it matters:
The GPT-2 line was not just a standalone KD script; it had to fit the same automated tool-flow logic as the rest of the project, since the coursework requires an integrated optimisation pipeline rather than isolated scripts. :contentReference[oaicite:11]{index=11}

Observed smoke results:
- A Dense: `0.6406`
- B Pruned: `0.6133`
- C Pruned+FT: `0.7148`
- D Pruned+KD: `0.7227`
- E Pruned+KD+FT: `0.7539`

### 4. Grid-search smoke test
File:
- `cw/by425/tests/test_gpt2_imdb_grid_search_smoke.py`

Purpose:
- verify that the grid-search script runs
- confirm output files are emitted
- check `results.json`, `results.csv`, and `best.json` creation

Why it matters:
The workflow notes explicitly identified a small KD hyperparameter sweep as the most natural next step for stronger trade-off analysis. :contentReference[oaicite:12]{index=12}

Observed best result:
- alpha: `0.5`
- temperature: `2.0`
- val_loss: `0.3241055379137397`
- val_accuracy: `0.8915`
- val_f1: `0.8914999728749933`
### 5. Best-config smoke test
File:
- `cw/by425/tests/test_gpt2_imdb_kd_best.py`

Purpose:
- verify that the selected best config can be rerun independently
- confirm `metrics.json` is emitted
- make the final best-KD path reproducible

Observed rerun result:
- alpha: `0.5`
- temperature: `2.0`
- val_loss: `0.32410557360202075`
- val_accuracy: `0.8915`
- val_f1: `0.8914999728749933`
## Engineering Checks

### Teacher model validity
An earlier GPT-2 teacher candidate (`lvwerra/distilgpt2-imdb`) failed because it was not publicly accessible/valid in the final workflow. The line was switched to `mnoukhov/gpt2-imdb-sentiment-classifier`. 

### Pruning validity
GPT-2 pruning is only meaningful when both `nn.Linear` and `Conv1D` are included. The A–E smoke pipeline is used to verify that pruning now produces a measurable drop from A to B before recovery.

### Sparsity preservation
Later stages are intended to start from the same sparse structure. This follows the same project-wide fairness principle used elsewhere in the report: later recovery stages must be compared on a consistent sparse checkpoint. :contentReference[oaicite:14]{index=14}

## Output Artifacts

The following output artifacts are expected from the GPT-2 testing/evaluation line:
- `results.json`
- `results.csv`
- `best.json`
- `metrics.json`

These are used both for reproducibility and for report writing.

## Summary

The GPT-2 contribution is tested at:
- unit/smoke level
- full-run level
- multi-stage pipeline level
- hyperparameter-sweep output level
- final-best-config reproducibility level

This testing structure is designed to satisfy both the engineering and documentation expectations in the team requirement. 
