# by425 — Experiment Log

## Scope

This log records the main development and evaluation milestones for the GPT-2 IMDb KD contribution.

## 1. Teacher-model debugging

### Initial issue
Early GPT-2 smoke attempts failed because the teacher model choice was incorrect. The workflow log records two related problems:
1. `lvwerra/distilgpt2-imdb` was not a valid/publicly accessible model ID in the final setup.
2. Mixing GPT-2 tokenisation with a BERT IMDb teacher caused a runtime failure because GPT-2 token IDs were fed into a BERT vocabulary. 

### Fix
The final teacher was changed to:

- `mnoukhov/gpt2-imdb-sentiment-classifier`

This aligned the teacher/student family and removed the tokenizer mismatch problem. 

## 2. GPT-2 KD implementation

Main file:
- `src/mase_kd/nlp/gpt2_imdb_kd.py`

Key capabilities added:
- logits-only KD
- `student_weights_path`
- zero-mask preservation
- IMDb random subset support
- GPT-2 teacher/student compatibility

This matches the GPT-2 workflow described in the NLP process document and supports both smoke and report-scale experiments. :contentReference[oaicite:18]{index=18}

## 3. GPT-2 A–E smoke pipeline

Student:
- `distilgpt2`

Teacher:
- `mnoukhov/gpt2-imdb-sentiment-classifier`

Pipeline stages:
- A Dense
- B Pruned
- C Pruned+FT
- D Pruned+KD
- E Pruned+KD+FT

Important implementation note:
- pruning target was expanded to include both `nn.Linear` and `Conv1D` so pruning would actually affect GPT-2

Observed smoke results:
- A Dense: `0.6406`
- B Pruned: `0.6133`
- C Pruned+FT: `0.7148`
- D Pruned+KD: `0.7227`
- E Pruned+KD+FT: `0.7539`

Interpretation:
- pruning causes a measurable drop from A to B
- both FT and KD recover performance
- D beats C
- E is the best smoke-stage variant

This result pattern was used directly in the GPT-2 report section and in `gpt2_ae_smoke.png`.

## 4. Full KD run

Config:
- `experiments/configs/distilgpt2_imdb_kd_full.yaml`

Observed result:
- `val_loss = 0.329678921431303`
- `val_accuracy = 0.8870`
- `val_f1 = 0.8869981919710715`

Interpretation:
- the larger run is much stronger than the smoke A–E figures
- the GPT-2 KD line is report-worthy as a standalone NLP case study

## 5. KD hyperparameter grid search

Motivation:
The requirement expects principled results and trade-off analysis, and the workflow log explicitly identified small KD sweeps as a useful next step rather than only reporting a single point result. 

Sweep:
- `alpha ∈ {0.3, 0.5, 0.7}`
- `temperature ∈ {2.0, 4.0, 6.0}`

Files:
- `experiments/configs/distilgpt2_imdb_grid_search.yaml`
- `experiments/scripts/run_gpt2_imdb_grid_search.py`

Saved outputs:
- `results.json`
- `results.csv`
- `best.json`

Best setting:
- `alpha = 0.5`
- `temperature = 2.0`
- `val_loss = 0.3241055379137397`
- `val_accuracy = 0.8915`
- `val_f1 = 0.8914999728749933`

Interpretation:
- moderate temperature performed best in this decoder-only setup
- balanced KD/CE weighting was stronger than more extreme settings

## 6. Best-config rerun

Files:
- `experiments/configs/distilgpt2_imdb_kd_best.yaml`
- `experiments/scripts/run_gpt2_imdb_kd_best.py`

Purpose:
- isolate the selected best KD setting
- rerun it independently
- save final metrics to `metrics.json`

Observed rerun result:
- `alpha = 0.5`
- `temperature = 2.0`
- `val_loss = 0.32410557360202075`
- `val_accuracy = 0.8915`
- `val_f1 = 0.8914999728749933`

Interpretation:
- the independent rerun reproduces the grid-search best setting
- this confirms that the selected best KD configuration is stable and report-ready
This makes the final report result reproducible from a single config/script pair.

## 7. Documentation status

Completed for the GPT-2 line:
- source code
- smoke tests
- full KD test
- A–E smoke pipeline
- grid-search script
- best-config script
- README
- testing notes
- experiment log

Still to ensure in the final submission:
- final PDF report includes the GPT-2 section
- the whole report includes at least one system architecture diagram, as required by the coursework brief. 
