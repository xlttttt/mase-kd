import copy
import json
from pathlib import Path
from mase_kd.passes.export_pass import ExportMetricsPass
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from mase_kd.core.losses import DistillationLossConfig
from mase_kd.nlp.gpt2_imdb_kd import (
    GPT2IMDbKDConfig,
    GPT2StudentConfig,
    build_gpt2_imdb_kd_trainer,
)
from mase_kd.passes.prune_pass import PrunePass, PruneConfig
import yaml

def count_nonzero_params(model: nn.Module):
    nonzero = 0
    total = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            nonzero += int((param.data != 0).sum().item())
    return nonzero, total


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)


def make_cfg(
    output_dir: str,
    alpha: float,
    student_weights_path: str | None = None,
    train_subset: int | None = 512,
    test_subset: int | None = 256,
    epochs: int = 1,
):
    return GPT2IMDbKDConfig(
        teacher_model_name="mnoukhov/gpt2-imdb-sentiment-classifier",
        student=GPT2StudentConfig(
            model_name="distilgpt2",
            num_labels=2,
        ),
        kd=DistillationLossConfig(
            alpha=alpha,
            temperature=4.0,
        ),
        max_seq_length=256,
        batch_size=4,
        learning_rate=0.00002,
        num_epochs=epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        seed=42,
        train_subset=train_subset,
        test_subset=test_subset,
        output_dir=output_dir,
        student_weights_path=student_weights_path,
    )


def main():
    with open("experiments/configs/distilgpt2_imdb_pipeline_smoke.yaml", "r") as f:
        raw = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path("outputs/gpt2_imdb_pipeline_smoke/sparsity_0.50")
    base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # A: Dense
    # -------------------------
    a_dir = base_dir / "A_dense"
    cfg_a = make_cfg(str(a_dir), alpha=0.0)
    trainer_a = build_gpt2_imdb_kd_trainer(cfg_a, device=device)
    trainer_a.train()
    val_a = trainer_a.evaluate()
    nz_a, tot_a = count_nonzero_params(trainer_a.student)

    metrics_a = {
        **val_a,
        "accuracy": val_a["val_accuracy"],
        "params_nonzero": nz_a,
        "params_total": tot_a,
        "sparsity": 0.0,
    }
    save_metrics(metrics_a, a_dir / "metrics.json")

    # -------------------------
    # B: Pruned
    # -------------------------
    b_dir = base_dir / "B_pruned"
    b_dir.mkdir(parents=True, exist_ok=True)

    pruned_student = copy.deepcopy(trainer_a.student)
    prune_pass = PrunePass()
    pruned_student, prune_info = prune_pass.run(
        pruned_student,
        PruneConfig(
            sparsity=raw["pruning"]["sparsity"],
            target_types=(nn.Linear,Conv1D),
            make_permanent=True,
        ),
        {},
    )
    pruned_path = b_dir / "pruned_student"
    pruned_student.save_pretrained(str(pruned_path))

    cfg_b = make_cfg(str(b_dir), alpha=0.0, student_weights_path=str(pruned_path))
    trainer_b = build_gpt2_imdb_kd_trainer(cfg_b, device=device)
    val_b = trainer_b.evaluate()

    metrics_b = {
        **val_b,
        "accuracy": val_b["val_accuracy"],
        "params_nonzero": prune_info["params_nonzero"],
        "params_total": prune_info["params_total"],
        "sparsity": prune_info["sparsity_actual"],
    }
    save_metrics(metrics_b, b_dir / "metrics.json")

    del trainer_a
    del trainer_b
    torch.cuda.empty_cache()

    # -------------------------
    # C: Pruned + FT
    # -------------------------
    c_dir = base_dir / "C_ft"
    cfg_c = make_cfg(str(c_dir), alpha=0.0, student_weights_path=str(pruned_path))
    trainer_c = build_gpt2_imdb_kd_trainer(cfg_c, device=device)
    trainer_c.train()
    val_c = trainer_c.evaluate()
    nz_c, tot_c = count_nonzero_params(trainer_c.student)

    metrics_c = {
        **val_c,
        "accuracy": val_c["val_accuracy"],
        "params_nonzero": nz_c,
        "params_total": tot_c,
        "sparsity": 1.0 - nz_c / max(tot_c, 1),
    }
    save_metrics(metrics_c, c_dir / "metrics.json")

    del trainer_c
    torch.cuda.empty_cache()

    # -------------------------
    # D: Pruned + KD
    # -------------------------
    d_dir = base_dir / "D_kd"
    cfg_d = make_cfg(str(d_dir), alpha=0.5, student_weights_path=str(pruned_path))
    trainer_d = build_gpt2_imdb_kd_trainer(cfg_d, device=device)
    trainer_d.train()
    val_d = trainer_d.evaluate()
    nz_d, tot_d = count_nonzero_params(trainer_d.student)

    metrics_d = {
        **val_d,
        "accuracy": val_d["val_accuracy"],
        "params_nonzero": nz_d,
        "params_total": tot_d,
        "sparsity": 1.0 - nz_d / max(tot_d, 1),
    }
    save_metrics(metrics_d, d_dir / "metrics.json")

    del trainer_d
    torch.cuda.empty_cache()

    # -------------------------
    # E: Pruned + KD + FT
    # -------------------------
    e_dir = base_dir / "E_kd_ft"
    d_best = d_dir / "best_student"
    cfg_e = make_cfg(str(e_dir), alpha=0.0, student_weights_path=str(d_best))
    trainer_e = build_gpt2_imdb_kd_trainer(cfg_e, device=device)
    trainer_e.train()
    val_e = trainer_e.evaluate()
    nz_e, tot_e = count_nonzero_params(trainer_e.student)

    metrics_e = {
        **val_e,
        "accuracy": val_e["val_accuracy"],
        "params_nonzero": nz_e,
        "params_total": tot_e,
        "sparsity": 1.0 - nz_e / max(tot_e, 1),
    }
    save_metrics(metrics_e, e_dir / "metrics.json")

    results = {
        "A": metrics_a,
        "B": metrics_b,
        "C": metrics_c,
        "D": metrics_d,
        "E": metrics_e,
    }

    ExportMetricsPass().run(
        results,
        str(base_dir),
        "gpt2",
        "accuracy",
    )

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
