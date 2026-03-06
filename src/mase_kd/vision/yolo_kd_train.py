"""YOLO knowledge distillation training loop.

Architecture
------------
* Teacher: pretrained YOLOv8 model (e.g. yolov8m.pt) — frozen, eval mode.
* Student: YOLOv8 model of a smaller variant (e.g. yolov8n) — trained from the
  teacher's checkpoint or from scratch.

Loss
----
    L_total = (1 - alpha) * L_task + alpha * L_kd

where L_task is the standard Ultralytics v8DetectionLoss (classification +
bounding-box regression + DFL) and L_kd is temperature-scaled KL divergence
between flattened student and teacher raw (pre-NMS) feature maps.

Data loading and evaluation reuse Ultralytics utilities so that standard COCO-
format datasets (described by a data YAML) work out of the box.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from mase_kd.core.losses import DistillationLossConfig, soft_logit_kl_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class YOLOTrainingConfig:
    """Hyper-parameters for one YOLO KD training run."""

    # Model checkpoints / architectures
    teacher_weights: str = "yolov8m.pt"
    # Student arch YAML (no pretrained weights → train from scratch using KD)
    student_arch: str = "yolov8n.yaml"
    # If provided, initialise student from these weights (fine-tuning scenario)
    student_weights: Optional[str] = None

    # Dataset
    data_yaml: str = "coco8.yaml"  # Replace with coco.yaml for full COCO

    # KD
    kd: DistillationLossConfig = None  # set in __post_init__

    # Training
    epochs: int = 50
    batch_size: int = 16
    imgsz: int = 640
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    use_amp: bool = True
    workers: int = 8
    seed: int = 42

    # I/O
    output_dir: str = "outputs/yolo_kd"

    def __post_init__(self):
        if self.kd is None:
            self.kd = DistillationLossConfig()

    def validate(self) -> None:
        self.kd.validate()
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_yolo_model(weights_or_arch: str) -> nn.Module:
    """Load a MASE YOLOv8 detection model.

    Accepts either a .pt checkpoint (e.g. 'yolov8m.pt') or an arch YAML
    (e.g. 'yolov8n.yaml').  The .pt path is tried first via chop's wrapper
    which also handles the C2f monkey-patch.
    """
    from chop.models.yolo.yolov8 import get_yolo_detection_model

    return get_yolo_detection_model(weights_or_arch)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _build_dataloader(
    data_yaml: str,
    batch_size: int,
    imgsz: int,
    workers: int,
    mode: str = "train",
):
    """Return an Ultralytics-compatible YOLO dataloader.

    Args:
        data_yaml: Path to dataset config, e.g. 'coco8.yaml'.
        batch_size: Batch size.
        imgsz: Input image size (square).
        workers: DataLoader worker processes.
        mode: 'train' or 'val'.
    """
    from ultralytics.cfg import get_cfg, DEFAULT_CFG
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    args = get_cfg(DEFAULT_CFG)
    args.data = data_yaml
    args.imgsz = imgsz
    args.batch = batch_size
    args.workers = workers
    args.rect = mode == "val"  # rectangular batches for validation

    data_dict = check_det_dataset(data_yaml)
    img_path = data_dict[mode]

    stride = 32  # YOLOv8 max stride
    dataset = build_yolo_dataset(
        args,
        img_path=img_path,
        batch=batch_size,
        data=data_dict,
        mode=mode,
        rect=args.rect,
        stride=stride,
    )
    return build_dataloader(
        dataset,
        batch=batch_size,
        workers=workers,
        shuffle=(mode == "train"),
        rank=-1,
    )


# ---------------------------------------------------------------------------
# KD loss helpers
# ---------------------------------------------------------------------------


def _flatten_detection_preds(preds: Any) -> torch.Tensor:
    """Recursively flatten nested YOLO detection head outputs to [B, D]."""
    if isinstance(preds, torch.Tensor):
        b = preds.shape[0]
        return preds.reshape(b, -1)

    if isinstance(preds, (list, tuple)):
        parts = [_flatten_detection_preds(p) for p in preds]
        # Concatenate along feature dimension
        b = parts[0].shape[0]
        if all(p.shape[0] == b for p in parts):
            return torch.cat(parts, dim=1)
        # Fallback: average-pool batch dim to 1 then cat
        parts = [p.mean(0, keepdim=True) for p in parts]
        max_w = max(p.shape[1] for p in parts)
        padded = [
            F.pad(p, (0, max_w - p.shape[1])) if p.shape[1] < max_w else p
            for p in parts
        ]
        return torch.cat(padded, dim=0)

    raise TypeError(f"Unsupported pred type: {type(preds)}")


def _compute_kd_soft_loss(
    student_preds: Any,
    teacher_preds: Any,
    temperature: float,
) -> torch.Tensor:
    """Flatten student/teacher preds, align dims, compute soft KL loss."""
    s_flat = _flatten_detection_preds(student_preds)
    t_flat = _flatten_detection_preds(teacher_preds)

    # Align batch dimension
    if s_flat.shape[0] != t_flat.shape[0]:
        s_flat = s_flat.mean(0, keepdim=True)
        t_flat = t_flat.mean(0, keepdim=True)

    # Align feature dimension (truncate to shared width)
    dim = min(s_flat.shape[1], t_flat.shape[1])
    if dim == 0:
        return torch.tensor(0.0, device=s_flat.device)

    return soft_logit_kl_loss(s_flat[:, :dim], t_flat[:, :dim], temperature)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class YOLOKDRunner:
    """Standalone YOLO knowledge distillation training runner.

    Combines:
      - Ultralytics v8DetectionLoss for the hard detection task loss.
      - Soft KL divergence on raw (pre-NMS) head outputs for knowledge transfer.
    """

    def __init__(self, config: YOLOTrainingConfig) -> None:
        config.validate()
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)

        # ------------------------------------------------------------------
        # Load teacher
        # ------------------------------------------------------------------
        logger.info("Loading teacher: %s", config.teacher_weights)
        self.teacher = _load_yolo_model(config.teacher_weights).to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------
        # Load student
        # ------------------------------------------------------------------
        student_src = config.student_weights or config.student_arch
        logger.info("Loading student: %s", student_src)
        self.student = _load_yolo_model(student_src).to(self.device)

        # ------------------------------------------------------------------
        # Detection task loss (hard-label, uses GT boxes)
        # ------------------------------------------------------------------
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.cfg import get_cfg, DEFAULT_CFG

        # MaseYoloDetectionModel does not carry .args (Ultralytics trainer
        # normally injects this); inject default hyper-parameters so that
        # v8DetectionLoss can read box/cls/dfl gain values.
        if not hasattr(self.student, "args"):
            self.student.args = get_cfg(DEFAULT_CFG)

        self.task_criterion = v8DetectionLoss(self.student)

        # ------------------------------------------------------------------
        # Optimizer (cosine-LR + linear warmup handled manually)
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.use_amp)

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, dataloader, epoch: int) -> dict[str, float]:
        self.student.train()
        self.teacher.eval()

        total_loss = task_loss_sum = kd_loss_sum = 0.0
        n_batches = 0

        # Cosine annealing + warmup for LR
        warmup_iters = int(self.cfg.warmup_epochs * len(dataloader))
        start_iter = (epoch - 1) * len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            global_iter = start_iter + batch_idx

            # Warmup linear LR ramp
            if global_iter < warmup_iters:
                lr_scale = (global_iter + 1) / warmup_iters
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.cfg.learning_rate * lr_scale

            # Move batch to device and normalise pixel values to [0, 1]
            imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch["img"] = imgs  # overwrite with normalised version

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.use_amp):
                # Student forward (training mode → raw detection head outputs)
                student_preds = self.student(imgs)

                # Hard task loss (GT bounding boxes + classes).
                # v8DetectionLoss returns a 3-element vector [box, cls, dfl];
                # sum to a scalar before combining with KD loss.
                task_loss_vec, _ = self.task_criterion(student_preds, batch)
                task_loss = task_loss_vec.sum()

                # Teacher forward — training mode to get raw logits (no NMS)
                with torch.no_grad():
                    self.teacher.train()
                    teacher_preds = self.teacher(imgs)
                    self.teacher.eval()

                # Soft KD loss
                kd_loss = _compute_kd_soft_loss(
                    student_preds, teacher_preds, self.cfg.kd.temperature
                )

                alpha = self.cfg.kd.alpha
                combined = (1.0 - alpha) * task_loss + alpha * kd_loss

            self.scaler.scale(combined).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += combined.item()
            task_loss_sum += task_loss.item()
            kd_loss_sum += kd_loss.item()
            n_batches += 1

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "task_loss": task_loss_sum / max(n_batches, 1),
            "kd_loss": kd_loss_sum / max(n_batches, 1),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Run Ultralytics mAP validation on the student model."""
        from ultralytics import YOLO as UltralyticsYOLO

        # Save current student weights to a temp file, then validate via YOLO
        tmp_path = self.out_dir / "_tmp_student.pt"
        torch.save({"model": self.student.state_dict()}, tmp_path)

        try:
            yolo = UltralyticsYOLO(self.cfg.student_arch)
            yolo.model.load_state_dict(self.student.state_dict())
            results = yolo.val(
                data=self.cfg.data_yaml,
                imgsz=self.cfg.imgsz,
                device=self.device.index if self.device.type == "cuda" else "cpu",
                verbose=False,
            )
            map50 = float(results.box.map50)
            map50_95 = float(results.box.map)
        except Exception as exc:
            logger.warning("Ultralytics val failed (%s); returning zeros.", exc)
            map50 = map50_95 = 0.0
        finally:
            tmp_path.unlink(missing_ok=True)

        return {"mAP50": map50, "mAP50_95": map50_95}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Run all epochs, save best checkpoint, return training history."""
        logger.info(
            "Starting YOLO KD training | alpha=%.2f | T=%.1f | epochs=%d",
            self.cfg.kd.alpha,
            self.cfg.kd.temperature,
            self.cfg.epochs,
        )

        train_loader = _build_dataloader(
            self.cfg.data_yaml,
            self.cfg.batch_size,
            self.cfg.imgsz,
            self.cfg.workers,
            mode="train",
        )

        best_map50 = 0.0

        # Cosine LR schedule (post-warmup)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs
        )

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self.evaluate()
            elapsed = time.perf_counter() - t0

            scheduler.step()

            epoch_log = {
                "epoch": epoch,
                "time_s": round(elapsed, 1),
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_log)

            logger.info(
                "Epoch %d/%d | loss=%.4f (task=%.4f kd=%.4f) | mAP50=%.4f | %.1fs",
                epoch,
                self.cfg.epochs,
                train_metrics["train_loss"],
                train_metrics["task_loss"],
                train_metrics["kd_loss"],
                val_metrics["mAP50"],
                elapsed,
            )

            if val_metrics["mAP50"] > best_map50:
                best_map50 = val_metrics["mAP50"]
                self._save_student("best_student.pt")
                logger.info("  => New best mAP50: %.4f — checkpoint saved.", best_map50)

        self._save_student("last_student.pt")
        self._save_history()
        logger.info("Training complete. Best mAP50: %.4f", best_map50)
        return self.history

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _save_student(self, filename: str) -> None:
        path = self.out_dir / filename
        torch.save(
            {
                "model": self.student.state_dict(),
                "arch": self.cfg.student_arch,
                "kd_config": {
                    "alpha": self.cfg.kd.alpha,
                    "temperature": self.cfg.kd.temperature,
                },
            },
            path,
        )

    def _save_history(self) -> None:
        path = self.out_dir / "training_history.json"
        with path.open("w") as fp:
            json.dump(self.history, fp, indent=2)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_yolo_kd_runner(config: YOLOTrainingConfig) -> YOLOKDRunner:
    """Validate config and return an initialised YOLOKDRunner."""
    return YOLOKDRunner(config)
