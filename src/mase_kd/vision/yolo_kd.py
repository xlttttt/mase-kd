"""Minimal YOLO logits distillation components."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from mase_kd.core.losses import (
    DistillationLossConfig,
    compute_distillation_loss,
    hard_label_ce_loss,
    soft_logit_kl_loss,
)


TaskLossFn = Callable[[Any, dict[str, Any]], torch.Tensor]


@dataclass(slots=True)
class YOLOLogitsKDOutput:
    """Scalar losses returned by one KD optimization step."""

    total_loss: float
    hard_loss: float
    soft_loss: float


class YOLOLogitsDistiller:
    """Minimal YOLO logits distillation trainer.

    Accepts a train_loader and optimizer at construction time so that the full
    training loop can be driven by a single ``train(steps)`` call.  An optional
    ``val_loader`` enables post-training evaluation via ``evaluate()``.

    Args:
        teacher: Frozen teacher model.
        student: Student model to be trained.
        kd_config: Distillation loss hyper-parameters.
        device: Device to run on.
        train_loader: DataLoader that yields ``(images, labels)`` pairs.
        optimizer: Optimizer pre-configured for the student parameters.
        num_train_epochs: Number of epochs to train when ``train()`` is called.
        val_loader: Optional DataLoader used by ``evaluate()`` to measure
            validation metrics.
        eval_teacher: When ``True``, ``evaluate()`` reports metrics for both
            the teacher and the student.  When ``False``, only the student is
            evaluated.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        kd_config: DistillationLossConfig,
        device: torch.device | str = "cpu",
        train_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        num_train_epochs: int = 1,
        val_loader: DataLoader | None = None,
        eval_teacher: bool = True,
    ) -> None:
        kd_config.validate()
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.kd_config = kd_config
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.num_train_epochs = num_train_epochs
        self.val_loader = val_loader
        self.eval_teacher = eval_teacher

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def _flatten_logits(self, output: Any) -> torch.Tensor:
        """Flatten nested tensor-like model outputs into a 2D logits tensor."""
        if isinstance(output, torch.Tensor):
            if output.ndim == 0:
                return output.reshape(1, 1)
            if output.shape[0] == 0:
                return output.reshape(1, 0)
            return output.reshape(output.shape[0], -1)

        if isinstance(output, dict):
            tensors = [
                self._flatten_logits(output[key])
                for key in sorted(output.keys())
                if isinstance(output[key], (torch.Tensor, dict, list, tuple))
            ]
            if not tensors:
                raise TypeError("Model output dict did not contain tensor-like values")
            return torch.cat(tensors, dim=1)

        if isinstance(output, (list, tuple)):
            if not output:
                raise TypeError("Model output list/tuple was empty")
            tensors = [self._flatten_logits(item) for item in output]

            first_batch = tensors[0].shape[0]
            if all(tensor.shape[0] == first_batch for tensor in tensors):
                return torch.cat(tensors, dim=1)

            if all(tensor.shape[0] == 1 for tensor in tensors):
                max_width = max(tensor.shape[1] for tensor in tensors)
                padded = []
                for tensor in tensors:
                    if tensor.shape[1] < max_width:
                        pad_width = max_width - tensor.shape[1]
                        tensor = torch.nn.functional.pad(tensor, (0, pad_width))
                    padded.append(tensor)
                return torch.cat(padded, dim=0)

            raise ValueError("Incompatible tensor batch dimensions in model output list")

        raise TypeError(f"Unsupported output type for logits distillation: {type(output)}")

    @staticmethod
    def _unwrap_classify_output(output: Any) -> Any:
        """Unwrap the ultralytics Classify head's eval-mode ``(softmax, logits)`` tuple.

        When the Classify head runs in eval mode it returns
        ``(softmax_probs, raw_logits)`` — two tensors with identical shapes.
        For distillation we only need the raw logits (the second element).

        If the output is not a 2-element tuple/list of same-shaped tensors the
        input is returned unchanged, so this is safe to call unconditionally.
        """
        if isinstance(output, (tuple, list)) and len(output) == 2:
            a, b = output
            if (
                isinstance(a, torch.Tensor)
                and isinstance(b, torch.Tensor)
                and a.shape == b.shape
            ):
                return b  # raw logits (second element by ultralytics convention)
        return output

    @staticmethod
    def _extract_logits_with_batch(output: Any, batch_size: int) -> torch.Tensor | None:
        """Recursively search nested model output for the first tensor whose leading
        dimension matches *batch_size*, then return it flattened to ``[batch_size, D]``.

        Returns ``None`` if no matching tensor is found.
        """
        if isinstance(output, torch.Tensor):
            if output.ndim >= 2 and output.shape[0] == batch_size:
                return output.reshape(batch_size, -1)
            if output.ndim >= 2 and output.shape[0] == 1 and batch_size > 1:
                return output.reshape(1, -1).expand(batch_size, -1)
            return None

        if isinstance(output, dict):
            for key in sorted(output.keys()):
                value = output[key]
                if isinstance(value, (torch.Tensor, dict, list, tuple)):
                    logits = YOLOLogitsDistiller._extract_logits_with_batch(value, batch_size)
                    if logits is not None:
                        return logits
            return None

        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, (torch.Tensor, dict, list, tuple)):
                    logits = YOLOLogitsDistiller._extract_logits_with_batch(item, batch_size)
                    if logits is not None:
                        return logits
            return None

        return None

    def _align_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify that student and teacher logits have compatible shapes.

        Raises ``ValueError`` on any mismatch so that silent data-corruption
        (e.g. from the Classify head returning different formats in train vs
        eval mode) is caught immediately.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Logits shape mismatch: student {tuple(student_logits.shape)} vs "
                f"teacher {tuple(teacher_logits.shape)}. "
                "This usually means the Classify head returned different formats "
                "in eval vs train mode — ensure _unwrap_classify_output() is "
                "applied to both outputs before flattening."
            )
        return student_logits, teacher_logits

    def train_step(
        self,
        batch: dict[str, Any],
        optimizer: torch.optim.Optimizer | None = None,
        task_loss_fn: TaskLossFn | None = None,
    ) -> YOLOLogitsKDOutput:
        """Run one optimization step with logits distillation.

        If *optimizer* is ``None``, ``self.optimizer`` (set at construction) is used.
        """
        optimizer = optimizer if optimizer is not None else self.optimizer
        if optimizer is None:
            raise ValueError(
                "No optimizer provided: pass one to train_step() or supply it to "
                "YOLOLogitsDistiller() via the `optimizer` argument."
            )
        images = batch["images"].to(self.device)
        targets = batch.get("targets")
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)

        self.student.train()
        optimizer.zero_grad(set_to_none=True)

        student_output = self.student(images)

        with torch.no_grad():
            teacher_output = self.teacher(images)

        student_output = self._unwrap_classify_output(student_output)
        teacher_output = self._unwrap_classify_output(teacher_output)

        student_logits = self._flatten_logits(student_output)
        teacher_logits = self._flatten_logits(teacher_output)
        student_logits, teacher_logits = self._align_logits(student_logits, teacher_logits)

        if task_loss_fn is not None:
            hard_loss = task_loss_fn(student_output, batch)
            total, _, soft_loss = compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=None,
                config=self.kd_config,
            )
            total = (1.0 - self.kd_config.alpha) * hard_loss + self.kd_config.alpha * soft_loss
        else:
            total, hard_loss, soft_loss = compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                config=self.kd_config,
            )

        total.backward()
        optimizer.step()

        return YOLOLogitsKDOutput(
            total_loss=float(total.detach().cpu().item()),
            hard_loss=float(hard_loss.detach().cpu().item()),
            soft_loss=float(soft_loss.detach().cpu().item()),
        )

    def train(
        self,
        log_every: int = 10,
        task_loss_fn: TaskLossFn | None = None,
    ) -> list[float]:
        """Run the full KD training loop for ``self.num_train_epochs`` epochs.

        Requires ``self.train_loader`` and ``self.optimizer`` to be set at
        construction time.

        Args:
            log_every: Print a progress line every this many batches within
                each epoch. Set to 0 to suppress per-batch output.
            task_loss_fn: Optional external task-loss function forwarded to
                each ``train_step`` call.

        Returns:
            A list of ``total_loss`` values, one per batch across all epochs.
        """
        if self.train_loader is None:
            raise ValueError(
                "train_loader is required for train(): supply it to "
                "YOLOLogitsDistiller() via the `train_loader` argument."
            )
        if self.optimizer is None:
            raise ValueError(
                "optimizer is required for train(): supply it to "
                "YOLOLogitsDistiller() via the `optimizer` argument."
            )

        loss_history: list[float] = []

        for epoch in range(1, self.num_train_epochs + 1):
            num_batches = len(self.train_loader)
            print(f"Epoch {epoch}/{self.num_train_epochs}")
            for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
                batch = {
                    "images": images.to(self.device),
                    "targets": labels.to(self.device),
                }

                output = self.train_step(
                    batch=batch,
                    task_loss_fn=task_loss_fn,
                )
                loss_history.append(output.total_loss)

                if log_every > 0 and (
                    batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches
                ):
                    print(
                        f"  Batch {batch_idx:04d}/{num_batches} | "
                        f"total={output.total_loss:.6f} | "
                        f"hard={output.hard_loss:.6f} | soft={output.soft_loss:.6f}"
                    )

        return loss_history


    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Evaluate teacher (when ``eval_teacher=True``) and student on ``val_loader``.

        Computes top-1 accuracy, average cross-entropy loss, and average
        forward time per batch for each model, plus the validation KD
        (KL-divergence) loss between teacher and student.

        Returns:
            A dict with keys:

            * ``"student"`` – metrics dict for the distilled student.
            * ``"teacher"`` – metrics dict for the teacher (only when
              ``eval_teacher=True``).
            * ``"val_kd_loss"`` – average KD loss over the validation set.
            * ``"kd_batches"`` – number of batches used for the KD loss.

            Each per-model metrics dict contains ``top1_acc``,
            ``avg_ce_loss``, ``avg_forward_ms_per_batch``, ``samples``,
            and ``batches``.

        Raises:
            ValueError: If ``val_loader`` was not supplied at construction.
        """
        import time

        if self.val_loader is None:
            raise ValueError(
                "val_loader is required for evaluate(): supply it to "
                "YOLOLogitsDistiller() via the `val_loader` argument."
            )

        def _eval_model(model: nn.Module) -> dict[str, Any]:
            model.eval()
            batches = 0
            samples = 0
            total_forward_ms = 0.0
            correct_top1 = 0
            total_ce_loss = 0.0

            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = model(images)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                logits = self._extract_logits_with_batch(outputs, images.shape[0])
                if logits is None or logits.numel() == 0:
                    continue

                total_forward_ms += (t1 - t0) * 1000.0
                batches += 1
                samples += images.shape[0]

                max_label = int(labels.max().item())
                if logits.shape[1] > max_label:
                    preds = logits.argmax(dim=1)
                    correct_top1 += int((preds == labels).sum().item())
                    total_ce_loss += hard_label_ce_loss(logits, labels).item()

            return {
                "batches": batches,
                "samples": samples,
                "avg_forward_ms_per_batch": total_forward_ms / max(batches, 1),
                "top1_acc": correct_top1 / max(samples, 1),
                "avg_ce_loss": total_ce_loss / max(batches, 1),
            }

        results: dict[str, Any] = {}
        if self.eval_teacher:
            results["teacher"] = _eval_model(self.teacher)
        results["student"] = _eval_model(self.student)

        # KD (KL-divergence) loss: teacher vs distilled student
        self.teacher.eval()
        self.student.eval()
        total_kd_loss = 0.0
        used_batches = 0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            teacher_outputs = self._unwrap_classify_output(self.teacher(images))
            student_outputs = self._unwrap_classify_output(self.student(images))

            teacher_logits = self._extract_logits_with_batch(teacher_outputs, labels.shape[0])
            student_logits = self._extract_logits_with_batch(student_outputs, labels.shape[0])
            if teacher_logits is None or student_logits is None:
                continue

            try:
                student_logits, teacher_logits = self._align_logits(student_logits, teacher_logits)
            except ValueError:
                continue

            total_kd_loss += soft_logit_kl_loss(
                student_logits, teacher_logits, self.kd_config.temperature
            ).item()
            used_batches += 1

        results["val_kd_loss"] = total_kd_loss / max(used_batches, 1)
        results["kd_batches"] = used_batches
        return results


def build_mase_yolo_detection_model(checkpoint: str) -> nn.Module:
    """Build a MASE YOLO detection model from a YOLO checkpoint path."""
    from chop.models.yolo import get_yolo_detection_model

    return get_yolo_detection_model(checkpoint)
