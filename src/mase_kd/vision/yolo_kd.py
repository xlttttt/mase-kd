"""Minimal YOLO logits distillation components."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from mase_kd.core.losses import DistillationLossConfig, compute_distillation_loss


TaskLossFn = Callable[[Any, dict[str, Any]], torch.Tensor]


@dataclass(slots=True)
class YOLOLogitsKDOutput:
    """Scalar losses returned by one KD optimization step."""

    total_loss: float
    hard_loss: float
    soft_loss: float


class YOLOLogitsDistiller:
    """Minimal YOLO logits distillation trainer.

    This first version focuses on soft-logit KD and optionally accepts
    an external task loss function for the hard-label component.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        kd_config: DistillationLossConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        kd_config.validate()
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.kd_config = kd_config
        self.device = torch.device(device)

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
        """Align student/teacher feature dimensions by truncating to shared width."""
        if student_logits.shape[0] != teacher_logits.shape[0]:
            if student_logits.shape[0] == 0 or teacher_logits.shape[0] == 0:
                raise ValueError("Student/teacher logits have empty batch dimension")

            student_logits = student_logits.mean(dim=0, keepdim=True)
            teacher_logits = teacher_logits.mean(dim=0, keepdim=True)

        if student_logits.shape[1] == teacher_logits.shape[1]:
            return student_logits, teacher_logits

        dim = min(student_logits.shape[1], teacher_logits.shape[1])
        if dim == 0:
            raise ValueError(
                "No shared logits dimension between student and teacher outputs. "
                "Ensure teacher output is pre-NMS (training-mode forward)."
            )
        return student_logits[:, :dim], teacher_logits[:, :dim]

    def train_step(
        self,
        batch: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        task_loss_fn: TaskLossFn | None = None,
    ) -> YOLOLogitsKDOutput:
        """Run one optimization step with logits distillation."""
        images = batch["images"].to(self.device)
        targets = batch.get("targets")
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)

        self.student.train()
        optimizer.zero_grad(set_to_none=True)

        student_output = self.student(images)

        teacher_previous_mode = self.teacher.training
        self.teacher.train()
        with torch.no_grad():
            teacher_output = self.teacher(images)
        self.teacher.train(teacher_previous_mode)

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


def build_mase_yolo_detection_model(checkpoint: str) -> nn.Module:
    """Build a MASE YOLO detection model from a YOLO checkpoint path."""
    from chop.models.yolo import get_yolo_detection_model

    return get_yolo_detection_model(checkpoint)
