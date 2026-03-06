"""ResNet18/CIFAR-10 knowledge distillation trainer.

Architecture
------------
* Teacher: a pre-trained ResNet (any variant) loaded from a checkpoint path.
  For the A-E pipeline the dense ResNet18 trained in Step A is reused as
  teacher for Steps D and E, so no external teacher download is required.
  Pass ``teacher=None`` when alpha=0 (baseline / fine-tune only) to skip the
  teacher forward pass entirely.
* Student: ResNet18 with a CIFAR-10-friendly first convolution
  (3×3, stride 1, no max-pool) so that 32×32 inputs are not down-sampled too
  aggressively.

Loss
----
    L = (1 - alpha) * L_hard + alpha * T^2 * L_soft

with L_hard = cross-entropy(student, labels) and L_soft = KL-divergence
(temperature-scaled) between student and teacher logits.  When alpha=0 the
teacher is ignored and the loss reduces to plain cross-entropy.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from mase_kd.core.losses import DistillationLossConfig, compute_distillation_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ResNetKDConfig:
    """All hyperparameters for one ResNet KD experiment on CIFAR-10/100."""

    # Teacher
    teacher_weights: str = ""      # path to teacher .pth; empty → no teacher
    teacher_arch: str = "resnet18" # "resnet18" or "resnet34"

    # Student
    student_weights: Optional[str] = None  # path to initial student weights
    num_classes: int = 10

    # Dataset
    dataset: str = "cifar10"  # "cifar10" | "cifar100"

    # KD hyperparameters
    kd: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    # Training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "cosine"  # "cosine" or "step"
    seed: int = 42

    # Data
    data_dir: str = "datasets/cifar10"
    val_split: float = 0.1
    subset_size: Optional[int] = None  # None = full dataset; int = smoke subset

    # I/O
    output_dir: str = "outputs/resnet_kd"

    def validate(self) -> None:
        self.kd.validate()
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError("val_split must be in (0, 1)")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_cifar_resnet18(
    num_classes: int = 10,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Build a ResNet18 tuned for CIFAR-10 32×32 inputs.

    Changes from the ImageNet variant:
    - First conv: 7×7 stride-2 → 3×3 stride-1 (keeps 32×32 spatial size)
    - Max-pool replaced by Identity (avoids collapsing 32×32 to 8×8)
    """
    import torchvision.models as models

    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Accept raw state dicts or {"model_state_dict": ...} / {"model": ...} wrappers
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model" in state:
                state = state["model"]
        model.load_state_dict(state)
        logger.info("Loaded ResNet18 weights from %s", weights_path)

    return model


def build_cifar_resnet34(
    num_classes: int = 10,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Build a ResNet34 tuned for CIFAR-10 32×32 inputs (same first-conv patch)."""
    import torchvision.models as models

    model = models.resnet34(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model" in state:
                state = state["model"]
        model.load_state_dict(state)
        logger.info("Loaded ResNet34 weights from %s", weights_path)

    return model


def _build_model_by_arch(arch: str, num_classes: int, weights_path: Optional[str] = None) -> nn.Module:
    if arch == "resnet18":
        return build_cifar_resnet18(num_classes, weights_path)
    elif arch == "resnet34":
        return build_cifar_resnet34(num_classes, weights_path)
    else:
        raise ValueError(f"Unsupported teacher_arch: {arch!r}. Choose 'resnet18' or 'resnet34'.")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_cifar10_dataloaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-10.

    Args:
        data_dir:    Root directory for CIFAR-10 (downloaded automatically if absent).
        batch_size:  Batch size for train and test loaders.
        val_split:   Fraction of the training set held out for validation.
        subset_size: If set, sub-sample this many training samples (for smoke tests).
        num_workers: DataLoader worker threads.
        seed:        RNG seed for reproducible val split.
    """
    from torchvision import datasets, transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Optional sub-sampling for smoke tests
    if subset_size is not None:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=rng)[:subset_size]
        full_train = Subset(full_train, indices.tolist())

    # Val split
    n_total = len(full_train)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    rng = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=rng)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def load_cifar100_dataloaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-100.

    Args:
        data_dir:    Root directory for CIFAR-100 (downloaded automatically if absent).
        batch_size:  Batch size for train and test loaders.
        val_split:   Fraction of the training set held out for validation.
        subset_size: If set, sub-sample this many training samples (for smoke tests).
        num_workers: DataLoader worker threads.
        seed:        RNG seed for reproducible val split.
    """
    from torchvision import datasets, transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    full_train = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Optional sub-sampling for smoke tests
    if subset_size is not None:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=rng)[:subset_size]
        full_train = Subset(full_train, indices.tolist())

    # Val split
    n_total = len(full_train)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    rng = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=rng)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ResNetKDTrainer:
    """Knowledge distillation trainer for ResNet18 on CIFAR-10.

    Training objective:
        L = (1 - alpha) * L_hard + alpha * T^2 * L_soft

    Pass ``teacher=None`` to disable KD (pure cross-entropy training).
    """

    def __init__(
        self,
        config: ResNetKDConfig,
        teacher: Optional[nn.Module],
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
        config.validate()
        self.config = config
        self.device = device

        # Teacher: frozen, eval mode (None is allowed when alpha=0)
        if teacher is not None:
            self.teacher = teacher.to(device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            self.teacher = None

        # Student: trainable
        self.student = student.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # SGD with momentum
        self.optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )

        if config.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        else:
            # Step LR: decay by 0.1 at 50%, 75% of training
            milestones = [int(config.epochs * 0.5), int(config.epochs * 0.75)]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=0.1
            )

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> dict[str, float]:
        self.student.train()
        total_loss = hard_loss_sum = soft_loss_sum = 0.0
        n_batches = 0

        use_kd = self.teacher is not None and self.config.kd.alpha > 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            student_logits = self.student(images)

            if use_kd:
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                loss, hard, soft = compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    targets=labels,
                    config=self.config.kd,
                )
            else:
                hard = F.cross_entropy(student_logits, labels)
                loss = hard
                soft = torch.zeros_like(hard)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            hard_loss_sum += hard.item()
            soft_loss_sum += soft.item()
            n_batches += 1

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_hard_loss": hard_loss_sum / max(n_batches, 1),
            "train_soft_loss": soft_loss_sum / max(n_batches, 1),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader) -> dict[str, float]:
        self.student.eval()
        correct = total = 0
        loss_sum = 0.0
        n_batches = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.student(images)
            loss_sum += F.cross_entropy(logits, labels).item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            n_batches += 1

        return {
            "loss": loss_sum / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
        }

    def evaluate(self, split: str = "val") -> dict[str, float]:
        """Evaluate on val or test split."""
        loader = self.val_loader if split == "val" else self.test_loader
        return self._eval_loader(loader)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Run all epochs, save best checkpoint, return training history."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        best_acc = 0.0

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._train_epoch()
            val_metrics = self.evaluate("val")
            self.scheduler.step()

            epoch_log = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_log)

            logger.info(
                "Epoch %d/%d | loss=%.4f | val_acc=%.4f | lr=%.5f",
                epoch,
                self.config.epochs,
                train_metrics["train_loss"],
                val_metrics["accuracy"],
                self.optimizer.param_groups[0]["lr"],
            )

            if val_metrics["accuracy"] > best_acc:
                best_acc = val_metrics["accuracy"]
                self.save_student(out_dir / "best_student.pth")
                logger.info("  => New best val_acc: %.4f — checkpoint saved.", best_acc)

        history_path = out_dir / "training_history.json"
        with history_path.open("w") as fp:
            json.dump(self.history, fp, indent=2)

        logger.info("Training complete. Best val accuracy: %.4f", best_acc)
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_student(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.student.state_dict(), path)

    def load_student(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.student.load_state_dict(state)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_resnet_kd_trainer(
    config: ResNetKDConfig,
    device: Optional[torch.device] = None,
) -> ResNetKDTrainer:
    """Instantiate teacher (optional), student, dataloaders and return trainer.

    If ``config.teacher_weights`` is empty and ``config.kd.alpha == 0``,
    the teacher is set to ``None`` so no extra forward pass is incurred.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed before building anything
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("Device: %s", device)

    # Teacher
    teacher: Optional[nn.Module] = None
    if config.teacher_weights and config.kd.alpha > 0.0:
        logger.info("Loading teacher (%s) from %s", config.teacher_arch, config.teacher_weights)
        teacher = _build_model_by_arch(
            config.teacher_arch, config.num_classes, config.teacher_weights
        )

    # Student
    logger.info(
        "Building student ResNet18 (weights=%s)",
        config.student_weights or "random init",
    )
    student = build_cifar_resnet18(
        num_classes=config.num_classes,
        weights_path=config.student_weights,
    )

    # Data loaders
    logger.info(
        "Loading %s from %s (subset=%s)", config.dataset.upper(), config.data_dir, config.subset_size
    )
    if config.dataset == "cifar100":
        train_loader, val_loader, test_loader = load_cifar100_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            val_split=config.val_split,
            subset_size=config.subset_size,
            seed=config.seed,
        )
    else:
        train_loader, val_loader, test_loader = load_cifar10_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            val_split=config.val_split,
            subset_size=config.subset_size,
            seed=config.seed,
        )

    return ResNetKDTrainer(
        config=config,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )
