"""Global magnitude pruning pass.

Uses ``torch.nn.utils.prune.global_unstructured`` with ``L1Unstructured`` to
prune a fixed fraction of the lowest-magnitude weights across *all* targeted
layers simultaneously.  After pruning the masks are made permanent (the
``weight_orig`` parameter is removed and ``weight`` is written directly) so
that plain ``state_dict()`` / ``torch.save`` can checkpoint the sparse model.

Reference: Han et al. (2015) "Learning both Weights and Connections for
Efficient Neural Network".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PruneConfig:
    """Configuration for the global L1 unstructured pruning pass."""

    sparsity: float = 0.5
    target_types: tuple = field(default_factory=lambda: (nn.Conv2d, nn.Linear))
    make_permanent: bool = True

    def validate(self) -> None:
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError("sparsity must be in (0, 1)")


# ---------------------------------------------------------------------------
# Sparsity helpers
# ---------------------------------------------------------------------------


def count_nonzero_params(model: nn.Module) -> tuple[int, int]:
    """Return ``(non_zero_count, total_count)`` for all weight parameters."""
    nonzero = total = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            nonzero += int((param.data != 0).sum().item())
    return nonzero, total


def compute_model_sparsity(model: nn.Module) -> float:
    """Return the fraction of zero weight entries across the model."""
    nonzero, total = count_nonzero_params(model)
    if total == 0:
        return 0.0
    return 1.0 - nonzero / total


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


class PrunePass:
    """Apply global L1 unstructured pruning to a PyTorch model.

    Interface:
        pruned_model, info = PrunePass().run(model, pass_args, info)

    ``pass_args`` should be a :class:`PruneConfig` instance.
    ``info`` is a dict that accumulates metrics; pruning stats are merged in.
    """

    name: str = "prune"

    def run(
        self,
        model: nn.Module,
        pass_args: Optional[PruneConfig] = None,
        info: Optional[dict] = None,
    ) -> tuple[nn.Module, dict]:
        """Prune *model* in-place and return it together with updated *info*.

        Args:
            model:     The model to prune (modified in-place).
            pass_args: A :class:`PruneConfig`; defaults to ``PruneConfig()``.
            info:      Existing info dict; pruning statistics are merged in.

        Returns:
            (model, info) with keys ``sparsity_actual``, ``params_nonzero``,
            ``params_total`` added to *info*.
        """
        if pass_args is None:
            pass_args = PruneConfig()
        if info is None:
            info = {}

        pass_args.validate()

        # Collect (module, 'weight') tuples for all targeted layer types
        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, tuple(pass_args.target_types))
            and hasattr(module, "weight")
            and module.weight is not None
        ]

        if not parameters_to_prune:
            raise RuntimeError(
                f"No layers of type {pass_args.target_types} found in model. "
                "Nothing to prune."
            )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pass_args.sparsity,
        )

        if pass_args.make_permanent:
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)

        nonzero, total = count_nonzero_params(model)
        sparsity_actual = 1.0 - nonzero / max(total, 1)

        info.update(
            {
                "sparsity_actual": round(sparsity_actual, 6),
                "params_nonzero": nonzero,
                "params_total": total,
            }
        )
        return model, info
