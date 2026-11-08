"""LoRA tensor detection and weight merging.

Single responsibility: detect unmerged LoRA adapter weights in checkpoints and fold them
into equivalent dense nn.Linear weights (W_eff = W + alpha * (A @ B).T).
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)


def detect_lora_tensors(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True if state dict contains unmerged LoRA parameters."""
    return any(k.endswith(".lora.A") or k.endswith(".lora.B") for k in state_dict)


def merge_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    alpha: float | None = None,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Fold LoRA parameters into standard dense weights.

    Replaces:
        {prefix}.linear.weight + {prefix}.lora.A + {prefix}.lora.B
    with:
        {prefix}.weight = W + alpha * (A @ B).T
    and copies {prefix}.linear.bias to {prefix}.bias if present.

    Args:
        state_dict: Checkpoint state dictionary.
        alpha: LoRA scaling hyperparameter. If None, defaults to 16.0 with a warning.

    Returns:
        Tuple of (merged_state_dict, was_merged_boolean).
    """
    if not detect_lora_tensors(state_dict):
        return dict(state_dict), False

    if alpha is None:
        logger.warning(
            "LoRA checkpoint without explicit [model.lora.alpha]; using default alpha=16.0. "
            "Predictions may differ if another alpha was used in training."
        )
        scaling = 16.0
    else:
        scaling = float(alpha)

    merged: dict[str, torch.Tensor] = {}
    handled_keys: set[str] = set()

    # Find all prefixes with .lora.A
    for key in state_dict:
        if not key.endswith(".lora.A"):
            continue

        prefix = key[: -len(".lora.A")]
        a_key = f"{prefix}.lora.A"
        b_key = f"{prefix}.lora.B"

        linear_w_key = f"{prefix}.linear.weight"
        plain_w_key = f"{prefix}.weight"

        if linear_w_key in state_dict:
            orig_w_key = linear_w_key
        elif plain_w_key in state_dict:
            orig_w_key = plain_w_key
        else:
            orig_w_key = None

        if orig_w_key is None or b_key not in state_dict:
            continue

        w = state_dict[orig_w_key]
        a = state_dict[a_key]
        b = state_dict[b_key]

        # W_eff = W + alpha * (A @ B).T
        delta_w = scaling * torch.matmul(a, b).t()
        w_eff = (w + delta_w).to(w.dtype)

        merged[f"{prefix}.weight"] = w_eff
        handled_keys.update({orig_w_key, a_key, b_key})

        # Copy bias if present
        linear_bias_key = f"{prefix}.linear.bias"
        plain_bias_key = f"{prefix}.bias"
        if linear_bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[linear_bias_key]
            handled_keys.add(linear_bias_key)
        elif plain_bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[plain_bias_key]
            handled_keys.add(plain_bias_key)

    # Copy remaining non-LoRA keys
    for key, tensor in state_dict.items():
        if key not in handled_keys:
            merged[key] = tensor

    return merged, True
