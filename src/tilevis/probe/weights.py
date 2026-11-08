"""Weight extraction and matrix orientation normalization.

Single responsibility: extract weights and biases from PyTorch modules into numpy arrays
transposed according to the (in x out) presentation orientation convention.
Allowed dependencies: stdlib, numpy, torch, tilevis.core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from tilevis.core.registry import Registry


@dataclass(frozen=True, slots=True)
class ExtractedWeights:
    """Extracted module weights and parameters normalized to presentation orientation.

    Orientation convention:
    - Linear/LoRA weight is (in_features, out_features).
    - Embedding weight is (num_embeddings, embedding_dim).
    - Bias, scale, shift are 1-D feature vectors.
    """

    weight: np.ndarray | None = None
    bias: np.ndarray | None = None
    scale: np.ndarray | None = None
    shift: np.ndarray | None = None


# Registry of specialized extractor functions: module_type_name -> extractor_func
WEIGHT_EXTRACTORS: Registry[str, Callable[[nn.Module], ExtractedWeights | None]] = Registry(
    "weight_extractors"
)


def _extract_lora(module: nn.Module) -> ExtractedWeights | None:
    """Extract effective weights from a LinearWithLoRA wrapper."""
    if not (hasattr(module, "linear") and hasattr(module, "lora")):
        return None

    linear = module.linear
    lora = module.lora

    w = getattr(linear, "weight", None)
    a = getattr(lora, "A", None)
    b = getattr(lora, "B", None)

    if not (
        isinstance(w, torch.Tensor) and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    ):
        return None

    alpha = float(getattr(lora, "alpha", 16.0))

    # W_eff = W + alpha * (A @ B).T
    # Shape of W: (out_features, in_features)
    # Shape of A: (in_features, rank)
    # Shape of B: (rank, out_features)
    delta_w = alpha * torch.matmul(a, b).t()
    w_eff = w + delta_w

    # Transpose to (in_features, out_features)
    weight_np = w_eff.t().detach().cpu().to(torch.float32).numpy()
    bias = getattr(linear, "bias", None)
    bias_np = (
        bias.detach().cpu().to(torch.float32).numpy() if isinstance(bias, torch.Tensor) else None
    )

    return ExtractedWeights(weight=weight_np, bias=bias_np)


def _extract_linear(module: nn.Module) -> ExtractedWeights | None:
    """Extract weight and bias from nn.Linear transposed to (in_features, out_features)."""
    if not isinstance(module, nn.Linear):
        return None

    # nn.Linear.weight is (out_features, in_features); transpose to (in_features, out_features)
    weight_np = module.weight.t().detach().cpu().to(torch.float32).numpy()
    bias_np = (
        module.bias.detach().cpu().to(torch.float32).numpy() if module.bias is not None else None
    )
    return ExtractedWeights(weight=weight_np, bias=bias_np)


def _extract_embedding(module: nn.Module) -> ExtractedWeights | None:
    """Extract weight from nn.Embedding (vocab_size, emb_dim)."""
    if not isinstance(module, nn.Embedding):
        return None

    weight_np = module.weight.detach().cpu().to(torch.float32).numpy()
    return ExtractedWeights(weight=weight_np)


def _extract_layernorm(module: nn.Module) -> ExtractedWeights | None:
    """Extract scale and shift from LayerNorm-like modules."""
    # Check for custom scale/shift attributes (e.g. previous_chapters LayerNorm)
    if hasattr(module, "scale") and hasattr(module, "shift"):
        scale_t = module.scale
        shift_t = module.shift
        scale_np = (
            scale_t.detach().cpu().to(torch.float32).numpy()
            if isinstance(scale_t, torch.Tensor)
            else None
        )
        shift_np = (
            shift_t.detach().cpu().to(torch.float32).numpy()
            if isinstance(shift_t, torch.Tensor)
            else None
        )
        return ExtractedWeights(scale=scale_np, shift=shift_np)

    # Standard nn.LayerNorm uses weight and bias
    if isinstance(module, nn.LayerNorm) or (hasattr(module, "weight") and hasattr(module, "bias")):
        w = module.weight
        b = module.bias
        scale_np = (
            w.detach().cpu().to(torch.float32).numpy() if isinstance(w, torch.Tensor) else None
        )
        shift_np = (
            b.detach().cpu().to(torch.float32).numpy() if isinstance(b, torch.Tensor) else None
        )
        return ExtractedWeights(scale=scale_np, shift=shift_np)

    return None


WEIGHT_EXTRACTORS.register("lora")(_extract_lora)
WEIGHT_EXTRACTORS.register("linear")(_extract_linear)
WEIGHT_EXTRACTORS.register("embedding")(_extract_embedding)
WEIGHT_EXTRACTORS.register("layernorm")(_extract_layernorm)


def extract_weights(module: nn.Module) -> ExtractedWeights | None:
    """Extract presentation-ready weights from an nn.Module instance.

    Tries extractors in priority order: LoRA wrapper -> Linear -> Embedding -> LayerNorm.

    Args:
        module: Target PyTorch module.

    Returns:
        ExtractedWeights if module matches a known extractor, else None.
    """
    # 1. First check LoRA wrapper before Linear
    lora_result = _extract_lora(module)
    if lora_result is not None:
        return lora_result

    # 2. Check linear
    linear_result = _extract_linear(module)
    if linear_result is not None:
        return linear_result

    # 3. Check embedding
    emb_result = _extract_embedding(module)
    if emb_result is not None:
        return emb_result

    # 4. Check layernorm
    ln_result = _extract_layernorm(module)
    if ln_result is not None:
        return ln_result

    # 5. Check any registered plugin extractors
    for key in WEIGHT_EXTRACTORS:
        extractor = WEIGHT_EXTRACTORS.get(key)
        res = extractor(module)
        if res is not None:
            return res

    return None
