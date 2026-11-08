"""Configuration inference from model checkpoint state-dict shapes.

Single responsibility: inspect state dict tensor shapes, deduce GPT model parameters,
merge explicit config overrides, and flag contradictions.
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from tilevis.core.errors import ConfigInferenceError


@dataclass(frozen=True, slots=True)
class InferredConfig:
    """Result of config inference distinguishing inferred and explicit parameters."""

    cfg: dict[str, Any]
    inferred_keys: frozenset[str]
    explicit_keys: frozenset[str]


_BLOCK_REGEX = re.compile(r"^trf_blocks\.(\d+)\.")


def infer_n_heads_provisional(emb_dim: int) -> int:
    """Compute a provisional n_heads value when omitted from profile and checkpoint.

    Prefers GPT-2 standard head dimension 64 if divisible, otherwise finds a sensible divisor.
    """
    if emb_dim % 64 == 0:
        return emb_dim // 64
    for divisor in (8, 4, 2):
        if emb_dim % divisor == 0:
            return divisor
    return 1


def infer_config_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    explicit_config: Mapping[str, Any] | None = None,
) -> InferredConfig:
    """Infer GPT model architecture configuration from checkpoint weights.

    Args:
        state_dict: Normalized checkpoint tensors.
        explicit_config: Optional profile/CLI configuration overrides.

    Returns:
        InferredConfig containing final merged dictionary and source tracking.

    Raises:
        ConfigInferenceError: If explicit config contradicts checkpoint tensor shapes.
    """
    explicit = dict(explicit_config or {})
    inferred: dict[str, Any] = {}
    inferred_keys: set[str] = set()
    explicit_keys: set[str] = set()

    # 1. vocab_size and emb_dim from tok_emb.weight
    if "tok_emb.weight" in state_dict:
        t_shape = state_dict["tok_emb.weight"].shape
        inferred_vocab, inferred_emb = int(t_shape[0]), int(t_shape[1])
        inferred["vocab_size"] = inferred_vocab
        inferred["emb_dim"] = inferred_emb
        inferred_keys.update({"vocab_size", "emb_dim"})

        if "vocab_size" in explicit and explicit["vocab_size"] != inferred_vocab:
            raise ConfigInferenceError(
                f"Contradiction: explicit vocab_size ({explicit['vocab_size']}) "
                f"does not match checkpoint shape ({inferred_vocab})"
            )
        if "emb_dim" in explicit and explicit["emb_dim"] != inferred_emb:
            raise ConfigInferenceError(
                f"Contradiction: explicit emb_dim ({explicit['emb_dim']}) "
                f"does not match checkpoint shape ({inferred_emb})"
            )

    # 2. context_length from pos_emb.weight
    if "pos_emb.weight" in state_dict:
        inferred_ctx = int(state_dict["pos_emb.weight"].shape[0])
        inferred["context_length"] = inferred_ctx
        inferred_keys.add("context_length")

        if "context_length" in explicit and explicit["context_length"] != inferred_ctx:
            raise ConfigInferenceError(
                f"Contradiction: explicit context_length ({explicit['context_length']}) "
                f"does not match checkpoint shape ({inferred_ctx})"
            )

    # 3. n_layers from trf_blocks.{i}.
    block_indices: set[int] = set()
    for key in state_dict:
        match = _BLOCK_REGEX.match(key)
        if match:
            block_indices.add(int(match.group(1)))

    if block_indices:
        inferred_layers = len(block_indices)
        inferred["n_layers"] = inferred_layers
        inferred_keys.add("n_layers")

        if "n_layers" in explicit and explicit["n_layers"] != inferred_layers:
            raise ConfigInferenceError(
                f"Contradiction: explicit n_layers ({explicit['n_layers']}) "
                f"does not match checkpoint layers ({inferred_layers})"
            )

    # 4. qkv_bias from W_query bias presence
    has_qkv_bias = any(
        k.endswith("att.W_query.bias") or k.endswith("att.W_query.linear.bias") for k in state_dict
    )
    # Check if W_query exists at all
    has_w_query = any("att.W_query" in k for k in state_dict)
    if has_w_query:
        inferred["qkv_bias"] = has_qkv_bias
        inferred_keys.add("qkv_bias")

        if "qkv_bias" in explicit and bool(explicit["qkv_bias"]) != has_qkv_bias:
            raise ConfigInferenceError(
                f"Contradiction: explicit qkv_bias ({explicit['qkv_bias']}) "
                f"contradicts bias tensor presence in checkpoint ({has_qkv_bias})"
            )

    # 5. drop_rate is always 0.0 for probe evaluation
    inferred["drop_rate"] = 0.0
    inferred_keys.add("drop_rate")

    # Final merge: explicit overrides inferred
    final_cfg: dict[str, Any] = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 64,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    final_cfg.update(inferred)

    for k, v in explicit.items():
        final_cfg[k] = v
        explicit_keys.add(k)
        if k in inferred_keys:
            inferred_keys.remove(k)

    # 6. n_heads handling
    if "n_heads" not in final_cfg:
        provisional_heads = infer_n_heads_provisional(int(final_cfg["emb_dim"]))
        final_cfg["n_heads"] = provisional_heads
        inferred_keys.add("n_heads")

    return InferredConfig(
        cfg=final_cfg,
        inferred_keys=frozenset(inferred_keys),
        explicit_keys=frozenset(explicit_keys),
    )
