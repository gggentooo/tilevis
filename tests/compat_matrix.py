"""Compatibility matrix defining model architecture variants tested across phases.

Referenced by tests in all phases to guarantee model-agnostic behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelMatrixEntry:
    """Specification of a test model architecture variant."""

    name: str
    emb_dim: int
    n_layers: int
    n_heads: int
    context_length: int
    qkv_bias: bool
    variant: str
    purpose: str
    extras: dict[str, Any] | None = None

    def to_config_dict(self, vocab_size: int = 50257) -> dict[str, Any]:
        """Convert entry into standard GPTModel configuration dictionary."""
        return {
            "vocab_size": vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "drop_rate": 0.0,
            "qkv_bias": self.qkv_bias,
        }


COMPAT_MATRIX: dict[str, ModelMatrixEntry] = {
    "tiny_ch07": ModelMatrixEntry(
        name="tiny_ch07",
        emb_dim=12,
        n_layers=1,
        n_heads=2,
        context_length=120,
        qkv_bias=False,
        variant="plain",
        purpose="exact --test_mode config from gpt_instruction_finetuning.py",
    ),
    "small_bias": ModelMatrixEntry(
        name="small_bias",
        emb_dim=48,
        n_layers=3,
        n_heads=4,
        context_length=64,
        qkv_bias=True,
        variant="plain",
        purpose="typical student model configuration",
    ),
    "heads_8_of_64": ModelMatrixEntry(
        name="heads_8_of_64",
        emb_dim=64,
        n_layers=2,
        n_heads=8,
        context_length=64,
        qkv_bias=False,
        variant="plain",
        purpose="n_heads detection with many possible divisors",
    ),
    "head_dim_1": ModelMatrixEntry(
        name="head_dim_1",
        emb_dim=8,
        n_layers=1,
        n_heads=8,
        context_length=32,
        qkv_bias=False,
        variant="plain",
        purpose="degenerate single-feature head size",
    ),
    "deep_narrow": ModelMatrixEntry(
        name="deep_narrow",
        emb_dim=32,
        n_layers=12,
        n_heads=4,
        context_length=64,
        qkv_bias=True,
        variant="plain",
        purpose="many blocks: tests playlist filtering and minimap collapsing",
    ),
    "lora_small": ModelMatrixEntry(
        name="lora_small",
        emb_dim=48,
        n_layers=2,
        n_heads=4,
        context_length=64,
        qkv_bias=True,
        variant="lora",
        purpose="LoRA rank=4 alpha=8 weight folding",
        extras={"lora_rank": 4, "lora_alpha": 8},
    ),
    "wrapped_ckpt": ModelMatrixEntry(
        name="wrapped_ckpt",
        emb_dim=48,
        n_layers=2,
        n_heads=4,
        context_length=64,
        qkv_bias=False,
        variant="wrapped",
        purpose="checkpoint unwrapping from {'model_state_dict': ...} + module. prefix",
    ),
    "gpt2_small_shape": ModelMatrixEntry(
        name="gpt2_small_shape",
        emb_dim=768,
        n_layers=12,
        n_heads=12,
        context_length=1024,
        qkv_bias=True,
        variant="plain",
        purpose="full GPT-2 small real-size performance and memory budget verification",
    ),
}
