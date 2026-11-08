"""ArchitectureAdapter protocol and specification dataclasses.

Single responsibility: define the contract for mapping arbitrary nn.Module architectures
to semantic LLM pipeline roles.
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch.nn as nn


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """Semantic mapping for a single transformer block."""

    index: int
    path: str  # e.g., "trf_blocks.0"
    norm1: str | None
    attention: str | None
    q: str | None
    k: str | None
    v: str | None
    out_proj: str | None
    norm2: str | None
    ff: str | None
    ff_in: str | None
    ff_act: str | None
    ff_out: str | None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Complete semantic role architecture mapping of a model."""

    adapter: str
    token_embedding: str
    position_embedding: str | None
    blocks: tuple[BlockSpec, ...]
    final_norm: str | None
    lm_head: str
    emb_dim: int
    vocab_size: int
    context_length: int | None
    n_heads: int | None
    residual_style: str = "pre_norm"


class ArchitectureAdapter(Protocol):
    """Protocol implemented by model architecture adapters."""

    name: str

    def matches(self, model: nn.Module) -> float:
        """Return match confidence between 0.0 and 1.0."""
        ...

    def describe(self, model: nn.Module, overrides: Mapping[str, str] | None = None) -> ModelSpec:
        """Produce a ModelSpec mapping semantic roles to module paths."""
        ...
