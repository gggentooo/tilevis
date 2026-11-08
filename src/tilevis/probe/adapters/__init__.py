"""Model architecture adapters package.

Single responsibility: register and select adapters that map nn.Module architectures
to semantic ModelSpec configurations.
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch.nn as nn

from tilevis.core.registry import Registry
from tilevis.probe.adapters.base import ArchitectureAdapter, BlockSpec, ModelSpec
from tilevis.probe.adapters.generic import GenericAdapter
from tilevis.probe.adapters.gpt_scratch import GptScratchAdapter

logger = logging.getLogger(__name__)

ADAPTERS: Registry[str, ArchitectureAdapter] = Registry("adapters")
ADAPTERS.register("gpt_scratch")(GptScratchAdapter())
ADAPTERS.register("generic")(GenericAdapter())


def select_adapter(
    model: nn.Module,
    adapter_name: str = "auto",
    overrides: Mapping[str, str] | None = None,
) -> tuple[ArchitectureAdapter, ModelSpec, float]:
    """Select the best matching adapter for a model or use the explicitly requested one.

    Args:
        model: Target PyTorch nn.Module instance.
        adapter_name: "auto" for heuristic detection, or explicit adapter name.
        overrides: Optional role mapping overrides.

    Returns:
        Tuple of (chosen_adapter, generated_model_spec, match_confidence).
    """
    ov = dict(overrides or {})

    if adapter_name != "auto":
        adapter = ADAPTERS.get(adapter_name)
        confidence = adapter.matches(model)
        spec = adapter.describe(model, ov)
        return adapter, spec, confidence

    # Automatic selection
    best_adapter: ArchitectureAdapter = ADAPTERS.get("generic")
    best_score = 0.0

    for key in ADAPTERS:
        cand = ADAPTERS.get(key)
        score = cand.matches(model)
        if score > best_score:
            best_score = score
            best_adapter = cand

    if best_score < 0.5:
        logger.info(
            "No architecture adapter matched with >= 0.5 confidence (best: %s @ %.2f); "
            "falling back to 'generic'",
            best_adapter.name,
            best_score,
        )
        best_adapter = ADAPTERS.get("generic")
        best_score = 0.5
    else:
        logger.info(
            "Selected architecture adapter %r with confidence %.2f",
            best_adapter.name,
            best_score,
        )

    spec = best_adapter.describe(model, ov)
    return best_adapter, spec, best_score


__all__ = ["ADAPTERS", "ArchitectureAdapter", "BlockSpec", "ModelSpec", "select_adapter"]
