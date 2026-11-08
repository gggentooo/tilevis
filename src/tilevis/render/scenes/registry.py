"""Scene registry mapping StepKind to Scene classes with GenericScene fallback.

Single responsibility: register and look up Scene implementations by StepKind.
Allowed dependencies: tilevis.core, tilevis.render, stdlib.
"""

from __future__ import annotations

import logging

from tilevis.core.kinds import StepKind, parse_kind
from tilevis.core.registry import Registry
from tilevis.render.scenes.base import Scene
from tilevis.render.scenes.generic import GenericScene

logger = logging.getLogger("tilevis.render.scenes.registry")

SCENES: Registry[str, type[Scene]] = Registry("SCENES")
SCENES.register(StepKind.GENERIC.value)(GenericScene)


def create_scene(kind: str) -> Scene:
    """Instantiate a registered Scene for StepKind string, falling back to GenericScene."""
    parsed = parse_kind(kind)
    key = str(parsed)
    if key in SCENES:
        cls = SCENES.get(key)
        return cls()
    if kind in SCENES:
        cls = SCENES.get(kind)
        return cls()
    return GenericScene()


# Load built-in dedicated scenes after SCENES & create_scene exist so their
# @SCENES.register(...) decorators execute automatically without circular imports.
from tilevis.render.scenes import (  # noqa: E402, F401
    activation,
    attention,
    attention_mix,
    block_summary,
    embedding,
    layernorm,
    logits,
    matmul,
    prediction,
    qkv,
    residual,
    title,
    tokenize,
)
