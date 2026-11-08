"""Scene registry mapping StepKind to Scene classes with GenericScene fallback.

Single responsibility: register and look up Scene implementations by StepKind.
Allowed dependencies: tilevis.core, tilevis.render, stdlib.
"""

from __future__ import annotations

import logging

from tilevis.core.kinds import parse_kind
from tilevis.core.registry import Registry
from tilevis.render.scenes.base import Scene
from tilevis.render.scenes.generic import GenericScene

logger = logging.getLogger("tilevis.render.scenes.registry")

SCENES: Registry[str, type[Scene]] = Registry("SCENES")


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
