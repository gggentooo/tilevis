"""Scene protocol and SceneContext dataclass.

Single responsibility: define the interface that every step scene must implement.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pygame

from tilevis.core.registry import Registry
from tilevis.core.trace import PassRecord, Step, Trace
from tilevis.render.colormap import Colormap
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.normalization import Normalizer
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


@dataclass(frozen=True, slots=True)
class Tooltip:
    """Hover tooltip payload returned by Scene.hit_test()."""

    title: str
    lines: tuple[str, ...]
    pos: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SceneContext:
    """Read-only shared rendering resources passed into Scene.enter()."""

    theme: Theme
    colormaps: Registry[str, Colormap]
    normalizer: Normalizer
    fonts: FontManager
    scale: UiScale
    trace: Trace
    pass_record: PassRecord
    detail_level: int
    nerd_mode: bool
    animation_speed: float
    active_head: int | None = None


class Scene(Protocol):
    """Stateless-per-step scene protocol implemented by all visualization scenes."""

    def enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        """Prepare visual elements for a trace step."""
        ...

    def update(self, dt: float) -> None:
        """Advance ongoing scene animations by dt seconds."""
        ...

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render the scene inside the allocated stage rectangle."""
        ...

    def finish(self) -> None:
        """Jump immediately to the end of the animation."""
        ...

    @property
    def is_animating(self) -> bool:
        """True while the scene has running animations."""
        ...

    def hit_test(self, pos: tuple[int, int]) -> Tooltip | None:
        """Return hover Tooltip at pixel position pos, or None."""
        ...

    def handle_command(self, cmd: Command) -> bool:
        """Optionally consume a scene-specific Command (e.g., head cycling)."""
        ...
