"""Window mode management (borderless, windowed, exclusive fullscreen).

Single responsibility: create, resize, and toggle pygame display surfaces safely.
Allowed dependencies: pygame, tilevis.core, tilevis.render.platform, stdlib.
"""

from __future__ import annotations

import logging

import pygame

from tilevis.core.errors import RenderError
from tilevis.render.platform import enable_high_dpi, get_desktop_size

logger = logging.getLogger("tilevis.render.window")

VALID_WINDOW_MODES = ("borderless", "windowed", "exclusive")


def build_window_icon() -> pygame.Surface:
    """Build a crisp 32x32 3x3 colored-tile programmatic icon surface."""
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    surf.fill((15, 18, 25))
    palette = [
        [(30, 110, 188), (136, 176, 216), (242, 243, 245)],
        [(136, 176, 216), (242, 243, 245), (233, 170, 128)],
        [(242, 243, 245), (233, 170, 128), (224, 96, 12)],
    ]
    for r in range(3):
        for c in range(3):
            pygame.draw.rect(
                surf,
                palette[r][c],
                pygame.Rect(2 + c * 10, 2 + r * 10, 8, 8),
                border_radius=1,
            )
    return surf


class WindowManager:
    """Manages pygame display initialization, window modes, icon/title, and resizes."""

    def __init__(
        self,
        mode: str = "borderless",
        display: int = 0,
        windowed_size: tuple[int, int] = (1600, 900),
        title: str = "LLM Tile Visualizer — tilevis",
    ) -> None:
        self.mode = mode if mode in VALID_WINDOW_MODES else "borderless"
        self.display = display
        self.windowed_size = windowed_size
        self.title = title
        self.surface: pygame.Surface | None = None
        self.size: tuple[int, int] = windowed_size
        self.focused: bool = True

    def create(self, *, headless: bool = False) -> pygame.Surface:
        """Initialize display (with Windows High-DPI first) and return active Surface."""
        enable_high_dpi()

        if headless:
            target_size = (
                get_desktop_size(self.display) if self.mode == "borderless" else self.windowed_size
            )
            self.size = (max(320, target_size[0]), max(240, target_size[1]))
            self.surface = pygame.Surface(self.size)
            return self.surface

        try:
            if not pygame.display.get_init():
                pygame.display.init()
            pygame.display.set_caption(self.title)
            pygame.display.set_icon(build_window_icon())
            return self._apply_mode(self.mode)
        except pygame.error as exc:
            raise RenderError(f"Unable to open pygame display window: {exc}") from exc

    def _apply_mode(self, target_mode: str) -> pygame.Surface:
        """Apply borderless, windowed, or exclusive mode to pygame display."""
        mode = target_mode if target_mode in VALID_WINDOW_MODES else "borderless"
        if mode == "borderless":
            size = get_desktop_size(self.display)
            flags = pygame.NOFRAME
        elif mode == "exclusive":
            size = get_desktop_size(self.display)
            flags = pygame.FULLSCREEN
        else:
            size = self.windowed_size
            flags = pygame.RESIZABLE

        try:
            surf = pygame.display.set_mode(size, flags, display=self.display)
        except (TypeError, pygame.error):
            # Some dummy/older drivers reject display kwarg or desktop sizes; fallback safely.
            surf = pygame.display.set_mode(size, flags)

        self.mode = mode
        self.surface = surf
        self.size = surf.get_size()
        pygame.display.set_caption(self.title)
        return surf

    def handle_resize(self, width: int, height: int, *, headless: bool = False) -> pygame.Surface:
        """Handle window resize event without losing playback state."""
        w = max(320, int(width))
        h = max(240, int(height))
        if self.mode == "windowed":
            self.windowed_size = (w, h)
        self.size = (w, h)

        if headless:
            self.surface = pygame.Surface(self.size)
            return self.surface

        existing = pygame.display.get_surface()
        if existing is not None and existing.get_size() == self.size:
            self.surface = existing
            return existing

        flags = pygame.RESIZABLE if self.mode == "windowed" else pygame.NOFRAME
        self.surface = pygame.display.set_mode(self.size, flags)
        return self.surface

    def toggle_fullscreen(self, *, headless: bool = False) -> pygame.Surface:
        """Toggle between 'windowed' and screenshare-safe 'borderless' modes (F11)."""
        next_mode = "windowed" if self.mode in {"borderless", "exclusive"} else "borderless"
        if headless:
            self.mode = next_mode
            target = (
                get_desktop_size(self.display) if next_mode == "borderless" else self.windowed_size
            )
            self.size = (max(320, target[0]), max(240, target[1]))
            self.surface = pygame.Surface(self.size)
            return self.surface
        return self._apply_mode(next_mode)
