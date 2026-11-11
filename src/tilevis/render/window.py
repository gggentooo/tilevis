"""Window mode management (borderless, windowed, exclusive fullscreen).

Single responsibility: create, resize, and toggle pygame display surfaces safely.
Allowed dependencies: pygame, tilevis.core, tilevis.render.platform, stdlib.
"""

from __future__ import annotations

import contextlib
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
        self._window: pygame.Window | None = None
        self.surface: pygame.Surface | None = None
        self.logical_size: tuple[int, int] = windowed_size
        self.drawable_size: tuple[int, int] = windowed_size
        self.size: tuple[int, int] = windowed_size
        self.dpi_scale: tuple[float, float] = (1.0, 1.0)
        self.focused: bool = True

    def _update_dpi_scale(self) -> None:
        lw = max(1, self.logical_size[0])
        lh = max(1, self.logical_size[1])
        dw, dh = self.drawable_size
        self.dpi_scale = (dw / float(lw), dh / float(lh))

    def map_point(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Convert logical window coordinates to physical drawable surface pixels."""
        sx, sy = self.dpi_scale
        return (round(pos[0] * sx), round(pos[1] * sy))

    def present(self) -> None:
        """Present current frame surface via pygame.Window or legacy display flip."""
        if self._window is not None:
            self._window.flip()
        elif pygame.display.get_init():
            pygame.display.flip()

    def destroy(self) -> None:
        """Release window resources cleanly."""
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception as exc:
                logger.debug("Failed to destroy pygame.Window: %s", exc)
            self._window = None
        if pygame.display.get_init():
            pygame.display.quit()

    def create(self, *, headless: bool = False) -> pygame.Surface:
        """Initialize display (with Windows High-DPI first) and return active Surface."""
        enable_high_dpi()

        if headless:
            target_size = (
                get_desktop_size(self.display) if self.mode == "borderless" else self.windowed_size
            )
            w = max(320, target_size[0])
            h = max(240, target_size[1])
            self.logical_size = (w, h)
            self.drawable_size = (w, h)
            self.size = (w, h)
            self.dpi_scale = (1.0, 1.0)
            self.surface = pygame.Surface(self.size)
            return self.surface

        try:
            if not pygame.display.get_init():
                pygame.display.init()

            # Attempt native High-DPI pygame.Window first
            if hasattr(pygame, "Window"):
                try:
                    return self._apply_window_mode(self.mode)
                except Exception as exc:
                    logger.debug("pygame.Window init failed (%s); falling back to set_mode", exc)
                    self._window = None

            return self._apply_legacy_mode(self.mode)
        except pygame.error as exc:
            raise RenderError(f"Unable to open pygame display window: {exc}") from exc

    def _apply_window_mode(self, target_mode: str) -> pygame.Surface:
        """Apply window mode using native High-DPI pygame.Window."""
        mode = target_mode if target_mode in VALID_WINDOW_MODES else "borderless"
        if mode in ("borderless", "exclusive"):
            size = get_desktop_size(self.display)
        else:
            size = self.windowed_size

        if self._window is None:
            self._window = pygame.Window(
                title=self.title,
                size=size,
                borderless=(mode == "borderless"),
                resizable=(mode == "windowed"),
                fullscreen=(mode == "exclusive"),
                allow_high_dpi=True,
            )
            with contextlib.suppress(Exception):
                self._window.set_icon(build_window_icon())
        else:
            if mode == "borderless":
                self._window.borderless = True
                self._window.resizable = False
                if hasattr(self._window, "set_windowed"):
                    self._window.set_windowed()
                self._window.size = size
                with contextlib.suppress(Exception):
                    self._window.position = (0, 0)
            elif mode == "exclusive":
                self._window.borderless = False
                self._window.resizable = False
                if hasattr(self._window, "set_fullscreen"):
                    self._window.set_fullscreen(True)
            else:
                if hasattr(self._window, "set_windowed"):
                    self._window.set_windowed()
                self._window.borderless = False
                self._window.resizable = True
                self._window.size = size

        surf = self._window.get_surface()
        self.mode = mode
        self.surface = surf
        self.logical_size = (int(self._window.size[0]), int(self._window.size[1]))
        self.drawable_size = surf.get_size()
        self.size = self.drawable_size
        self._update_dpi_scale()
        return surf

    def _apply_legacy_mode(self, target_mode: str) -> pygame.Surface:
        """Apply borderless, windowed, or exclusive mode using legacy set_mode."""
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
        self.logical_size = surf.get_size()
        self.drawable_size = surf.get_size()
        self.size = self.drawable_size
        self._update_dpi_scale()
        pygame.display.set_caption(self.title)
        with contextlib.suppress(Exception):
            pygame.display.set_icon(build_window_icon())
        return surf

    def handle_resize(self, width: int, height: int, *, headless: bool = False) -> pygame.Surface:
        """Handle window resize event without losing playback state."""
        w = max(320, int(width))
        h = max(240, int(height))
        if self.mode == "windowed":
            self.windowed_size = (w, h)

        if headless:
            self.logical_size = (w, h)
            self.drawable_size = (w, h)
            self.size = (w, h)
            self.dpi_scale = (1.0, 1.0)
            self.surface = pygame.Surface(self.size)
            return self.surface

        if self._window is not None:
            if self._window.size != (w, h):
                with contextlib.suppress(Exception):
                    self._window.size = (w, h)
            surf = self._window.get_surface()
            self.surface = surf
            self.logical_size = (int(self._window.size[0]), int(self._window.size[1]))
            self.drawable_size = surf.get_size()
            self.size = self.drawable_size
            self._update_dpi_scale()
            return surf

        self.size = (w, h)
        self.logical_size = (w, h)
        self.drawable_size = (w, h)
        self._update_dpi_scale()

        existing = pygame.display.get_surface()
        if existing is not None and existing.get_size() == self.size:
            self.surface = existing
            return existing

        flags = pygame.RESIZABLE if self.mode == "windowed" else pygame.NOFRAME
        self.surface = pygame.display.set_mode(self.size, flags)
        self.drawable_size = self.surface.get_size()
        self.size = self.drawable_size
        self._update_dpi_scale()
        return self.surface

    def toggle_fullscreen(self, *, headless: bool = False) -> pygame.Surface:
        """Toggle between 'windowed' and screenshare-safe 'borderless' modes (F11)."""
        next_mode = "windowed" if self.mode in {"borderless", "exclusive"} else "borderless"
        if headless:
            target = (
                get_desktop_size(self.display) if next_mode == "borderless" else self.windowed_size
            )
            w = max(320, target[0])
            h = max(240, target[1])
            self.mode = next_mode
            self.logical_size = (w, h)
            self.drawable_size = (w, h)
            self.size = (w, h)
            self.dpi_scale = (1.0, 1.0)
            self.surface = pygame.Surface(self.size)
            return self.surface

        if self._window is not None:
            return self._apply_window_mode(next_mode)
        return self._apply_legacy_mode(next_mode)
