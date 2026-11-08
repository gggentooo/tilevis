"""Color bar legend displaying active colormap gradient and scale endpoints.

Single responsibility: draw gradient bar indicating min, zero, and max numerical values.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
"""

from __future__ import annotations

import numpy as np
import pygame

from tilevis.render.colormap import Colormap
from tilevis.render.layout import UiScale
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


class LegendWidget:
    """Color bar legend indicating numeric range and color mapping."""

    def __init__(self) -> None:
        self._cached_surf: pygame.Surface | None = None
        self._cached_cmap_name: str = ""
        self._cached_w: int = 0
        self._cached_h: int = 0

    def _get_gradient_surface(self, colormap: Colormap, width: int, height: int) -> pygame.Surface:
        """Create or return cached gradient surface from colormap LUT."""
        w = max(1, width)
        h = max(1, height)
        if (
            self._cached_surf is not None
            and self._cached_cmap_name == colormap.name
            and self._cached_w == w
            and self._cached_h == h
        ):
            return self._cached_surf

        # Sample colormap across 256 points
        lut = colormap.lut  # (256, 3) uint8
        # Create 1 x 256 surface then scale up
        strip = pygame.Surface((256, 1))
        # Direct pixel buffer assignment or pygame.surfarray
        # surfarray.blit_array expects (W, H, 3)
        arr = np.transpose(lut.reshape(1, 256, 3), (1, 0, 2))  # (256, 1, 3)
        pygame.surfarray.blit_array(strip, arr)

        surf = pygame.transform.smoothscale(strip, (w, h))
        self._cached_surf = surf
        self._cached_cmap_name = colormap.name
        self._cached_w = w
        self._cached_h = h
        return surf

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        colormap: Colormap,
        scale_val: float,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
        label: str = "Values",
    ) -> None:
        """Render legend gradient bar and numeric endpoint labels inside rect."""
        if rect.width <= 0 or rect.height <= 0:
            return

        is_sequential = colormap.name.startswith("sequential")
        if is_sequential:
            left_lbl = "0%"
            mid_lbl = "50%"
            right_lbl = "100%"
        else:
            mag = max(0.01, abs(float(scale_val)))
            left_lbl = f"-{mag:.2f}"
            mid_lbl = "0"
            right_lbl = f"+{mag:.2f}"

        font_px = max(9, min(scale.font_px(12, minimum=9), max(9, rect.height // 2)))

        # Partition rect: top row title/values, bottom row gradient bar
        h_bar = max(4, min(scale.px(8, minimum=4), rect.height // 3))
        bar_y = rect.bottom - h_bar - 1
        bar_w = rect.width
        bar_rect = pygame.Rect(rect.x, bar_y, bar_w, h_bar)

        # Draw gradient
        grad = self._get_gradient_surface(colormap, bar_rect.width, bar_rect.height)
        surface.blit(grad, bar_rect.topleft)
        pygame.draw.rect(surface, theme.border, bar_rect, width=1)

        # Draw labels above bar
        left_surf = fonts.render_text(
            left_lbl, font_px, theme.text_secondary, family=theme.font_family
        )
        mid_surf = fonts.render_text(
            mid_lbl, font_px, theme.text_secondary, family=theme.font_family
        )
        right_surf = fonts.render_text(
            right_lbl, font_px, theme.text_secondary, family=theme.font_family
        )

        lbl_y = max(rect.y, bar_y - left_surf.get_height() - 2)
        surface.blit(left_surf, (rect.x, lbl_y))
        surface.blit(mid_surf, (rect.centerx - mid_surf.get_width() // 2, lbl_y))
        surface.blit(right_surf, (rect.right - right_surf.get_width(), lbl_y))
