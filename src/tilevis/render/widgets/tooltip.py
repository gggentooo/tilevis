"""Hover tooltip displaying coordinates and unreduced numerical values.

Single responsibility: render edge-clamped float values and original axis indices on mouse hover.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import pygame

from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import Tooltip
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


class TooltipWidget:
    """Renders edge-clamped floating hover tooltip card."""

    def compute_rect(
        self,
        tooltip: Tooltip,
        viewport_w: int,
        viewport_h: int,
        *,
        fonts: FontManager,
        scale: UiScale,
        theme: Theme,
    ) -> tuple[pygame.Rect, list[pygame.Surface], pygame.Surface]:
        """Compute clamped tooltip card rect and pre-rendered text surfaces."""
        vw = max(1, int(viewport_w))
        vh = max(1, int(viewport_h))

        pad_x = max(6, scale.px(10, minimum=6))
        pad_y = max(6, scale.px(8, minimum=6))
        max_card_w = max(100, min(scale.px(360, minimum=200), vw - 2 * pad_x))
        text_max_w = max(40, max_card_w - 2 * pad_x)

        title_font_px = max(10, scale.font_px(13, minimum=10))
        line_font_px = max(9, scale.font_px(11, minimum=9))

        title_surf = fonts.render_text(
            tooltip.title,
            title_font_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=text_max_w,
        )

        line_surfs: list[pygame.Surface] = []
        for line in tooltip.lines:
            # Handle long labels gracefully via max_width truncation
            s = fonts.render_text(
                line,
                line_font_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=text_max_w,
            )
            line_surfs.append(s)

        content_w = title_surf.get_width()
        for s in line_surfs:
            if s.get_width() > content_w:
                content_w = s.get_width()

        card_w = min(vw, content_w + 2 * pad_x)

        gap_y = max(2, scale.px(3, minimum=2))
        total_text_h = title_surf.get_height() + gap_y * 2
        for s in line_surfs:
            total_text_h += s.get_height() + gap_y

        card_h = min(vh, total_text_h + 2 * pad_y)

        # Positioning with flip-and-clamp logic
        cx, cy = tooltip.pos
        offset_x = max(8, scale.px(12, minimum=8))
        offset_y = max(8, scale.px(12, minimum=8))

        # Default: bottom-right
        x = cx + offset_x
        y = cy + offset_y

        # Flip horizontally if overflowing right
        if x + card_w > vw:
            x = cx - offset_x - card_w

        # Flip vertically if overflowing bottom
        if y + card_h > vh:
            y = cy - offset_y - card_h

        # Strictly clamp within 0..vw and 0..vh
        x = max(0, min(x, vw - card_w))
        y = max(0, min(y, vh - card_h))

        return pygame.Rect(x, y, card_w, card_h), line_surfs, title_surf

    def draw(
        self,
        surface: pygame.Surface,
        tooltip: Tooltip,
        viewport_w: int,
        viewport_h: int,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
    ) -> None:
        """Render tooltip card into surface."""
        rect, line_surfs, title_surf = self.compute_rect(
            tooltip,
            viewport_w,
            viewport_h,
            fonts=fonts,
            scale=scale,
            theme=theme,
        )

        pad_x = max(6, scale.px(10, minimum=6))
        pad_y = max(6, scale.px(8, minimum=6))
        radius = max(2, scale.px(4, minimum=2))

        # Draw card background and accent border
        pygame.draw.rect(surface, theme.panel, rect, border_radius=radius)
        pygame.draw.rect(surface, theme.accent, rect, width=1, border_radius=radius)

        cur_y = rect.y + pad_y
        surface.blit(title_surf, (rect.x + pad_x, cur_y))
        cur_y += title_surf.get_height() + max(2, scale.px(4, minimum=2))

        # Subtle divider
        pygame.draw.line(
            surface,
            theme.border,
            (rect.x + pad_x, cur_y),
            (rect.right - pad_x, cur_y),
            width=1,
        )
        cur_y += max(3, scale.px(4, minimum=3))

        gap_y = max(2, scale.px(3, minimum=2))
        for s in line_surfs:
            if cur_y + s.get_height() <= rect.bottom - pad_y + 2:
                surface.blit(s, (rect.x + pad_x, cur_y))
                cur_y += s.get_height() + gap_y
