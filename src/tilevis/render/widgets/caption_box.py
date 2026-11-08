"""Caption box widget displaying title and explanation at the bottom of the screen.

Single responsibility: format and draw multi-line friendly captions with word wrapping.
Allowed dependencies: tilevis.core, tilevis.narration, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import pygame

from tilevis.narration.captions import Caption
from tilevis.render.layout import UiScale
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


class CaptionBox:
    """Renders formatted Caption title, body, and optional nerd-mode summary line."""

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        caption: Caption,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
        nerd_mode: bool = False,
    ) -> None:
        """Draw caption panel into rect using scaled legible typography."""
        if rect.width <= 0 or rect.height <= 0:
            return

        pad_x = max(4, min(scale.px(20, minimum=4), rect.width // 16))
        pad_y = max(2, min(scale.px(10, minimum=2), rect.height // 8))
        box = pygame.Rect(
            rect.x + pad_x,
            rect.y + max(1, pad_y // 2),
            max(1, rect.width - 2 * pad_x),
            max(1, rect.height - max(2, pad_y)),
        )
        pygame.draw.rect(
            surface,
            theme.panel,
            box,
            border_radius=max(2, scale.px(theme.corner_radius)),
        )
        pygame.draw.rect(
            surface,
            theme.border,
            box,
            width=1,
            border_radius=max(2, scale.px(theme.corner_radius)),
        )

        inner_pad = max(4, min(scale.px(16, minimum=4), box.width // 16, box.height // 6))
        content_w = max(12, box.width - 2 * inner_pad)

        # Typography scales cleanly while guaranteeing non-zero legibility in tiny windows
        title_px = max(10, min(scale.font_px(theme.heading_font_px, minimum=10), box.height // 3))
        body_px = max(10, min(scale.font_px(theme.body_font_px, minimum=10), box.height // 4))

        cur_y = box.y + inner_pad // 2
        title_surf = fonts.render_text(
            caption.title,
            title_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=content_w,
        )
        surface.blit(title_surf, (box.x + inner_pad, cur_y))
        cur_y += title_surf.get_height() + max(2, scale.px(4))

        avail_h = max(1, box.bottom - inner_pad // 2 - cur_y)
        line_h = max(10, fonts.get(body_px, family=theme.font_family).get_linesize())
        max_lines = max(1, avail_h // line_h)

        if nerd_mode and caption.nerd and max_lines >= 2:
            body_lines = fonts.wrap_text(
                caption.body,
                content_w,
                body_px,
                family=theme.font_family,
                max_lines=max_lines - 1,
            )
        else:
            body_lines = fonts.wrap_text(
                caption.body,
                content_w,
                body_px,
                family=theme.font_family,
                max_lines=max_lines,
            )

        for line in body_lines:
            if cur_y + line_h > box.bottom:
                break
            line_surf = fonts.render_text(
                line,
                body_px,
                theme.text_secondary,
                family=theme.font_family,
            )
            surface.blit(line_surf, (box.x + inner_pad, cur_y))
            cur_y += line_h

        if nerd_mode and caption.nerd and cur_y + 10 <= box.bottom:
            nerd_px = max(9, body_px - 3)
            nerd_surf = fonts.render_text(
                f"[Nerd] {caption.nerd}",
                nerd_px,
                theme.accent,
                family=theme.font_family,
                max_width=content_w,
            )
            surface.blit(nerd_surf, (box.x + inner_pad, cur_y))
