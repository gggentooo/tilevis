"""Resolution-independent layout and UiScale scaling.

Single responsibility: partition viewport into header, stage, caption, and footer regions.
Allowed dependencies: pygame, stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

LOGICAL_WIDTH = 1920
LOGICAL_HEIGHT = 1080


@dataclass(frozen=True, slots=True)
class UiScale:
    """Computes letterboxed logical-to-physical scaling and non-overlapping screen regions."""

    viewport_w: int
    viewport_h: int
    s: float
    canvas_rect: pygame.Rect
    header: pygame.Rect
    stage: pygame.Rect
    caption: pygame.Rect
    footer: pygame.Rect

    @classmethod
    def from_viewport(cls, viewport_w: int, viewport_h: int) -> UiScale:
        """Compute scaled 16:9 letterboxed regions fitting within (viewport_w, viewport_h)."""
        vw = max(1, int(viewport_w))
        vh = max(1, int(viewport_h))
        s = min(vw / float(LOGICAL_WIDTH), vh / float(LOGICAL_HEIGHT))
        if abs(s - 1.0) < 0.01:
            s = 1.0

        cw = max(1, round(LOGICAL_WIDTH * s))
        ch = max(1, round(LOGICAL_HEIGHT * s))
        cx = max(0, (vw - cw) // 2)
        cy = max(0, (vh - ch) // 2)
        canvas = pygame.Rect(cx, cy, cw, ch)

        # Vertical layout without gaps or overlaps: header 9%, stage 70%, caption 15%, footer 6%
        h_header = max(1, round(ch * 0.09))
        h_stage = max(1, round(ch * 0.70))
        h_caption = max(1, round(ch * 0.15))

        # Clamp heights on ultra-tiny viewports to guarantee footer >= 1 and exact sum == ch
        total_top = h_header + h_stage + h_caption
        if total_top >= ch:
            excess = total_top - max(0, ch - 1)
            take_stage = min(excess, max(0, h_stage - 1))
            h_stage -= take_stage
            excess -= take_stage
            if excess > 0:
                take_cap = min(excess, max(0, h_caption - 1))
                h_caption -= take_cap
                excess -= take_cap
            if excess > 0:
                h_header = max(1, h_header - excess)

        h_footer = max(1, ch - (h_header + h_stage + h_caption))

        y0 = canvas.y
        y1 = y0 + h_header
        y2 = y1 + h_stage
        y3 = y2 + h_caption

        return cls(
            viewport_w=vw,
            viewport_h=vh,
            s=s,
            canvas_rect=canvas,
            header=pygame.Rect(canvas.x, y0, cw, h_header),
            stage=pygame.Rect(canvas.x, y1, cw, h_stage),
            caption=pygame.Rect(canvas.x, y2, cw, h_caption),
            footer=pygame.Rect(canvas.x, y3, cw, h_footer),
        )

    def px(self, logical_px: float, minimum: int = 1) -> int:
        """Convert logical 1080p pixels into physical viewport pixels."""
        return max(minimum, round(logical_px * self.s))

    def font_px(self, logical_px: float, minimum: int = 10) -> int:
        """Convert logical font size into physical pixels with a legible floor."""
        return max(minimum, round(logical_px * self.s))
