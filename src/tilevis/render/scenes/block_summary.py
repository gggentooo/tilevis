"""Block summary scene for Story mode and collapsed transformer layers.

Single responsibility: show layer input -> output and a 'what changed' delta grid.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math

import pygame

from tilevis.core.kinds import StepKind, TensorRole
from tilevis.core.trace import Step, TensorSnapshot
from tilevis.render.animation import ease_out_cubic
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    GridCard,
    draw_arrow,
    draw_grid_card,
    draw_pill,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


@SCENES.register(StepKind.BLOCK_SUMMARY.value)
class BlockSummaryScene(BaseScene):
    """Presents one transformer block's input -> output alongside output - input change."""

    DEFAULT_DURATION_SEC = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._in_snap: TensorSnapshot | None = None
        self._out_snap: TensorSnapshot | None = None
        self._upd_snap: TensorSnapshot | None = None
        self._banner_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._arrows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._block_label: str = "Layer Summary"

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        self._in_snap = find_snapshot(step, TensorRole.INPUT)
        self._out_snap = find_snapshot(step, TensorRole.OUTPUT)
        self._upd_snap = find_snapshot(step, TensorRole.UPDATE)

        b_num = (step.block_index + 1) if step.block_index is not None else "?"
        n_layers = ctx.trace.model.n_layers
        rel_txt = ""
        if self._in_snap is not None and self._upd_snap is not None:
            rel_pct = (self._upd_snap.stats.std / max(1e-6, self._in_snap.stats.std)) * 100.0
            rel_txt = f"  .  avg pattern change ~{rel_pct:.0f}%"
        self._block_label = f"Layer {b_num} of {n_layers}{rel_txt}"

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._arrows = []
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        banner_h = max(24, min(scale.px(38, minimum=22), stage.height // 10))
        self._banner_rect = pygame.Rect(
            stage.x + outer_pad,
            stage.y + outer_pad // 2,
            min(max(120, stage.width - 2 * outer_pad), scale.px(620, minimum=120)),
            banner_h,
        )

        panels: list[tuple[str, str, TensorSnapshot, bool]] = []
        if self._in_snap is not None:
            panels.append(("Entering Layer", "Input tiles", self._in_snap, False))
        if self._out_snap is not None:
            panels.append(("Leaving Layer", "Updated output tiles", self._out_snap, True))
        if self._upd_snap is not None:
            panels.append(("What Changed", "Output minus Input", self._upd_snap, False))
        if not panels:
            return

        top_y = self._banner_rect.bottom + max(6, scale.px(12, minimum=4))
        body_h = max(36, stage.bottom - outer_pad // 2 - top_y)
        avail_w = max(30, stage.width - 2 * outer_pad)
        gap = max(14, min(scale.px(28, minimum=10), avail_w // 18))

        n = len(panels)
        card_w = max(20, (avail_w - gap * (n - 1)) // n)
        hdr_h = min(max(24, scale.px(48, minimum=24)), max(24, body_h // 4))
        inner_pad = max(4, min(scale.px(12, minimum=4), card_w // 14))

        for i, (title, hint, snap, hi) in enumerate(panels):
            cx = stage.x + outer_pad + i * (card_w + gap)
            rect = pygame.Rect(cx, top_y, card_w, body_h)
            if i == 1:
                self._arrows.append(((cx - gap + 2, rect.centery), (cx - 2, rect.centery)))
            gw = max(1, rect.width - 2 * inner_pad)
            gh = max(1, rect.height - hdr_h - 2 * inner_pad)
            grid = make_grid(
                snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(18, minimum=6),
                min_cell_px=max(1, min(6, gw, gh)),
            )
            place_grid_in_card(rect, grid, hdr_h, inner_pad)
            if grid.downsample_note:
                self.downsample_notes.append(grid.downsample_note)
            shape_txt = "x".join(str(s) for s in snap.full_shape)
            sub = f"{hint} ({shape_txt})" + (
                f" . {grid.downsample_note}" if grid.downsample_note else ""
            )
            self._cards.append(
                GridCard(
                    title=title,
                    subtitle=sub,
                    rect=rect,
                    grid=grid,
                    tooltip_title=title,
                    role=snap.role,
                    highlighted=hi,
                )
            )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render block input -> output and delta tiles."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        b_px = max(10, min(scale.font_px(20, minimum=10), max(10, self._banner_rect.height // 2)))
        draw_pill(
            surface,
            self._banner_rect,
            self._block_label,
            fonts,
            b_px,
            theme.accent,
            theme.panel_alt,
            theme.accent_muted,
            family=theme.font_family,
        )

        for a0, a1 in self._arrows:
            draw_arrow(surface, a0, a1, theme.accent, thickness=3, head_len=9)

        out_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.15) / 0.85)))
        for card in self._cards:
            vis_cols: int | None = None
            if card.role in (TensorRole.OUTPUT.value, TensorRole.UPDATE.value) and prog < 0.999:
                vis_cols = max(1, math.ceil(card.grid.disp_cols * out_t))
            draw_grid_card(surface, card, fonts, theme, scale, reveal_cols=vis_cols)
