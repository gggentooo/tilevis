"""Attention mix scene blending Value (V) rows by spotlight weights into context vectors.

Single responsibility: visualize attn_weights x V -> context vectors blend step.
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
    draw_grid_card,
    draw_operator_symbol,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


@SCENES.register(StepKind.ATTENTION_MIX.value)
class AttentionMixScene(BaseScene):
    """Shows spotlight weights blending Content (V) rows into new context vectors."""

    DEFAULT_DURATION_SEC = 1.05

    def __init__(self) -> None:
        super().__init__()
        self._weights_snap: TensorSnapshot | None = None
        self._v_snap: TensorSnapshot | None = None
        self._out_snap: TensorSnapshot | None = None
        self._op_centers: list[tuple[tuple[int, int], str]] = []

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = ctx, animate
        self._weights_snap = find_snapshot(step, TensorRole.EXTRA) or find_snapshot(
            step, TensorRole.ATTN_WEIGHTS
        )
        self._v_snap = find_snapshot(step, TensorRole.VALUES)
        self._out_snap = find_snapshot(step, TensorRole.OUTPUT)

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._op_centers = []
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        panels: list[tuple[str, str, TensorSnapshot, bool]] = []
        if self._weights_snap is not None:
            panels.append(("Spotlight Weights", "How much of each word", self._weights_snap, False))
        if self._v_snap is not None:
            panels.append(("Content (V)", "Word value rows", self._v_snap, False))
        if self._out_snap is not None:
            panels.append(("Blended Words", "Weighted mix output", self._out_snap, True))
        if not panels:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        avail_w = max(30, stage.width - 2 * outer_pad)
        avail_h = max(36, stage.height - 2 * outer_pad)
        op_gap = max(14, min(scale.px(34, minimum=12), avail_w // 18))

        n = len(panels)
        card_w = max(20, (avail_w - op_gap * (n - 1)) // n)
        hdr_h = min(max(24, scale.px(48, minimum=24)), max(24, avail_h // 4))
        inner_pad = max(4, min(scale.px(12, minimum=4), card_w // 14))

        for i, (title, hint, snap, hi) in enumerate(panels):
            cx = stage.x + outer_pad + i * (card_w + op_gap)
            rect = pygame.Rect(cx, stage.y + outer_pad, card_w, avail_h)
            if i > 0:
                kind = "x" if (n == 3 and i == 1) else "="
                self._op_centers.append(((cx - op_gap // 2, rect.centery), kind))

            is_weights = snap.role in (TensorRole.EXTRA.value, TensorRole.ATTN_WEIGHTS.value)
            gw = max(1, rect.width - 2 * inner_pad)
            gh = max(1, rect.height - hdr_h - 2 * inner_pad)
            grid = make_grid(
                snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(20, minimum=6),
                min_cell_px=max(1, min(6, gw, gh)),
                signed=(not is_weights),
            )
            place_grid_in_card(rect, grid, hdr_h, inner_pad)
            if grid.downsample_note:
                self.downsample_notes.append(grid.downsample_note)

            shape_txt = "x".join(str(s) for s in snap.full_shape)
            sub = f"{hint} ({shape_txt})"
            if grid.downsample_note:
                sub += f" . {grid.downsample_note}"
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
        """Render attention weights blending V rows into mixed output rows."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        sym_r = max(6, scale.px(11, minimum=6))
        for center, kind in self._op_centers:
            draw_operator_symbol(surface, kind, center, sym_r, theme.accent, thickness=3)

        blend_t = ease_out_cubic(min(1.0, max(0.0, prog)))
        for idx, card in enumerate(self._cards):
            vis_rows: int | None = None
            hi_rows: tuple[int, ...] = ()
            is_last = idx == len(self._cards) - 1
            if prog < 0.999 and card.grid.disp_rows > 0:
                row_idx = min(
                    card.grid.disp_rows - 1,
                    int(blend_t * card.grid.disp_rows),
                )
                if is_last:
                    vis_rows = max(1, math.ceil(card.grid.disp_rows * blend_t))
                    hi_rows = (min(card.grid.disp_rows - 1, vis_rows - 1),)
                elif card.role != TensorRole.VALUES.value:
                    hi_rows = (row_idx,)

            draw_grid_card(
                surface,
                card,
                fonts,
                theme,
                scale,
                reveal_rows=vis_rows,
                highlight_rows=hi_rows,
            )
