"""Residual highway add scene showing shortcut + update combining into output.

Single responsibility: visualize residual bypass highway stream merging with layer update.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math

import pygame

from tilevis.core.kinds import Semantic, StepKind, TensorRole
from tilevis.core.trace import Step, TensorSnapshot
from tilevis.render.animation import ease_out_cubic
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    GridCard,
    draw_arrow,
    draw_grid_card,
    draw_operator_symbol,
    draw_pill,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


@SCENES.register(StepKind.RESIDUAL_ADD.value)
class ResidualScene(BaseScene):
    """Visualizes highway bypass (shortcut) + layer update sliding together into output."""

    DEFAULT_DURATION_SEC = 1.05

    def __init__(self) -> None:
        super().__init__()
        self._sc_snap: TensorSnapshot | None = None
        self._up_snap: TensorSnapshot | None = None
        self._out_snap: TensorSnapshot | None = None
        self._sc_card: GridCard | None = None
        self._up_card: GridCard | None = None
        self._out_card: GridCard | None = None
        self._up_rest_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._up_grid_rest: tuple[int, int] = (0, 0)
        self._plus_center: tuple[int, int] = (0, 0)
        self._eq_center: tuple[int, int] = (0, 0)
        self._highway_label: str = "Highway Stream (Shortcut)"
        self._update_label: str = "Layer Update"

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = ctx, animate
        self._sc_snap = find_snapshot(step, TensorRole.SHORTCUT)
        self._up_snap = find_snapshot(step, TensorRole.UPDATE)
        self._out_snap = find_snapshot(step, TensorRole.OUTPUT)
        if step.semantic == Semantic.BLOCK_RESID1.value:
            self._update_label = "Attention Update"
        elif step.semantic == Semantic.BLOCK_RESID2.value:
            self._update_label = "MLP Thinking Update"
        else:
            self._update_label = "Layer Update"

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._sc_card = None
        self._up_card = None
        self._out_card = None
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        avail_w = max(40, stage.width - 2 * outer_pad)
        avail_h = max(40, stage.height - 2 * outer_pad)

        gap_x = max(24, min(scale.px(56, minimum=18), avail_w // 12))
        gap_y = max(10, scale.px(20, minimum=6))

        left_w = max(30, int((avail_w - gap_x) * 0.48))
        right_w = max(30, avail_w - gap_x - left_w)
        lane_h = max(32, (avail_h - gap_y) // 2)

        sc_rect = pygame.Rect(stage.x + outer_pad, stage.y + outer_pad, left_w, lane_h)
        up_rect = pygame.Rect(stage.x + outer_pad, sc_rect.bottom + gap_y, left_w, lane_h)
        out_h = max(40, int(avail_h * 0.62))
        out_rect = pygame.Rect(
            sc_rect.right + gap_x,
            stage.y + outer_pad + (avail_h - out_h) // 2,
            right_w,
            out_h,
        )

        self._plus_center = (sc_rect.right + gap_x // 3, out_rect.centery)
        self._eq_center = (out_rect.x - gap_x // 3, out_rect.centery)
        self._up_rest_rect = up_rect

        hdr_small = min(max(22, scale.px(42, minimum=20)), max(20, lane_h // 2))
        hdr_large = min(max(24, scale.px(48, minimum=24)), max(24, out_h // 4))

        def _build(
            rect: pygame.Rect,
            snap: TensorSnapshot | None,
            title: str,
            sub: str,
            hdr_h: int,
            *,
            hi: bool = False,
        ) -> GridCard | None:
            if snap is None:
                return None
            pad = max(4, min(scale.px(12, minimum=4), rect.width // 16))
            gw = max(1, rect.width - 2 * pad)
            gh = max(1, rect.height - hdr_h - 2 * pad)
            grid = make_grid(
                snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(18, minimum=6),
                min_cell_px=max(1, min(6, gw, gh)),
            )
            place_grid_in_card(rect, grid, hdr_h, pad)
            if grid.downsample_note:
                self.downsample_notes.append(grid.downsample_note)
            s_txt = sub + (f" . {grid.downsample_note}" if grid.downsample_note else "")
            card = GridCard(
                title=title,
                subtitle=s_txt,
                rect=rect,
                grid=grid,
                tooltip_title=title,
                role=snap.role,
                highlighted=hi,
            )
            self._cards.append(card)
            return card

        self._sc_card = _build(
            sc_rect,
            self._sc_snap,
            self._highway_label,
            "Memory untouched by this sublayer",
            hdr_small,
            hi=True,
        )
        self._up_card = _build(
            up_rect,
            self._up_snap,
            self._update_label,
            "New details computed by layer",
            hdr_small,
        )
        if self._up_card is not None:
            self._up_grid_rest = self._up_card.grid.rect.topleft
        self._out_card = _build(
            out_rect,
            self._out_snap,
            "Combined Output",
            "Highway + Update merged",
            hdr_large,
            hi=True,
        )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render highway stream + sliding update merging into combined output."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        # Top highway banner pill across top gap if room
        if self._sc_card is not None:
            # Draw highway arrow rail from top card to + junction
            start_pt = (self._sc_card.rect.right + 2, self._sc_card.rect.centery)
            draw_arrow(surface, start_pt, self._plus_center, theme.accent, thickness=3, head_len=9)
            draw_grid_card(surface, self._sc_card, fonts, theme, scale)

        # Sliding update card eases into its resting lane
        if self._up_card is not None:
            slide_p = ease_out_cubic(min(1.0, prog / 0.55))
            offset_y = round(
                (1.0 - slide_p)
                * min(scale.px(24, minimum=6), max(4, stage.bottom - self._up_rest_rect.bottom - 2))
            )
            self._up_card.rect = self._up_rest_rect.move(0, offset_y)
            self._up_card.grid.set_position(self._up_grid_rest[0], self._up_grid_rest[1] + offset_y)

            draw_arrow(
                surface,
                (self._up_card.rect.right + 2, self._up_card.rect.centery),
                self._plus_center,
                theme.highlight,
                thickness=3,
                head_len=9,
            )
            draw_grid_card(surface, self._up_card, fonts, theme, scale)

        sym_r = max(6, scale.px(11, minimum=6))
        draw_operator_symbol(surface, "+", self._plus_center, sym_r, theme.accent, thickness=3)
        if self._out_card is not None:
            draw_arrow(
                surface,
                (self._plus_center[0] + sym_r + 4, self._plus_center[1]),
                (self._out_card.rect.x - 2, self._out_card.rect.centery),
                theme.success,
                thickness=3,
                head_len=9,
            )
            out_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.4) / 0.6)))
            vis_cols = (
                max(1, math.ceil(self._out_card.grid.disp_cols * out_t)) if prog < 0.999 else None
            )
            draw_grid_card(surface, self._out_card, fonts, theme, scale, reveal_cols=vis_cols)

            # Compact highway badge atop output card
            b_w = min(self._out_card.rect.width - 16, scale.px(220, minimum=90))
            b_h = max(20, scale.px(28, minimum=18))
            badge_r = pygame.Rect(
                self._out_card.rect.right - b_w - 8,
                max(stage.y + 4, self._out_card.rect.y - b_h - 6),
                b_w,
                b_h,
            )
            draw_pill(
                surface,
                badge_r,
                "Old memory kept intact!",
                fonts,
                max(9, scale.font_px(15, minimum=9)),
                theme.background,
                theme.success,
                family=theme.font_family,
            )
