"""Nonlinear activation scene (GELU filter curve + negative tile muting).

Single responsibility: illustrate elementwise activation dimming negatives while passing positives.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from tilevis.core.kinds import StepKind, TensorRole
from tilevis.core.trace import Step, TensorSnapshot
from tilevis.render.animation import ease_out_cubic
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    GridCard,
    HitRegion,
    draw_arrow,
    draw_grid_card,
    draw_panel,
    draw_pill,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


def _gelu(x: float) -> float:
    """Exact GELU activation matching PyTorch F.gelu."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


@SCENES.register(StepKind.ACTIVATION.value)
class ActivationScene(BaseScene):
    """Compares before/after tiles alongside a procedural GELU curve diagram."""

    DEFAULT_DURATION_SEC = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._in_snap: TensorSnapshot | None = None
        self._out_snap: TensorSnapshot | None = None
        self._act_name: str = "GELU"
        self._curve_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._neg_pct_in: float | None = None
        self._muted_pct_out: float | None = None

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = ctx, animate
        self._in_snap = find_snapshot(step, TensorRole.INPUT)
        self._out_snap = find_snapshot(step, TensorRole.OUTPUT)
        self._act_name = str(step.meta.get("activation", "GELU"))

        if self._in_snap is not None and self._in_snap.data.size > 0:
            self._neg_pct_in = float(np.mean(self._in_snap.data < 0.0) * 100.0)
        else:
            self._neg_pct_in = None

        if self._out_snap is not None and self._out_snap.data.size > 0:
            self._muted_pct_out = float(np.mean(np.abs(self._out_snap.data) < 0.1) * 100.0)
        else:
            self._muted_pct_out = None

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        avail_w = max(40, stage.width - 2 * outer_pad)
        avail_h = max(40, stage.height - 2 * outer_pad)
        gap = max(10, min(scale.px(24, minimum=8), avail_w // 18))

        if self._in_snap is not None and self._out_snap is not None:
            curve_w = max(80, min(scale.px(320, minimum=80), int(avail_w * 0.28)))
            grid_card_w = max(30, (avail_w - 2 * gap - curve_w) // 2)
            in_rect = pygame.Rect(stage.x + outer_pad, stage.y + outer_pad, grid_card_w, avail_h)
            self._curve_rect = pygame.Rect(
                in_rect.right + gap, stage.y + outer_pad, curve_w, avail_h
            )
            out_rect = pygame.Rect(
                self._curve_rect.right + gap, stage.y + outer_pad, grid_card_w, avail_h
            )
        elif self._out_snap is not None:
            curve_w = max(100, min(scale.px(400, minimum=100), int(avail_w * 0.38)))
            in_rect = None
            self._curve_rect = pygame.Rect(
                stage.x + outer_pad, stage.y + outer_pad, curve_w, avail_h
            )
            out_rect = pygame.Rect(
                self._curve_rect.right + gap,
                stage.y + outer_pad,
                max(30, avail_w - gap - curve_w),
                avail_h,
            )
        else:
            self._curve_rect = pygame.Rect(
                stage.x + outer_pad, stage.y + outer_pad, avail_w, avail_h
            )
            in_rect = None
            out_rect = None

        self._hit_regions.append(
            HitRegion(
                rect=self._curve_rect,
                tooltip_title=f"{self._act_name} Activation Filter",
                tooltip_lines=(
                    "Negative values dim gently toward zero.",
                    "Positive values pass straight through.",
                ),
            )
        )

        hdr_h = min(max(24, scale.px(48, minimum=24)), max(24, avail_h // 4))

        for rect, snap, title, sub_hint, hi in (
            (in_rect, self._in_snap, "Before Filter", "Mix of + and - values", False),
            (out_rect, self._out_snap, "After Filter", "Negatives dimmed near 0", True),
        ):
            if rect is None or snap is None:
                continue
            pad = max(4, min(scale.px(12, minimum=4), rect.width // 14))
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
            sub = sub_hint + (f" . {grid.downsample_note}" if grid.downsample_note else "")
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

    def _draw_curve_widget(self, surface: pygame.Surface, prog: float, scale: UiScale) -> None:
        if self._ctx is None:
            return
        theme = self._ctx.theme
        fonts = self._ctx.fonts
        r = self._curve_rect
        draw_panel(surface, r, theme, scale, alt=True)

        pad = max(6, min(scale.px(16, minimum=6), r.width // 12))
        t_px = max(11, min(scale.font_px(22, minimum=11), max(11, r.height // 14)))
        s_px = max(9, min(scale.font_px(16, minimum=9), t_px - 2))

        title = fonts.render_text(
            f"{self._act_name} Filter",
            t_px,
            theme.accent,
            bold=True,
            family=theme.font_family,
            max_width=max(20, r.width - 2 * pad),
        )
        surface.blit(title, (r.x + max(4, (r.width - title.get_width()) // 2), r.y + pad))

        # Plot box in middle of _curve_rect
        badge_h = max(24, scale.px(44, minimum=22))
        plot_top = r.y + pad + title.get_height() + max(6, scale.px(12))
        plot_h = max(24, r.bottom - pad - badge_h - 8 - plot_top)
        plot = pygame.Rect(r.x + pad, plot_top, max(20, r.width - 2 * pad), plot_h)
        pygame.draw.rect(surface, theme.background, plot, border_radius=max(2, scale.px(6)))
        pygame.draw.rect(surface, theme.border, plot, width=1, border_radius=max(2, scale.px(6)))

        # Map x in [-3.5, 3.5], y in [-0.6, 3.5] into plot rectangle
        x_min, x_max = -3.5, 3.5
        y_min, y_max = -0.6, 3.5

        def _to_px(xv: float, yv: float) -> tuple[int, int]:
            nx = (xv - x_min) / (x_max - x_min)
            ny = 1.0 - (yv - y_min) / (y_max - y_min)
            return (
                round(plot.x + 6 + nx * max(4, plot.width - 12)),
                round(plot.y + 6 + ny * max(4, plot.height - 12)),
            )

        # Axes x=0 and y=0
        x0_top = _to_px(0.0, y_max)
        x0_bot = _to_px(0.0, y_min)
        y0_left = _to_px(x_min, 0.0)
        y0_right = _to_px(x_max, 0.0)
        pygame.draw.line(surface, theme.border, x0_top, x0_bot, 1)
        pygame.draw.line(surface, theme.border, y0_left, y0_right, 1)

        # Highlight negative muted half gently on left of x=0
        zero_x = x0_top[0]
        if zero_x > plot.x + 2:
            neg_zone = pygame.Rect(plot.x + 1, plot.y + 1, zero_x - plot.x - 1, plot.height - 2)
            pygame.draw.rect(surface, theme.panel, neg_zone)
            pygame.draw.line(surface, theme.border, y0_left, (zero_x, y0_left[1]), 1)

        # Procedural GELU curve polyline revealed smoothly across prog
        n_pts = 48
        reveal_pts = max(2, round(n_pts * ease_out_cubic(prog)))
        pts: list[tuple[int, int]] = []
        for i in range(reveal_pts + 1):
            xv = x_min + (x_max - x_min) * (i / n_pts)
            pts.append(_to_px(xv, _gelu(xv)))
        if len(pts) >= 2:
            pygame.draw.lines(surface, theme.accent, False, pts, max(2, scale.px(3)))
            pygame.draw.aalines(surface, theme.accent, False, pts)
            if prog < 0.999:
                pygame.draw.circle(surface, theme.highlight, pts[-1], max(3, scale.px(5)))

        # Bottom intuitive summary badge
        badge_rect = pygame.Rect(r.x + pad, plot.bottom + 8, max(20, r.width - 2 * pad), badge_h)
        if self._muted_pct_out is not None:
            summary = f"{self._muted_pct_out:.0f}% tiles quieted near 0"
        else:
            summary = "Neg -> ~0   |   Pos -> bright"
        draw_pill(
            surface,
            badge_rect,
            summary,
            fonts,
            s_px,
            theme.text_primary,
            theme.panel,
            theme.border,
            bold=False,
            family=theme.font_family,
        )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render before/after activation grids and procedural GELU curve widget."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        self._draw_curve_widget(surface, prog, scale)

        filter_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.15) / 0.85)))
        for card in self._cards:
            vis_cols: int | None = None
            if card.role == TensorRole.OUTPUT.value and prog < 0.999:
                vis_cols = max(1, math.ceil(card.grid.disp_cols * filter_t))
            draw_grid_card(surface, card, fonts, theme, scale, reveal_cols=vis_cols)

            if card.role == TensorRole.INPUT.value:
                draw_arrow(
                    surface,
                    (card.rect.right + 2, card.rect.centery),
                    (self._curve_rect.x - 2, card.rect.centery),
                    theme.accent,
                    thickness=2,
                    head_len=8,
                )
            elif card.role == TensorRole.OUTPUT.value:
                draw_arrow(
                    surface,
                    (self._curve_rect.right + 2, card.rect.centery),
                    (card.rect.x - 2, card.rect.centery),
                    theme.accent,
                    thickness=2,
                    head_len=8,
                )
