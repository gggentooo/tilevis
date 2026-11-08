"""LayerNorm scene illustrating spread compression and learned scale/shift weights.

Single responsibility: show before/after normalization squeeze, spread indicator, scale & shift.
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
    HitRegion,
    blend_rgb,
    draw_arrow,
    draw_grid_card,
    draw_panel,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


@SCENES.register(StepKind.LAYERNORM.value)
class LayerNormScene(BaseScene):
    """Visualizes noisy inputs squeezed into calm range alongside scale and shift strips."""

    DEFAULT_DURATION_SEC = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._in_snap: TensorSnapshot | None = None
        self._out_snap: TensorSnapshot | None = None
        self._scale_snap: TensorSnapshot | None = None
        self._shift_snap: TensorSnapshot | None = None
        self._spread_before_rect: pygame.Rect | None = None
        self._spread_after_rect: pygame.Rect | None = None
        self._arrow_starts: list[tuple[int, int]] = []
        self._arrow_ends: list[tuple[int, int]] = []

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = ctx, animate
        self._in_snap = find_snapshot(step, TensorRole.INPUT)
        self._out_snap = find_snapshot(step, TensorRole.OUTPUT)
        self._scale_snap = find_snapshot(step, TensorRole.SCALE)
        self._shift_snap = find_snapshot(step, TensorRole.SHIFT)

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._spread_before_rect = None
        self._spread_after_rect = None
        self._arrow_starts = []
        self._arrow_ends = []
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(24, minimum=6), stage.width // 24, stage.height // 24))
        gap = max(8, scale.px(20, minimum=6))
        avail_w = max(30, stage.width - 2 * outer_pad)
        avail_h = max(30, stage.height - 2 * outer_pad)
        meter_h = max(28, min(scale.px(54, minimum=24), avail_h // 7))

        # 3-column layout if input+output present: [Before] -> [Scale/Shift strips] -> [After]
        has_in = self._in_snap is not None
        has_mid = self._scale_snap is not None or self._shift_snap is not None

        if has_in and has_mid:
            side_w = max(20, int((avail_w - 2 * gap) * 0.36))
            mid_w = max(20, avail_w - 2 * gap - 2 * side_w)
            in_rect = pygame.Rect(
                stage.x + outer_pad, stage.y + outer_pad, side_w, avail_h - meter_h - 8
            )
            mid_rect = pygame.Rect(in_rect.right + gap, stage.y + outer_pad, mid_w, avail_h)
            out_rect = pygame.Rect(
                mid_rect.right + gap, stage.y + outer_pad, side_w, avail_h - meter_h - 8
            )
            self._spread_before_rect = pygame.Rect(in_rect.x, in_rect.bottom + 6, side_w, meter_h)
            self._spread_after_rect = pygame.Rect(out_rect.x, out_rect.bottom + 6, side_w, meter_h)
        elif has_in:
            side_w = max(20, (avail_w - gap) // 2)
            in_rect = pygame.Rect(
                stage.x + outer_pad, stage.y + outer_pad, side_w, avail_h - meter_h - 8
            )
            out_rect = pygame.Rect(
                in_rect.right + gap, stage.y + outer_pad, side_w, avail_h - meter_h - 8
            )
            mid_rect = None
            self._spread_before_rect = pygame.Rect(in_rect.x, in_rect.bottom + 6, side_w, meter_h)
            self._spread_after_rect = pygame.Rect(out_rect.x, out_rect.bottom + 6, side_w, meter_h)
        else:
            out_rect = pygame.Rect(
                stage.x + outer_pad, stage.y + outer_pad, avail_w, avail_h - meter_h - 8
            )
            in_rect = None
            mid_rect = None
            self._spread_after_rect = pygame.Rect(out_rect.x, out_rect.bottom + 6, avail_w, meter_h)

        header_h = min(max(24, scale.px(48, minimum=24)), max(24, out_rect.height // 4))
        inner_pad = max(4, min(scale.px(12, minimum=4), out_rect.width // 14))

        if self._in_snap is not None and in_rect is not None:
            gw = max(1, in_rect.width - 2 * inner_pad)
            gh = max(1, in_rect.height - header_h - 2 * inner_pad)
            grid_in = make_grid(
                self._in_snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(20, minimum=8),
                min_cell_px=max(1, min(8, gw, gh)),
            )
            place_grid_in_card(in_rect, grid_in, header_h, inner_pad)
            if grid_in.downsample_note:
                self.downsample_notes.append(grid_in.downsample_note)
            in_sub = f"std={self._in_snap.stats.std:.2f} . max={self._in_snap.stats.abs_max:.2f}"
            self._cards.append(
                GridCard(
                    title="Before (Raw Spread)",
                    subtitle=in_sub,
                    rect=in_rect,
                    grid=grid_in,
                    tooltip_title="LayerNorm Input",
                    role=self._in_snap.role,
                )
            )
            if self._spread_before_rect is not None:
                in_min = self._in_snap.stats.min
                in_max = self._in_snap.stats.max
                self._hit_regions.append(
                    HitRegion(
                        rect=self._spread_before_rect,
                        tooltip_title="Spread Before Balancing",
                        tooltip_lines=(
                            f"std: {self._in_snap.stats.std:.3f}",
                            f"range: [{in_min:+.2f}, {in_max:+.2f}]",
                        ),
                    )
                )

        if mid_rect is not None:
            params: list[tuple[str, TensorSnapshot]] = []
            if self._scale_snap is not None:
                params.append(("Scale (gamma)", self._scale_snap))
            if self._shift_snap is not None:
                params.append(("Shift (beta)", self._shift_snap))

            p_gap = max(8, scale.px(12, minimum=4))
            p_h = max(
                36,
                min(
                    scale.px(130, minimum=36),
                    (mid_rect.height - p_gap * (len(params) + 1)) // max(1, len(params)),
                ),
            )
            start_y = mid_rect.y + max(
                0, (mid_rect.height - (len(params) * p_h + max(0, len(params) - 1) * p_gap)) // 2
            )
            for idx, (lbl, p_snap) in enumerate(params):
                pr = pygame.Rect(mid_rect.x, start_y + idx * (p_h + p_gap), mid_rect.width, p_h)
                p_hdr = min(max(20, scale.px(38, minimum=20)), max(20, pr.height // 2))
                p_pad = max(4, min(scale.px(10, minimum=4), pr.width // 16))
                pgw = max(1, pr.width - 2 * p_pad)
                pgh = max(1, pr.height - p_hdr - 2 * p_pad)
                p_grid = make_grid(
                    p_snap,
                    step,
                    ctx,
                    (pgw, pgh),
                    cell_px=scale.px(18, minimum=8),
                    min_cell_px=max(1, min(8, pgw, pgh)),
                )
                place_grid_in_card(pr, p_grid, p_hdr, p_pad)
                self._cards.append(
                    GridCard(
                        title=lbl,
                        subtitle=f"mean={p_snap.stats.mean:+.2f}",
                        rect=pr,
                        grid=p_grid,
                        tooltip_title=lbl,
                        role=p_snap.role,
                    )
                )
            if in_rect is not None:
                self._arrow_starts = [
                    (in_rect.right + 2, in_rect.centery),
                    (mid_rect.right + 2, out_rect.centery),
                ]
                self._arrow_ends = [
                    (mid_rect.x - 2, in_rect.centery),
                    (out_rect.x - 2, out_rect.centery),
                ]
        elif in_rect is not None:
            self._arrow_starts = [(in_rect.right + 2, in_rect.centery)]
            self._arrow_ends = [(out_rect.x - 2, out_rect.centery)]

        if self._out_snap is not None:
            gw = max(1, out_rect.width - 2 * inner_pad)
            gh = max(1, out_rect.height - header_h - 2 * inner_pad)
            grid_out = make_grid(
                self._out_snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(20, minimum=8),
                min_cell_px=max(1, min(8, gw, gh)),
            )
            place_grid_in_card(out_rect, grid_out, header_h, inner_pad)
            if grid_out.downsample_note:
                self.downsample_notes.append(grid_out.downsample_note)
            out_sub = f"std={self._out_snap.stats.std:.2f} . max={self._out_snap.stats.abs_max:.2f}"
            self._cards.append(
                GridCard(
                    title="After (Calm & Balanced)",
                    subtitle=out_sub,
                    rect=out_rect,
                    grid=grid_out,
                    tooltip_title="LayerNorm Output",
                    role=self._out_snap.role,
                    highlighted=True,
                )
            )
            if self._spread_after_rect is not None:
                out_min = self._out_snap.stats.min
                out_max = self._out_snap.stats.max
                self._hit_regions.append(
                    HitRegion(
                        rect=self._spread_after_rect,
                        tooltip_title="Spread After Balancing",
                        tooltip_lines=(
                            f"std: {self._out_snap.stats.std:.3f}",
                            f"range: [{out_min:+.2f}, {out_max:+.2f}]",
                        ),
                    )
                )

    def _draw_spread_meter(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        std_val: float,
        ref_max_std: float,
        bar_color: tuple[int, int, int],
        scale: UiScale,
    ) -> None:
        if self._ctx is None:
            return
        theme = self._ctx.theme
        fonts = self._ctx.fonts
        draw_panel(surface, rect, theme, scale, alt=True)
        pad = max(6, scale.px(10, minimum=4))
        px = max(9, min(scale.font_px(18, minimum=9), rect.height // 2))
        txt_surf = fonts.render_text(
            label,
            px,
            theme.text_secondary,
            family=theme.font_family,
            max_width=max(20, rect.width // 2),
        )
        surface.blit(txt_surf, (rect.x + pad, rect.centery - txt_surf.get_height() // 2))

        track_x = rect.x + pad + txt_surf.get_width() + pad
        track_w = max(12, rect.right - pad - track_x)
        track_h = max(6, min(14, rect.height // 3))
        track_rect = pygame.Rect(track_x, rect.centery - track_h // 2, track_w, track_h)
        pygame.draw.rect(surface, theme.background, track_rect, border_radius=max(2, track_h // 2))

        ratio = max(0.06, min(1.0, std_val / max(1e-6, ref_max_std)))
        fill_w = max(4, round(track_w * ratio))
        pygame.draw.rect(
            surface,
            bar_color,
            pygame.Rect(track_x, track_rect.y, fill_w, track_h),
            border_radius=max(2, track_h // 2),
        )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render normalization comparison and squeeze indicators."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress
        ease_p = ease_out_cubic(prog)

        for s_pt, e_pt in zip(self._arrow_starts, self._arrow_ends, strict=False):
            if e_pt[0] > s_pt[0] + 4:
                draw_arrow(surface, s_pt, e_pt, theme.accent, thickness=2, head_len=8)

        for card in self._cards:
            vis_cols: int | None = None
            if card.role == TensorRole.OUTPUT.value and prog < 0.999:
                vis_cols = max(
                    1, math.ceil(card.grid.disp_cols * ease_out_cubic(max(0.0, (prog - 0.2) / 0.8)))
                )
            draw_grid_card(surface, card, fonts, theme, scale, reveal_cols=vis_cols)

        ref_std = max(
            1e-4,
            self._in_snap.stats.std if self._in_snap is not None else 0.0,
            self._out_snap.stats.std if self._out_snap is not None else 1.0,
        )
        if self._spread_before_rect is not None and self._in_snap is not None:
            self._draw_spread_meter(
                surface,
                self._spread_before_rect,
                f"Before spread (std={self._in_snap.stats.std:.2f})",
                self._in_snap.stats.std,
                ref_std,
                theme.warning,
                scale,
            )
        if self._spread_after_rect is not None and self._out_snap is not None:
            start_std = (
                self._in_snap.stats.std if self._in_snap is not None else self._out_snap.stats.std
            )
            disp_std = start_std * (1.0 - ease_p) + self._out_snap.stats.std * ease_p
            bar_col = blend_rgb(theme.warning, theme.success, ease_p)
            self._draw_spread_meter(
                surface,
                self._spread_after_rect,
                f"After spread (std={self._out_snap.stats.std:.2f})",
                disp_std,
                ref_std,
                bar_col,
                scale,
            )
