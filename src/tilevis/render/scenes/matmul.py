"""Core linear matrix-multiplication scene (input x weights (+ bias) -> output).

Single responsibility: animate input x weight -> output with synchronized column sweep.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from tilevis.core.kinds import Semantic, StepKind, TensorRole
from tilevis.core.trace import Step, TensorSnapshot
from tilevis.render.animation import ease_in_out_quad
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    GridCard,
    HitRegion,
    draw_grid_card,
    draw_operator_symbol,
    draw_panel,
    draw_pill,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


@dataclass(slots=True)
class _MatmulOp:
    center: tuple[int, int]
    kind: str


@SCENES.register(StepKind.LINEAR.value)
class MatmulScene(BaseScene):
    """Core scene visualizing input (T x in) x weights (in x out) -> output (T x out)."""

    DEFAULT_DURATION_SEC = 1.25

    def __init__(self) -> None:
        super().__init__()
        self._in_snap: TensorSnapshot | None = None
        self._w_snap: TensorSnapshot | None = None
        self._bias_snap: TensorSnapshot | None = None
        self._out_snap: TensorSnapshot | None = None
        self._in_card: GridCard | None = None
        self._w_card: GridCard | None = None
        self._bias_card: GridCard | None = None
        self._out_card: GridCard | None = None
        self._ops: list[_MatmulOp] = []
        self._banner_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._placeholder_rect: pygame.Rect | None = None
        self._banner_text: str = ""
        self.displayed_in_dim: int | None = None
        self.displayed_out_dim: int | None = None
        self.displayed_tokens: int | None = None

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = ctx, animate
        self._in_snap = find_snapshot(step, TensorRole.INPUT)
        self._w_snap = find_snapshot(step, TensorRole.WEIGHT)
        self._bias_snap = find_snapshot(step, TensorRole.BIAS)
        self._out_snap = find_snapshot(step, TensorRole.OUTPUT)

        # Infer exact (T, in_dim, out_dim) dynamically across real and synthetic traces
        t_cnt: int | None = None
        in_d: int | None = None
        out_d: int | None = None
        if self._in_snap is not None and len(self._in_snap.full_shape) == 2:
            t_cnt = int(self._in_snap.full_shape[0])
            in_d = int(self._in_snap.full_shape[1])
        if self._w_snap is not None and len(self._w_snap.full_shape) == 2:
            in_d = in_d or int(self._w_snap.full_shape[0])
            out_d = int(self._w_snap.full_shape[1])
        if self._out_snap is not None and len(self._out_snap.full_shape) == 2:
            t_cnt = t_cnt or int(self._out_snap.full_shape[0])
            out_d = out_d or int(self._out_snap.full_shape[1])

        self.displayed_tokens = t_cnt
        self.displayed_in_dim = in_d
        self.displayed_out_dim = out_d

        sem = step.semantic
        in_lbl = str(in_d) if in_d is not None else "?"
        out_lbl = str(out_d) if out_d is not None else "?"
        if sem == Semantic.BLOCK_FF_EXPAND.value:
            ratio = (
                f"{out_d // in_d}x"
                if (in_d and out_d and out_d % in_d == 0)
                else (f"{out_d / in_d:.1f}x" if in_d and out_d else "wider")
            )
            self._banner_text = f"ff.expand: makes the list {ratio} longer ({in_lbl} to {out_lbl})"
        elif sem == Semantic.BLOCK_FF_CONTRACT.value:
            self._banner_text = f"ff.contract: back to normal size ({in_lbl} to {out_lbl})"
        elif sem == Semantic.BLOCK_ATT_OUT.value:
            self._banner_text = (
                f"att.out: stitches all spotlight heads together ({in_lbl} to {out_lbl})"
            )
        else:
            self._banner_text = f"Linear transform: {in_lbl} features to {out_lbl} features"

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._in_card = None
        self._w_card = None
        self._bias_card = None
        self._out_card = None
        self._ops = []
        self._placeholder_rect = None
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
            min(max(120, stage.width - 2 * outer_pad), scale.px(780, minimum=120)),
            banner_h,
        )

        top_y = self._banner_rect.bottom + max(6, scale.px(12, minimum=4))
        body_h = max(36, stage.bottom - outer_pad // 2 - top_y)
        avail_w = max(30, stage.width - 2 * outer_pad)

        # Build 3 main slots: Input x Weights = Output (plus optional slim Bias strip under Weights)
        has_in = self._in_snap is not None
        has_w = self._w_snap is not None
        has_bias = self._bias_snap is not None
        has_out = self._out_snap is not None

        op_gap = max(14, min(scale.px(34, minimum=12), avail_w // 18))
        # Always reserve slot 0 if input exists or show compact card when omitted in fake traces
        show_placeholder_in = (not has_in) and has_w and has_out
        in_rect: pygame.Rect | None = None
        w_rect: pygame.Rect | None = None
        out_rect: pygame.Rect | None = None

        if has_in and has_w and has_out:
            in_w = max(20, int((avail_w - 2 * op_gap) * 0.28))
            w_w = max(20, int((avail_w - 2 * op_gap) * 0.42))
            out_w = max(20, avail_w - 2 * op_gap - in_w - w_w)
            in_rect = pygame.Rect(stage.x + outer_pad, top_y, in_w, body_h)
            w_rect = pygame.Rect(in_rect.right + op_gap, top_y, w_w, body_h)
            out_rect = pygame.Rect(w_rect.right + op_gap, top_y, out_w, body_h)
            self._ops = [
                _MatmulOp((in_rect.right + op_gap // 2, top_y + body_h // 2), "x"),
                _MatmulOp((w_rect.right + op_gap // 2, top_y + body_h // 2), "="),
            ]
        elif show_placeholder_in:
            in_w = max(20, int((avail_w - 2 * op_gap) * 0.16))
            w_w = max(20, int((avail_w - 2 * op_gap) * 0.46))
            out_w = max(20, avail_w - 2 * op_gap - in_w - w_w)
            in_rect = pygame.Rect(stage.x + outer_pad, top_y, in_w, body_h)
            w_rect = pygame.Rect(in_rect.right + op_gap, top_y, w_w, body_h)
            out_rect = pygame.Rect(w_rect.right + op_gap, top_y, out_w, body_h)
            self._placeholder_rect = in_rect
            self._ops = [
                _MatmulOp((in_rect.right + op_gap // 2, top_y + body_h // 2), "x"),
                _MatmulOp((w_rect.right + op_gap // 2, top_y + body_h // 2), "="),
            ]
        else:
            # Generic 1-or-2 active snapshots layout
            active = [s for s in (self._in_snap, self._w_snap, self._out_snap) if s is not None]
            n = max(1, len(active))
            cw = max(20, (avail_w - op_gap * (n - 1)) // n)
            rects = [
                pygame.Rect(stage.x + outer_pad + i * (cw + op_gap), top_y, cw, body_h)
                for i in range(n)
            ]
            in_rect = rects[0] if has_in else None
            w_rect = rects[1 if has_in else 0] if has_w else None
            out_rect = rects[-1] if has_out else None
            if n >= 2:
                self._ops = [_MatmulOp((rects[0].right + op_gap // 2, top_y + body_h // 2), "=")]

        bias_reserve_h = (
            max(36, scale.px(64, minimum=34))
            if (has_bias and w_rect is not None and body_h > 140)
            else 0
        )
        if w_rect is not None and bias_reserve_h > 0:
            bias_rect = pygame.Rect(
                w_rect.x, w_rect.bottom - bias_reserve_h, w_rect.width, bias_reserve_h
            )
            w_rect = pygame.Rect(
                w_rect.x, w_rect.y, w_rect.width, max(36, w_rect.height - bias_reserve_h - 6)
            )
        else:
            bias_rect = None

        header_h = min(max(24, scale.px(48, minimum=24)), max(24, body_h // 4))

        def _build_card(
            rect: pygame.Rect,
            snap: TensorSnapshot,
            title: str,
            hint: str,
            *,
            highlighted: bool = False,
            hdr_h: int = header_h,
        ) -> GridCard:
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
            shape_txt = "x".join(str(s) for s in snap.full_shape)
            sub = f"{hint} ({shape_txt})"
            if grid.downsample_note:
                sub += f" . {grid.downsample_note}"
            card = GridCard(
                title=title,
                subtitle=sub,
                rect=rect,
                grid=grid,
                tooltip_title=title,
                role=snap.role,
                highlighted=highlighted,
            )
            self._cards.append(card)
            return card

        if self._in_snap is not None and in_rect is not None:
            self._in_card = _build_card(in_rect, self._in_snap, "Input Vectors", "T x in")
        elif self._placeholder_rect is not None:
            t_lbl = self.displayed_tokens or "?"
            d_lbl = self.displayed_in_dim or "?"
            self._hit_regions.append(
                HitRegion(
                    rect=self._placeholder_rect,
                    tooltip_title="Input Vectors",
                    tooltip_lines=(
                        f"Incoming vector shape: ({t_lbl} x {d_lbl})",
                        "Snapshot omitted in minimal trace.",
                    ),
                )
            )

        if self._w_snap is not None and w_rect is not None:
            self._w_card = _build_card(w_rect, self._w_snap, "Learned Weights", "in x out")

        if self._bias_snap is not None and bias_rect is not None:
            b_hdr = min(max(18, scale.px(26, minimum=18)), max(18, bias_rect.height // 2))
            self._bias_card = _build_card(
                bias_rect,
                self._bias_snap,
                "+ Bias",
                "1 x out",
                hdr_h=b_hdr,
            )

        if self._out_snap is not None and out_rect is not None:
            self._out_card = _build_card(
                out_rect,
                self._out_snap,
                "Output Vectors",
                "T x out",
                highlighted=True,
            )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render input x weight (+ bias) -> output with synchronized weight-column sweep."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        # Top semantic banner
        b_px = max(10, min(scale.font_px(20, minimum=10), max(10, self._banner_rect.height // 2)))
        draw_pill(
            surface,
            self._banner_rect,
            self._banner_text,
            fonts,
            b_px,
            theme.accent,
            theme.panel_alt,
            theme.accent_muted,
            family=theme.font_family,
        )

        # Geometric matrix operators 'x' and '='
        sym_r = max(6, scale.px(11, minimum=6))
        for op in self._ops:
            draw_operator_symbol(surface, op.kind, op.center, sym_r, theme.accent, thickness=3)

        # Placeholder input card when synthetic minimal trace omits INPUT tensor
        if self._placeholder_rect is not None:
            r = self._placeholder_rect
            draw_panel(surface, r, theme, scale, alt=True)
            txt_px = max(10, min(scale.font_px(18, minimum=10), max(10, r.width // 8)))
            t_s = fonts.render_text(
                "Input Vectors",
                txt_px,
                theme.text_primary,
                bold=True,
                family=theme.font_family,
                max_width=max(12, r.width - 12),
            )
            dim_s = fonts.render_text(
                f"{self.displayed_tokens or '?'} x {self.displayed_in_dim or '?'}",
                max(9, txt_px - 2),
                theme.accent,
                family=theme.font_family,
                max_width=max(12, r.width - 12),
            )
            surface.blit(t_s, (r.centerx - t_s.get_width() // 2, r.centery - t_s.get_height() - 2))
            surface.blit(dim_s, (r.centerx - dim_s.get_width() // 2, r.centery + 4))

        # Compute synchronized column sweep state while animating
        sweep = ease_in_out_quad(prog)
        active_w_cols: tuple[int, ...] = ()
        active_out_cols: tuple[int, ...] = ()
        active_in_rows: tuple[int, ...] = ()
        vis_out_cols: int | None = None

        if prog < 0.999 and self._out_card is not None:
            out_cols = self._out_card.grid.disp_cols
            vis_out_cols = max(1, math.ceil(out_cols * sweep))
            out_c = min(out_cols - 1, vis_out_cols - 1)
            active_out_cols = (out_c,)
            if self._w_card is not None:
                w_cols = self._w_card.grid.disp_cols
                w_c = round(out_c * (w_cols - 1) / max(1, out_cols - 1)) if w_cols > 1 else 0
                active_w_cols = (max(0, min(w_cols - 1, w_c)),)
            if self._in_card is not None and self._in_card.grid.disp_rows > 0:
                active_in_rows = (self._in_card.grid.disp_rows - 1,)

        if self._in_card is not None:
            draw_grid_card(
                surface,
                self._in_card,
                fonts,
                theme,
                scale,
                highlight_rows=active_in_rows,
            )

        if self._w_card is not None:
            draw_grid_card(
                surface,
                self._w_card,
                fonts,
                theme,
                scale,
                highlight_cols=active_w_cols,
            )

        if self._bias_card is not None:
            same_cols = (
                self._out_card is not None
                and self._bias_card.grid.disp_cols == self._out_card.grid.disp_cols
            )
            b_cols = active_out_cols if (active_out_cols and same_cols) else ()
            draw_grid_card(
                surface,
                self._bias_card,
                fonts,
                theme,
                scale,
                highlight_cols=b_cols,
            )

        if self._out_card is not None:
            draw_grid_card(
                surface,
                self._out_card,
                fonts,
                theme,
                scale,
                reveal_cols=vis_out_cols,
                highlight_cols=active_out_cols,
            )

        # Synchronized connecting arc/beam from active weight column to filling output column
        if (
            prog < 0.999
            and active_w_cols
            and active_out_cols
            and self._w_card is not None
            and self._out_card is not None
        ):
            wc = active_w_cols[0]
            oc = active_out_cols[0]
            w_pt = self._w_card.grid.cell_center(0, wc)
            o_pt = self._out_card.grid.cell_center(max(0, self._out_card.grid.disp_rows - 1), oc)
            pygame.draw.line(surface, theme.highlight, w_pt, o_pt, 2)
