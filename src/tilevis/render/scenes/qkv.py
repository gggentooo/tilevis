"""QKV projection scene showing input branching into Question, Label, and Content strips.

Single responsibility: visualize normalized input multiplying W_Q, W_K, W_V into Q, K, and V.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    chip_accent,
    draw_arrow,
    draw_bezier_arc,
    draw_grid_card,
    draw_panel,
    find_snapshot,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES

_QKV_ROLES: tuple[tuple[str, str, TensorRole, str], ...] = (
    ("Question (Q)", "What I'm looking for", TensorRole.QUERIES, "query"),
    ("Label (K)", "What I offer", TensorRole.KEYS, "key"),
    ("Content (V)", "What I'll share", TensorRole.VALUES, "val"),
)


@dataclass(slots=True)
class _QkvBranch:
    role_title: str
    role_hint: str
    weight_card: GridCard | None
    out_card: GridCard | None
    color_idx: int


@SCENES.register(StepKind.QKV_PROJECTION.value)
class QkvProjectionScene(BaseScene):
    """Branches normalized word vectors through 3 projections into Q, K, and V."""

    DEFAULT_DURATION_SEC = 1.15

    def __init__(self) -> None:
        super().__init__()
        self._in_snap: TensorSnapshot | None = None
        self._weight_snaps: list[TensorSnapshot] = []
        self._qkv_snaps: dict[str, TensorSnapshot | None] = {}
        self._in_card: GridCard | None = None
        self._branches: list[_QkvBranch] = []
        self._placeholder_in: pygame.Rect | None = None

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = ctx, animate
        self._in_snap = find_snapshot(step, TensorRole.INPUT)
        self._weight_snaps = list(step.by_role(TensorRole.WEIGHT))
        self._qkv_snaps = {
            TensorRole.QUERIES.value: find_snapshot(step, TensorRole.QUERIES),
            TensorRole.KEYS.value: find_snapshot(step, TensorRole.KEYS),
            TensorRole.VALUES.value: find_snapshot(step, TensorRole.VALUES),
        }

    def _pick_weight(self, branch_idx: int, keyword: str) -> TensorSnapshot | None:
        for ws in self._weight_snaps:
            if keyword in ws.name.lower():
                return ws
        if branch_idx < len(self._weight_snaps):
            return self._weight_snaps[branch_idx]
        return None

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._in_card = None
        self._branches = []
        self._placeholder_in = None
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        avail_w = max(40, stage.width - 2 * outer_pad)
        avail_h = max(40, stage.height - 2 * outer_pad)

        has_weights = len(self._weight_snaps) > 0
        gap_x = max(16, min(scale.px(44, minimum=14), avail_w // 12))
        gap_y = max(6, scale.px(12, minimum=4))

        if has_weights:
            in_w = max(24, int((avail_w - 2 * gap_x) * 0.25))
            w_w = max(24, int((avail_w - 2 * gap_x) * 0.28))
            out_w = max(24, avail_w - 2 * gap_x - in_w - w_w)
            in_x = stage.x + outer_pad
            w_x = in_x + in_w + gap_x
            out_x = w_x + w_w + gap_x
        else:
            in_w = max(24, int((avail_w - gap_x) * 0.24))
            out_w = max(24, avail_w - gap_x - in_w)
            w_w = 0
            in_x = stage.x + outer_pad
            w_x = 0
            out_x = in_x + in_w + gap_x

        in_h = max(40, min(avail_h, int(avail_h * 0.68)))
        in_rect = pygame.Rect(in_x, stage.y + outer_pad + (avail_h - in_h) // 2, in_w, in_h)

        if self._in_snap is not None:
            header_h = min(max(22, scale.px(44, minimum=22)), max(22, in_rect.height // 4))
            pad = max(4, min(scale.px(10, minimum=4), in_rect.width // 14))
            gw = max(1, in_rect.width - 2 * pad)
            gh = max(1, in_rect.height - header_h - 2 * pad)
            grid = make_grid(
                self._in_snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(18, minimum=6),
                min_cell_px=max(1, min(6, gw, gh)),
            )
            place_grid_in_card(in_rect, grid, header_h, pad)
            if grid.downsample_note:
                self.downsample_notes.append(grid.downsample_note)
            self._in_card = GridCard(
                title="Normalized Input",
                subtitle="x".join(str(s) for s in self._in_snap.full_shape),
                rect=in_rect,
                grid=grid,
                tooltip_title="Normalized Input",
                role=self._in_snap.role,
            )
            self._cards.append(self._in_card)
        else:
            self._placeholder_in = in_rect
            self._hit_regions.append(
                HitRegion(
                    rect=in_rect,
                    tooltip_title="Normalized Word Input",
                    tooltip_lines=("Feeds all 3 attention projections (Q, K, V).",),
                )
            )

        row_h = max(28, (avail_h - 2 * gap_y) // 3)
        sub_hdr = min(max(20, scale.px(36, minimum=18)), max(18, row_h // 2))

        for i, (title, hint, role, kw) in enumerate(_QKV_ROLES):
            ry = stage.y + outer_pad + i * (row_h + gap_y)
            w_card: GridCard | None = None
            out_card: GridCard | None = None

            if has_weights and w_w > 0:
                w_snap = self._pick_weight(i, kw)
                if w_snap is not None:
                    w_rect = pygame.Rect(w_x, ry, w_w, row_h)
                    pad = max(3, min(scale.px(8, minimum=3), w_rect.width // 16))
                    gw = max(1, w_rect.width - 2 * pad)
                    gh = max(1, w_rect.height - sub_hdr - 2 * pad)
                    wg = make_grid(
                        w_snap,
                        step,
                        ctx,
                        (gw, gh),
                        cell_px=scale.px(14, minimum=5),
                        min_cell_px=max(1, min(4, gw, gh)),
                    )
                    place_grid_in_card(w_rect, wg, sub_hdr, pad)
                    if wg.downsample_note:
                        self.downsample_notes.append(wg.downsample_note)
                    w_card = GridCard(
                        title=f"W_{kw[0].upper()}",
                        subtitle="weights",
                        rect=w_rect,
                        grid=wg,
                        tooltip_title=f"{title} Weights",
                        role=w_snap.role,
                    )
                    self._cards.append(w_card)

            o_snap = self._qkv_snaps.get(role.value)
            if o_snap is not None:
                o_rect = pygame.Rect(out_x, ry, out_w, row_h)
                pad = max(3, min(scale.px(10, minimum=3), o_rect.width // 16))
                gw = max(1, o_rect.width - 2 * pad)
                gh = max(1, o_rect.height - sub_hdr - 2 * pad)
                og = make_grid(
                    o_snap,
                    step,
                    ctx,
                    (gw, gh),
                    cell_px=scale.px(16, minimum=6),
                    min_cell_px=max(1, min(5, gw, gh)),
                )
                place_grid_in_card(o_rect, og, sub_hdr, pad)
                if og.downsample_note:
                    self.downsample_notes.append(og.downsample_note)
                out_card = GridCard(
                    title=title,
                    subtitle=hint,
                    rect=o_rect,
                    grid=og,
                    tooltip_title=title,
                    role=o_snap.role,
                    highlighted=True,
                )
                self._cards.append(out_card)

            self._branches.append(
                _QkvBranch(
                    role_title=title,
                    role_hint=hint,
                    weight_card=w_card,
                    out_card=out_card,
                    color_idx=i,
                )
            )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render 1-to-3 QKV branching projections with color-coded flow arrows."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        # Left input card or placeholder
        if self._in_card is not None:
            draw_grid_card(surface, self._in_card, fonts, theme, scale)
            src_rect = self._in_card.rect
        elif self._placeholder_in is not None:
            src_rect = self._placeholder_in
            draw_panel(surface, src_rect, theme, scale, alt=True)
            px = max(10, min(scale.font_px(20, minimum=10), max(10, src_rect.width // 8)))
            t_s = fonts.render_text(
                "Input Words",
                px,
                theme.text_primary,
                bold=True,
                family=theme.font_family,
                max_width=max(12, src_rect.width - 12),
            )
            surface.blit(
                t_s,
                (src_rect.centerx - t_s.get_width() // 2, src_rect.centery - t_s.get_height() // 2),
            )
        else:
            return

        branch_t = ease_out_cubic(min(1.0, max(0.0, (prog - 0.1) / 0.55)))
        out_t = ease_out_cubic(min(1.0, max(0.0, (prog - 0.4) / 0.6)))

        for br in self._branches:
            col = chip_accent(br.color_idx, theme)
            target_rect = (
                br.weight_card.rect
                if br.weight_card is not None
                else (br.out_card.rect if br.out_card is not None else None)
            )
            if target_rect is not None and branch_t > 0.05:
                p0 = (float(src_rect.right + 2), float(src_rect.centery))
                p2 = (float(target_rect.x - 4), float(target_rect.centery))
                p1 = ((p0[0] + p2[0]) * 0.5, p2[1])
                draw_bezier_arc(surface, p0, p1, p2, col, thickness=2)

            if br.weight_card is not None:
                draw_grid_card(
                    surface,
                    br.weight_card,
                    fonts,
                    theme,
                    scale,
                    title_px=max(
                        10, min(scale.font_px(18, minimum=10), br.weight_card.rect.height // 4)
                    ),
                    sub_px=max(
                        9, min(scale.font_px(14, minimum=9), br.weight_card.rect.height // 5)
                    ),
                )
                if br.out_card is not None and branch_t > 0.2:
                    draw_arrow(
                        surface,
                        (br.weight_card.rect.right + 2, br.weight_card.rect.centery),
                        (br.out_card.rect.x - 2, br.out_card.rect.centery),
                        col,
                        thickness=2,
                        head_len=8,
                    )

            if br.out_card is not None:
                vis_cols = (
                    max(0, math.ceil(br.out_card.grid.disp_cols * out_t)) if prog < 0.999 else None
                )
                draw_grid_card(
                    surface,
                    br.out_card,
                    fonts,
                    theme,
                    scale,
                    reveal_cols=vis_cols,
                    title_px=max(
                        10, min(scale.font_px(19, minimum=10), br.out_card.rect.height // 4)
                    ),
                    sub_px=max(9, min(scale.font_px(14, minimum=9), br.out_card.rect.height // 5)),
                )
