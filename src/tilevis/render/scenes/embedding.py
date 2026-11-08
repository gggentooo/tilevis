"""Embedding lookup scene combining token vectors and position vectors.

Single responsibility: visualize token shelf lookup, positional encoding addition, and sum output.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
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
    draw_grid_card,
    draw_operator_symbol,
    draw_panel,
    draw_pill,
    find_snapshot,
    format_token_chip,
    make_grid,
    place_grid_in_card,
)
from tilevis.render.scenes.registry import SCENES


@dataclass(slots=True)
class _OpPos:
    center: tuple[int, int]
    kind: str


@dataclass(slots=True)
class _RowLabelPill:
    rect: pygame.Rect
    text: str
    color_idx: int


@SCENES.register(StepKind.EMBEDDING.value)
class EmbeddingScene(BaseScene):
    """Shows tokens pulling rows from word shelf, adding position tiles -> embedded vectors."""

    DEFAULT_DURATION_SEC = 1.15

    def __init__(self) -> None:
        super().__init__()
        self._ops: list[_OpPos] = []
        self._row_pills: list[_RowLabelPill] = []
        self._has_pos_emb: bool = True
        self._missing_pos_rect: pygame.Rect | None = None

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        pos_snap = find_snapshot(step, TensorRole.POSITION_VECTORS)
        has_pos_meta = step.meta.get("has_pos_emb")
        if has_pos_meta is False or pos_snap is None:
            self._has_pos_emb = False
        else:
            self._has_pos_emb = True

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._ops = []
        self._row_pills = []
        self._missing_pos_rect = None
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        tok_snap = find_snapshot(step, TensorRole.TOKEN_VECTORS)
        pos_snap = find_snapshot(step, TensorRole.POSITION_VECTORS)
        out_snap = find_snapshot(step, TensorRole.OUTPUT)

        panels_spec: list[tuple[str, str, TensorSnapshot | None]] = []
        if tok_snap is not None:
            panels_spec.append(("Word Meaning", "Token fingerprint rows", tok_snap))
        if pos_snap is not None:
            pos_sub = (
                "Where each word sits" if self._has_pos_emb else "No position embeddings (zeros)"
            )
            panels_spec.append(("Position Tiles", pos_sub, pos_snap))
        else:
            panels_spec.append(("Position Tiles", "Not used in this model", None))
        if out_snap is not None:
            panels_spec.append(("Input Ready", "Word + Position sum", out_snap))

        if not panels_spec:
            return

        outer_pad = max(6, min(scale.px(24, minimum=6), stage.width // 24, stage.height // 24))
        shelf_w = (
            max(64, min(scale.px(148, minimum=64), stage.width // 6)) if stage.width >= 520 else 0
        )
        content_x = stage.x + outer_pad + (shelf_w + outer_pad // 2 if shelf_w > 0 else 0)
        content_w = max(30, stage.right - outer_pad - content_x)

        n_cards = len(panels_spec)
        op_gap = max(14, min(scale.px(36, minimum=14), content_w // 18))
        card_w = max(20, (content_w - op_gap * (n_cards - 1)) // n_cards)
        card_h = max(40, stage.height - 2 * outer_pad)
        header_h = min(max(24, scale.px(52, minimum=24)), max(24, card_h // 4))
        inner_pad = max(4, min(scale.px(14, minimum=4), card_w // 12))

        for i, (title, sub, maybe_snap) in enumerate(panels_spec):
            cx = content_x + i * (card_w + op_gap)
            cy = stage.y + outer_pad
            c_rect = pygame.Rect(cx, cy, card_w, card_h)

            if i > 0:
                op_kind = "+" if i == 1 else "="
                self._ops.append(_OpPos(center=(cx - op_gap // 2, cy + card_h // 2), kind=op_kind))

            if maybe_snap is None:
                self._missing_pos_rect = c_rect
                self._hit_regions.append(
                    HitRegion(
                        rect=c_rect,
                        tooltip_title="Position Embeddings",
                        tooltip_lines=("This model has no positional embedding layer.",),
                    )
                )
                continue

            snap = maybe_snap
            grid_w = max(1, c_rect.width - 2 * inner_pad)
            grid_h = max(1, c_rect.height - header_h - 2 * inner_pad)
            grid = make_grid(
                snap,
                step,
                ctx,
                (grid_w, grid_h),
                cell_px=scale.px(20, minimum=8),
                min_cell_px=max(1, min(8, grid_w, grid_h)),
            )
            place_grid_in_card(c_rect, grid, header_h, inner_pad)
            if grid.downsample_note:
                self.downsample_notes.append(grid.downsample_note)

            shape_txt = "x".join(str(s) for s in snap.full_shape)
            full_sub = f"{sub} ({shape_txt})"
            if grid.downsample_note:
                full_sub += f" . {grid.downsample_note}"
            self._cards.append(
                GridCard(
                    title=title,
                    subtitle=full_sub,
                    rect=c_rect,
                    grid=grid,
                    tooltip_title=title,
                    role=snap.role,
                    highlighted=(i == len(panels_spec) - 1),
                )
            )

        # Build token-shelf pills aligned or fanned to row centers of first TileGrid
        if shelf_w > 0 and self._cards and ctx.pass_record.input_tokens:
            first_card = self._cards[0]
            first_grid = first_card.grid
            labels = first_grid.row_labels or ctx.pass_record.input_tokens
            n_show = min(first_grid.disp_rows, len(labels))
            pill_x = stage.x + outer_pad
            pitch = first_grid.cell_px + first_grid.effective_gap_px
            if pitch >= 18 and n_show <= first_grid.disp_rows:
                pill_h = max(16, min(scale.px(32, minimum=16), pitch - 2))
                ys = [first_grid.cell_center(r, 0)[1] for r in range(n_show)]
            elif n_show > 0 and n_show <= 12:
                pill_h = max(18, scale.px(30, minimum=16))
                gap = max(6, scale.px(10, minimum=4))
                band_h = n_show * pill_h + (n_show - 1) * gap
                avail_band = max(pill_h, first_card.rect.height - header_h - 16)
                if band_h <= avail_band:
                    top_y = first_grid.rect.centery - band_h // 2
                    ys = [top_y + r * (pill_h + gap) + pill_h // 2 for r in range(n_show)]
                else:
                    ys = []
            else:
                pill_h = 20
                ys = []

            for r, cy in enumerate(ys):
                p_rect = pygame.Rect(pill_x, cy - pill_h // 2, shelf_w, pill_h)
                tok = format_token_chip(labels[r])
                self._row_pills.append(_RowLabelPill(rect=p_rect, text=tok, color_idx=r))
                self._hit_regions.append(
                    HitRegion(
                        rect=p_rect,
                        tooltip_title=f'Token Row {r + 1}: "{labels[r]}"',
                        tooltip_lines=("Looks up its matching word & position tile rows.",),
                    )
                )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render embedding lookup shelf, word + position -> sum transition."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        # Left token lookup shelf
        shelf_px = max(10, min(scale.font_px(18, minimum=10), 22))
        for i, rp in enumerate(self._row_pills):
            local = ease_out_cubic(min(1.0, prog / 0.35))
            if local <= 0.02:
                continue
            draw_pill(
                surface,
                rp.rect,
                rp.text,
                fonts,
                shelf_px,
                theme.background,
                chip_accent(rp.color_idx, theme),
                family=theme.font_family,
            )
            if self._cards:
                target_y = self._cards[0].grid.cell_center(i, 0)[1]
                pygame.draw.line(
                    surface,
                    chip_accent(rp.color_idx, theme),
                    (rp.rect.right + 2, rp.rect.centery),
                    (self._cards[0].rect.x - 2, target_y),
                    2,
                )

        # Draw '+' and '=' geometric symbols
        sym_r = max(6, scale.px(12, minimum=6))
        for op in self._ops:
            draw_operator_symbol(surface, op.kind, op.center, sym_r, theme.accent, thickness=3)

        # Missing pos embedding panel if applicable
        if self._missing_pos_rect is not None:
            draw_panel(surface, self._missing_pos_rect, theme, scale, alt=True)
            m_px = max(10, scale.font_px(20, minimum=10))
            ms = fonts.render_text(
                "No Position Embeddings",
                m_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=max(20, self._missing_pos_rect.width - 16),
            )
            surface.blit(
                ms,
                (
                    self._missing_pos_rect.x
                    + max(4, (self._missing_pos_rect.width - ms.get_width()) // 2),
                    self._missing_pos_rect.centery - ms.get_height() // 2,
                ),
            )

        # Reveal 3 cards in staggered rhythm: Word (0..0.45), Position (0.2..0.7), Output (0.5..1.0)
        for idx, card in enumerate(self._cards):
            if idx == 0:
                stage_t = ease_out_cubic(min(1.0, prog / 0.45))
            elif idx == 1 and len(self._cards) >= 3:
                stage_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.18) / 0.50)))
            else:
                stage_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.45) / 0.55)))

            vis_rows: int | None = None
            vis_cols: int | None = None
            if prog < 0.999:
                if idx == 0:
                    vis_rows = (
                        max(0, math.ceil(card.grid.disp_rows * stage_t)) if stage_t > 0.02 else 0
                    )
                else:
                    vis_cols = (
                        max(0, math.ceil(card.grid.disp_cols * stage_t)) if stage_t > 0.02 else 0
                    )

            draw_grid_card(
                surface,
                card,
                fonts,
                theme,
                scale,
                reveal_rows=vis_rows,
                reveal_cols=vis_cols,
            )
