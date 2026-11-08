"""LM head logits scene showing last token meeting candidate words to score them.

Single responsibility: visualize final token vector x candidate vocab weights -> logit scores.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    blend_rgb,
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
class _LogitRow:
    rect: pygame.Rect
    word: str
    score: float
    token_id: int | None
    rank: int


@SCENES.register(StepKind.LOGITS.value)
class LogitsScene(BaseScene):
    """Shows the last token vector matching candidate word weights to produce raw scores."""

    DEFAULT_DURATION_SEC = 1.1

    def __init__(self) -> None:
        super().__init__()
        self._in_snap: TensorSnapshot | None = None
        self._w_snap: TensorSnapshot | None = None
        self._logits_snap: TensorSnapshot | None = None
        self._scoreboard_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._logit_rows: list[_LogitRow] = []
        self._ops: list[tuple[tuple[int, int], str]] = []
        self._last_token: str = ""

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        self._in_snap = find_snapshot(step, TensorRole.INPUT)
        self._w_snap = find_snapshot(step, TensorRole.WEIGHT)
        self._logits_snap = find_snapshot(step, TensorRole.LOGITS)
        self._last_token = (
            format_token_chip(ctx.pass_record.input_tokens[-1])
            if ctx.pass_record.input_tokens
            else ""
        )

    def _extract_candidates(self) -> list[tuple[str, float, int | None]]:
        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return []

        scores: list[float] = []
        if self._logits_snap is not None and self._logits_snap.data.size > 0:
            scores = [float(v) for v in np.asarray(self._logits_snap.data).reshape(-1)]

        meta_tokens = step.meta.get("candidate_tokens")
        meta_ids = step.meta.get("candidate_ids")
        labels: list[str] = []
        if isinstance(meta_tokens, (list, tuple)) and meta_tokens:
            labels = [str(t) for t in meta_tokens]
        elif self._logits_snap is not None and self._logits_snap.axes:
            ax = self._logits_snap.axes[-1]
            if ax.labels:
                labels = [str(t) for t in ax.labels]
        elif ctx.pass_record.prediction is not None:
            labels = [str(t) for t in ctx.pass_record.prediction.candidate_tokens]

        ids: list[int | None] = []
        if isinstance(meta_ids, (list, tuple)):
            ids = [int(i) for i in meta_ids]
        elif ctx.pass_record.prediction is not None:
            ids = [int(i) for i in ctx.pass_record.prediction.candidate_ids]

        n = max(len(scores), len(labels))
        out: list[tuple[str, float, int | None]] = []
        for i in range(n):
            lbl = labels[i] if i < len(labels) else f"#{i + 1}"
            sc = scores[i] if i < len(scores) else 0.0
            tid = ids[i] if i < len(ids) else None
            out.append((lbl, sc, tid))
        return out

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self._logit_rows = []
        self._ops = []
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        avail_w = max(40, stage.width - 2 * outer_pad)
        avail_h = max(40, stage.height - 2 * outer_pad)
        gap = max(14, min(scale.px(32, minimum=10), avail_w // 18))

        has_in = self._in_snap is not None
        has_w = self._w_snap is not None
        in_rect: pygame.Rect | None = None
        w_rect: pygame.Rect | None = None

        if has_in and has_w:
            in_w = max(24, int((avail_w - 2 * gap) * 0.24))
            w_w = max(24, int((avail_w - 2 * gap) * 0.34))
            board_w = max(40, avail_w - 2 * gap - in_w - w_w)
            in_rect = pygame.Rect(stage.x + outer_pad, stage.y + outer_pad, in_w, avail_h)
            w_rect = pygame.Rect(in_rect.right + gap, stage.y + outer_pad, w_w, avail_h)
            self._scoreboard_rect = pygame.Rect(
                w_rect.right + gap, stage.y + outer_pad, board_w, avail_h
            )
            self._ops = [
                ((in_rect.right + gap // 2, stage.y + outer_pad + avail_h // 2), "x"),
                ((w_rect.right + gap // 2, stage.y + outer_pad + avail_h // 2), "="),
            ]
        elif has_in or has_w:
            left_w = max(24, int((avail_w - gap) * 0.44))
            board_w = max(40, avail_w - gap - left_w)
            left_rect = pygame.Rect(stage.x + outer_pad, stage.y + outer_pad, left_w, avail_h)
            in_rect = left_rect if has_in else None
            w_rect = left_rect if has_w else None
            self._scoreboard_rect = pygame.Rect(
                left_rect.right + gap, stage.y + outer_pad, board_w, avail_h
            )
            self._ops = [((left_rect.right + gap // 2, stage.y + outer_pad + avail_h // 2), "=")]
        else:
            in_rect = None
            w_rect = None
            self._scoreboard_rect = pygame.Rect(
                stage.x + outer_pad, stage.y + outer_pad, avail_w, avail_h
            )

        hdr_h = min(max(24, scale.px(48, minimum=24)), max(24, avail_h // 4))

        if self._in_snap is not None and in_rect is not None:
            pad = max(4, min(scale.px(12, minimum=4), in_rect.width // 14))
            gw = max(1, in_rect.width - 2 * pad)
            gh = max(1, in_rect.height - hdr_h - 2 * pad)
            g_in = make_grid(
                self._in_snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(20, minimum=6),
                min_cell_px=max(1, min(6, gw, gh)),
            )
            place_grid_in_card(in_rect, g_in, hdr_h, pad)
            if g_in.downsample_note:
                self.downsample_notes.append(g_in.downsample_note)
            tok_txt = f'Token "{self._last_token}"' if self._last_token else "Last Token"
            self._cards.append(
                GridCard(
                    title=tok_txt,
                    subtitle="Final vector to score",
                    rect=in_rect,
                    grid=g_in,
                    tooltip_title="Last Token Vector",
                    role=self._in_snap.role,
                )
            )

        if self._w_snap is not None and w_rect is not None:
            pad = max(4, min(scale.px(12, minimum=4), w_rect.width // 14))
            gw = max(1, w_rect.width - 2 * pad)
            gh = max(1, w_rect.height - hdr_h - 2 * pad)
            g_w = make_grid(
                self._w_snap,
                step,
                ctx,
                (gw, gh),
                cell_px=scale.px(18, minimum=6),
                min_cell_px=max(1, min(6, gw, gh)),
            )
            place_grid_in_card(w_rect, g_w, hdr_h, pad)
            if g_w.downsample_note:
                self.downsample_notes.append(g_w.downsample_note)
            self._cards.append(
                GridCard(
                    title="Word Weights",
                    subtitle="Top candidate columns",
                    rect=w_rect,
                    grid=g_w,
                    tooltip_title="LM Head Candidate Weights",
                    role=self._w_snap.role,
                )
            )

        # Build right scoreboard: optional 1-D/2-D logits TileGrid + sorted candidate score pills
        sb = self._scoreboard_rect
        pad = max(6, min(scale.px(14, minimum=4), sb.width // 16))
        sb_hdr_h = max(hdr_h, scale.px(64, minimum=50))
        strip_h = 0
        g_log_bottom = sb.y + sb_hdr_h
        if self._logits_snap is not None and sb.height >= 150:
            strip_h = max(26, min(scale.px(52, minimum=26), sb.height // 6))
            g_log = make_grid(
                self._logits_snap,
                step,
                ctx,
                (max(1, sb.width - 2 * pad), max(1, strip_h - 8)),
                cell_px=scale.px(22, minimum=8),
                min_cell_px=max(1, min(8, sb.width - 2 * pad, strip_h - 8)),
            )
            gx = sb.x + pad + max(0, (sb.width - 2 * pad - g_log.rect.width) // 2)
            gy = sb.y + sb_hdr_h
            g_log.set_position(gx, gy)
            g_log_bottom = g_log.rect.bottom
            if g_log.downsample_note:
                self.downsample_notes.append(g_log.downsample_note)
            self._cards.append(
                GridCard(
                    title="Raw Logit Scores",
                    subtitle="Tile scores per candidate",
                    rect=sb,
                    grid=g_log,
                    tooltip_title="Candidate Logits",
                    role=self._logits_snap.role,
                    highlighted=True,
                )
            )

        candidates = self._extract_candidates()
        if not candidates:
            return
        rows_top = max(sb.y + sb_hdr_h, g_log_bottom + (8 if strip_h > 0 else 0))
        rows_h = max(20, sb.bottom - pad - rows_top)
        max_rows = min(len(candidates), max(1, rows_h // max(20, scale.px(26, minimum=18))))
        slot_h = rows_h / max(1, max_rows)
        row_h = max(16, min(scale.px(34, minimum=16), int(slot_h * 0.84)))

        for rank, (word, sc, tid) in enumerate(candidates[:max_rows]):
            ry = round(rows_top + rank * slot_h)
            rr = pygame.Rect(sb.x + pad, ry, max(16, sb.width - 2 * pad), row_h)
            self._logit_rows.append(
                _LogitRow(rect=rr, word=word, score=sc, token_id=tid, rank=rank)
            )
            id_line = f"Vocab ID: {tid}" if tid is not None else f"Candidate #{rank + 1}"
            self._hit_regions.append(
                HitRegion(
                    rect=rr,
                    tooltip_title=f'Candidate "{word}"',
                    tooltip_lines=(f"raw logit score: {sc:+.3f}", id_line),
                )
            )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render last token x candidate weights -> raw word scoreboard."""
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
        for center, kind in self._ops:
            draw_operator_symbol(surface, kind, center, sym_r, theme.accent, thickness=3)

        # Draw scoreboard background first if no logits GridCard owns it
        has_logits_card = any(c.role == TensorRole.LOGITS.value for c in self._cards)
        if not has_logits_card:
            draw_panel(surface, self._scoreboard_rect, theme, scale, highlighted=True)
            pad = max(6, scale.px(14, minimum=4))
            t_px = max(
                11, min(scale.font_px(22, minimum=11), max(11, self._scoreboard_rect.height // 12))
            )
            ts = fonts.render_text(
                "Word Scoreboard (Logits)",
                t_px,
                theme.accent,
                bold=True,
                family=theme.font_family,
                max_width=max(20, self._scoreboard_rect.width - 2 * pad),
            )
            surface.blit(ts, (self._scoreboard_rect.x + pad, self._scoreboard_rect.y + pad // 2))

        sweep_t = ease_out_cubic(min(1.0, prog))
        for card in self._cards:
            vis_cols: int | None = None
            hi_cols: tuple[int, ...] = ()
            if card.role == TensorRole.LOGITS.value and prog < 0.999:
                vis_cols = max(1, math.ceil(card.grid.disp_cols * sweep_t))
                hi_cols = (min(card.grid.disp_cols - 1, vis_cols - 1),)
            elif card.role == TensorRole.WEIGHT.value and prog < 0.999 and card.grid.disp_cols > 0:
                hi_cols = (min(card.grid.disp_cols - 1, int(sweep_t * card.grid.disp_cols)),)
            draw_grid_card(
                surface,
                card,
                fonts,
                theme,
                scale,
                reveal_cols=vis_cols,
                highlight_cols=hi_cols,
            )

        # Candidate score bars inside scoreboard
        if not self._logit_rows:
            return
        max_abs = max(1e-4, max(abs(r.score) for r in self._logit_rows))
        row_px = max(
            10, min(scale.font_px(18, minimum=10), max(10, self._logit_rows[0].rect.height - 4))
        )
        n_rows = len(self._logit_rows)
        for idx, row in enumerate(self._logit_rows):
            appear = ease_out_cubic(
                max(0.0, min(1.0, (prog - 0.25 - 0.55 * (idx / max(1, n_rows))) / 0.3))
            )
            if appear <= 0.02:
                continue
            r = row.rect
            draw_panel(
                surface,
                r,
                theme,
                scale,
                alt=True,
                border_color=(theme.highlight if idx == 0 else None),
            )
            word_w = max(44, min(scale.px(128, minimum=44), r.width // 3))
            chip_r = pygame.Rect(r.x + 3, r.y + 2, word_w, max(12, r.height - 4))
            draw_pill(
                surface,
                chip_r,
                format_token_chip(row.word),
                fonts,
                row_px,
                theme.background if idx == 0 else theme.text_primary,
                chip_accent(idx, theme) if idx == 0 else theme.panel,
                bold=(idx == 0),
                family=theme.font_family,
            )

            score_s = fonts.render_text(
                f"{row.score:+.2f}",
                max(9, row_px - 1),
                theme.highlight if idx == 0 else theme.text_secondary,
                bold=(idx == 0),
                family=theme.font_family,
            )
            score_x = r.right - score_s.get_width() - 8
            surface.blit(score_s, (score_x, r.centery - score_s.get_height() // 2))

            bar_area_x = chip_r.right + 8
            bar_area_w = max(8, score_x - bar_area_x - 8)
            bar_ratio = max(
                0.06,
                min(
                    1.0,
                    (row.score - min(0.0, min(x.score for x in self._logit_rows)))
                    / max(1e-4, max_abs * 2.0),
                ),
            )
            bar_w = max(4, round(bar_area_w * bar_ratio * appear))
            bar_h = max(6, r.height // 2)
            bar_col = blend_rgb(theme.accent_muted, theme.accent, 1.0 - idx / max(1, n_rows))
            pygame.draw.rect(
                surface,
                bar_col,
                pygame.Rect(bar_area_x, r.centery - bar_h // 2, bar_w, bar_h),
                border_radius=max(2, bar_h // 2),
            )
