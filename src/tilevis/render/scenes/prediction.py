"""Next-token prediction scene with monotonic probability bars, word landing, and confetti.

Single responsibility: animate candidate probabilities, highlight chosen word, append to text.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from tilevis.core.kinds import StepKind, TensorRole
from tilevis.core.trace import Step, TensorSnapshot
from tilevis.render.animation import ease_out_cubic
from tilevis.render.colormap import ColorRGB
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    HitRegion,
    blend_rgb,
    chip_accent,
    draw_panel,
    draw_pill,
    find_snapshot,
    format_token_chip,
    make_grid,
)
from tilevis.render.scenes.registry import SCENES


@dataclass(slots=True)
class PredictionBar:
    """Layout record of one candidate word probability bar."""

    candidate_id: int | None
    token: str
    prob: float
    row_rect: pygame.Rect
    bar_rect: pygame.Rect
    is_winner: bool


@dataclass(frozen=True, slots=True)
class _ConfettiParticle:
    angle: float
    speed: float
    size: int
    color_idx: int


@SCENES.register(StepKind.NEXT_TOKEN.value)
class PredictionScene(BaseScene):
    """Presents top-k word probabilities, chosen token landing on text, and optional confetti."""

    DEFAULT_DURATION_SEC = 1.35

    def __init__(self) -> None:
        super().__init__()
        self._cand_ids: list[int | None] = []
        self._cand_tokens: list[str] = []
        self._cand_probs: list[float] = []
        self._chosen_id: int | None = None
        self._chosen_token: str | None = None
        self._strategy: str = "greedy"
        self._winner_idx: int | None = None
        self._text_before: str = ""
        self._probs_snap: TensorSnapshot | None = None
        self._particles: list[_ConfettiParticle] = []

        self.bars: list[PredictionBar] = []
        self.confetti_drawn_last_frame: int = 0
        self._chart_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._text_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._landing_pos: tuple[int, int] = (0, 0)
        self._probs_card_rect: pygame.Rect | None = None

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        pred = ctx.pass_record.prediction
        self._text_before = ctx.pass_record.text_before or ctx.trace.prompt
        self._probs_snap = find_snapshot(step, TensorRole.PROBS)

        if pred is not None:
            self._cand_ids = [int(i) for i in pred.candidate_ids]
            self._cand_tokens = [str(t) for t in pred.candidate_tokens]
            self._cand_probs = [float(p) for p in pred.candidate_probs]
            self._chosen_id = int(pred.chosen_id)
            self._chosen_token = str(pred.chosen_token)
            self._strategy = str(pred.strategy)
        else:
            meta = step.meta
            raw_ids = meta.get("candidate_ids")
            raw_toks = meta.get("candidate_tokens")
            raw_probs = meta.get("candidate_probs")
            self._cand_ids = [int(i) for i in raw_ids] if isinstance(raw_ids, (list, tuple)) else []
            self._cand_tokens = (
                [str(t) for t in raw_toks] if isinstance(raw_toks, (list, tuple)) else []
            )
            self._cand_probs = (
                [float(p) for p in raw_probs] if isinstance(raw_probs, (list, tuple)) else []
            )
            cid = meta.get("chosen_id")
            ctok = meta.get("chosen_token")
            self._chosen_id = int(cid) if isinstance(cid, int) else None
            self._chosen_token = str(ctok) if isinstance(ctok, str) else None
            self._strategy = str(meta.get("strategy", "eos"))

        # Identify true chosen candidate index (supports non-argmax sampling!)
        self._winner_idx = None
        if self._chosen_id is not None and self._chosen_id in self._cand_ids:
            self._winner_idx = self._cand_ids.index(self._chosen_id)
        elif self._chosen_token is not None and self._chosen_token in self._cand_tokens:
            self._winner_idx = self._cand_tokens.index(self._chosen_token)

        # Deterministic particle seed per step
        self._particles = []
        if ctx.theme.confetti and self._chosen_token is not None:
            rng = random.Random(hash(step.id) & 0xFFFFFFFF)
            for _ in range(24):
                self._particles.append(
                    _ConfettiParticle(
                        angle=rng.uniform(-math.pi * 0.95, -math.pi * 0.05),
                        speed=rng.uniform(70.0, 210.0),
                        size=rng.randint(4, 8),
                        color_idx=rng.randint(0, 5),
                    )
                )

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._cards = []
        self._hit_regions = []
        self.bars = []
        self._probs_card_rect = None
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        avail_w = max(40, stage.width - 2 * outer_pad)
        avail_h = max(40, stage.height - 2 * outer_pad)
        gap = max(8, scale.px(14, minimum=6))

        text_h = max(56, min(scale.px(126, minimum=52), avail_h // 4))
        chart_h = max(48, avail_h - text_h - gap)
        self._chart_rect = pygame.Rect(stage.x + outer_pad, stage.y + outer_pad, avail_w, chart_h)
        self._text_rect = pygame.Rect(
            stage.x + outer_pad,
            self._chart_rect.bottom + gap,
            avail_w,
            text_h,
        )

        # Optional compact PROBS TileGrid strip at top-right of chart card
        pad = max(8, min(scale.px(16, minimum=6), avail_w // 20))
        hdr_h = max(32, min(scale.px(56, minimum=30), chart_h // 5))
        if self._probs_snap is not None and avail_w >= 460 and chart_h >= 160:
            strip_w = min(scale.px(300, minimum=120), avail_w // 3)
            g_probs = make_grid(
                self._probs_snap,
                step,
                ctx,
                (strip_w, max(12, hdr_h - 10)),
                cell_px=scale.px(22, minimum=8),
                min_cell_px=max(1, min(8, strip_w, hdr_h - 10)),
                signed=False,
            )
            gx = self._chart_rect.right - pad - g_probs.rect.width
            gy = self._chart_rect.y + max(6, (hdr_h - g_probs.rect.height) // 2)
            g_probs.set_position(gx, gy)
            self._probs_card_rect = self._chart_rect
            from tilevis.render.scenes.common import GridCard

            self._cards.append(
                GridCard(
                    title="Probability Tiles",
                    subtitle="",
                    rect=self._chart_rect,
                    grid=g_probs,
                    tooltip_title="Next-Word Probabilities",
                    role=self._probs_snap.role,
                )
            )
            if g_probs.downsample_note:
                self.downsample_notes.append(g_probs.downsample_note)

        # Horizontal probability bars inside chart_rect
        n = min(len(self._cand_tokens), len(self._cand_probs))
        if n > 0:
            bars_top = self._chart_rect.y + hdr_h + 4
            bars_h = max(24, self._chart_rect.bottom - pad - bars_top)
            max_show = min(n, max(1, bars_h // max(18, scale.px(24, minimum=16))))
            # Always keep chosen winner visible if ranked lower inside visible top-k
            indices = list(range(max_show))
            if (
                self._winner_idx is not None
                and self._winner_idx < n
                and self._winner_idx not in indices
            ):
                indices[-1] = self._winner_idx

            slot_h = bars_h / max(1, len(indices))
            row_h = max(14, min(scale.px(38, minimum=14), int(slot_h * 0.82)))
            label_w = max(56, min(scale.px(170, minimum=56), self._chart_rect.width // 4))
            pct_w = max(56, min(scale.px(165, minimum=56), self._chart_rect.width // 5))
            max_bar_w = max(16, self._chart_rect.width - 2 * pad - label_w - pct_w - 20)
            bar_x = self._chart_rect.x + pad + label_w + 10
            max_prob = max(1e-6, max(self._cand_probs[i] for i in indices))

            for slot, idx in enumerate(indices):
                ry = round(bars_top + slot * slot_h)
                row_rect = pygame.Rect(
                    self._chart_rect.x + pad, ry, self._chart_rect.width - 2 * pad, row_h
                )
                p = max(0.0, self._cand_probs[idx])
                full_bw = max(2, round(max_bar_w * (p / max_prob)))
                bar_rect = pygame.Rect(
                    bar_x, ry + max(2, row_h // 6), full_bw, max(6, int(row_h * 0.66))
                )
                is_win = self._winner_idx == idx
                tok = self._cand_tokens[idx]
                cid = self._cand_ids[idx] if idx < len(self._cand_ids) else None
                self.bars.append(
                    PredictionBar(
                        candidate_id=cid,
                        token=tok,
                        prob=p,
                        row_rect=row_rect,
                        bar_rect=bar_rect,
                        is_winner=is_win,
                    )
                )
                status = "CHOSEN WINNER" if is_win else f"Rank #{idx + 1}"
                self._hit_regions.append(
                    HitRegion(
                        rect=row_rect,
                        tooltip_title=f'Word "{tok}" ({status})',
                        tooltip_lines=(
                            f"probability: {p:.2%}",
                            f"strategy: {self._strategy}"
                            + (f" . ID {cid}" if cid is not None else ""),
                        ),
                    )
                )

        # Running text landing anchor
        self._landing_pos = (
            self._text_rect.right - max(90, scale.px(180, minimum=70)),
            self._text_rect.centery,
        )
        self._hit_regions.append(
            HitRegion(
                rect=self._text_rect,
                tooltip_title="Growing Sentence",
                tooltip_lines=(
                    f'Next word chosen: "{self._chosen_token or "<none (EOS)>"}"',
                    f"Selection mode: {self._strategy}",
                ),
            )
        )

    def _draw_confetti(
        self,
        surface: pygame.Surface,
        origin: tuple[int, int],
        prog: float,
        scale: UiScale,
    ) -> int:
        if self._ctx is None or not self._ctx.theme.confetti or not self._particles:
            return 0
        # Confetti sparkles cleanly from t=0.55..0.95 and finishes completely by t>=0.96
        if prog < 0.55 or prog >= 0.96:
            return 0
        local_t = (prog - 0.55) / (0.96 - 0.55)
        gravity = 260.0 * scale.s
        fade = 1.0 - max(0.0, (local_t - 0.55) / 0.45)
        theme = self._ctx.theme
        drawn = 0
        for p in self._particles:
            dx = math.cos(p.angle) * p.speed * scale.s * local_t
            dy = math.sin(p.angle) * p.speed * scale.s * local_t + 0.5 * gravity * (local_t**2)
            px = round(origin[0] + dx)
            py = round(origin[1] + dy)
            sz = max(2, round(p.size * scale.s * max(0.35, fade)))
            col = blend_rgb(theme.background, chip_accent(p.color_idx, theme), fade)
            if self._text_rect.inflate(80, 120).collidepoint(px, py):
                pygame.draw.rect(surface, col, pygame.Rect(px, py, sz, sz), border_radius=1)
                drawn += 1
        return drawn

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render top-k probability chart, chosen token flight, and growing text."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        draw_panel(surface, self._chart_rect, theme, scale)
        pad = max(8, min(scale.px(16, minimum=6), self._chart_rect.width // 20))

        # Title + strategy badge
        t_px = max(11, min(scale.font_px(24, minimum=11), max(11, self._chart_rect.height // 14)))
        s_px = max(9, min(scale.font_px(16, minimum=9), t_px - 2))
        if self._chosen_token is None and not self.bars:
            hdr_txt = "Generation Complete (End of Sequence)"
        elif self._strategy == "sample":
            hdr_txt = "Next-Word Odds (Sampled from Top Candidates)"
        else:
            hdr_txt = "Next-Word Odds (Highest-Probability Choice)"

        hdr_surf = fonts.render_text(
            hdr_txt,
            t_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=max(20, self._chart_rect.width // 2),
        )
        surface.blit(hdr_surf, (self._chart_rect.x + pad, self._chart_rect.y + pad // 2))

        if self._cards:
            self._cards[0].grid.draw(surface)

        # Animated monotonic probability bars
        bar_t = ease_out_cubic(min(1.0, prog / 0.55))
        winner_src_center: tuple[int, int] | None = None

        for idx, bar in enumerate(self.bars):
            rr = bar.row_rect
            is_win = bar.is_winner
            if is_win and prog >= 0.4:
                pygame.draw.rect(
                    surface,
                    theme.panel_alt,
                    rr,
                    border_radius=max(3, scale.px(6)),
                )
                pygame.draw.rect(
                    surface,
                    theme.success,
                    rr,
                    width=2,
                    border_radius=max(3, scale.px(6)),
                )

            lbl_w = max(48, min(scale.px(166, minimum=48), rr.width // 4))
            chip_r = pygame.Rect(rr.x + 2, rr.y + 2, lbl_w, max(12, rr.height - 4))
            draw_pill(
                surface,
                chip_r,
                format_token_chip(bar.token),
                fonts,
                max(10, min(t_px, rr.height - 4)),
                theme.background if is_win else theme.text_primary,
                theme.success if is_win else theme.panel_alt,
                theme.border if not is_win else None,
                bold=is_win,
                family=theme.font_family,
            )

            # Bar width strictly monotonic with prob
            cur_w = max(2, round(bar.bar_rect.width * bar_t))
            cur_bar = pygame.Rect(bar.bar_rect.x, bar.bar_rect.y, cur_w, bar.bar_rect.height)
            bar_col: ColorRGB = (
                theme.success
                if is_win
                else blend_rgb(theme.accent_muted, theme.accent, max(0.25, 1.0 - idx * 0.08))
            )
            pygame.draw.rect(surface, bar_col, cur_bar, border_radius=max(2, cur_bar.height // 3))

            prob_str = f"{bar.prob:.2%}" if 0.0 < bar.prob < 0.005 else f"{bar.prob:.1%}"
            pct_txt = prob_str + ("  [PICKED]" if is_win and prog >= 0.45 else "")
            pct_s = fonts.render_text(
                pct_txt,
                s_px,
                theme.success if is_win else theme.text_secondary,
                bold=is_win,
                family=theme.font_family,
                max_width=max(20, rr.right - cur_bar.right - 6),
            )
            px_x = min(rr.right - pct_s.get_width() - 4, cur_bar.right + 8)
            surface.blit(pct_s, (max(cur_bar.x + 4, px_x), rr.centery - pct_s.get_height() // 2))

            if is_win:
                winner_src_center = (chip_r.centerx, chip_r.centery)

        # Bottom Running Text Box + Flying Chosen Word
        draw_panel(surface, self._text_rect, theme, scale, alt=True)
        txt_px = max(12, min(scale.font_px(28, minimum=12), max(12, self._text_rect.height // 3)))
        inner_pad = max(10, scale.px(20, minimum=8))

        lbl_s = fonts.render_text(
            "Sentence So Far + Next Word:",
            max(10, s_px),
            theme.accent,
            bold=True,
            family=theme.font_family,
        )
        surface.blit(lbl_s, (self._text_rect.x + inner_pad, self._text_rect.y + 6))

        text_y = self._text_rect.y + 6 + lbl_s.get_height() + max(4, scale.px(8))
        if self._chosen_token is not None:
            new_word_disp = format_token_chip(self._chosen_token)
            word_s = fonts.render_text(
                new_word_disp, txt_px, theme.background, bold=True, family=theme.font_family
            )
            land_w = max(scale.px(64, minimum=44), word_s.get_width() + scale.px(24, minimum=12))
            land_h = max(24, txt_px + scale.px(12, minimum=6))

            # Trim/fit prompt tail smoothly before the appended token pill
            avail_before_w = max(20, self._text_rect.width - 2 * inner_pad - land_w - 16)
            before_disp = self._text_before.replace("·", " ")
            if fonts.get(txt_px, family=theme.font_family).size(before_disp)[0] > avail_before_w:
                before_disp = fonts.truncate(
                    before_disp, avail_before_w, txt_px, family=theme.font_family
                )
            before_s = fonts.render_text(
                before_disp, txt_px, theme.text_primary, family=theme.font_family
            )
            surface.blit(before_s, (self._text_rect.x + inner_pad, text_y))

            land_rect = pygame.Rect(
                self._text_rect.x + inner_pad + before_s.get_width() + 10,
                text_y + (before_s.get_height() - land_h) // 2,
                land_w,
                land_h,
            )
            self._landing_pos = land_rect.center

            # Animate pill flying from winning candidate row into land_rect
            fly_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.45) / 0.47)))
            if fly_t > 0.01:
                src_cx, src_cy = winner_src_center or (land_rect.centerx, self._chart_rect.centery)
                cur_cx = round(src_cx + (land_rect.centerx - src_cx) * fly_t)
                cur_cy = round(
                    src_cy
                    + (land_rect.centery - src_cy) * fly_t
                    - math.sin(fly_t * math.pi) * scale.px(24)
                )
                flying_rect = pygame.Rect(0, 0, land_w, land_h)
                flying_rect.center = (cur_cx, cur_cy)
                draw_pill(
                    surface,
                    flying_rect,
                    new_word_disp,
                    fonts,
                    txt_px,
                    theme.background,
                    theme.success,
                    theme.highlight if prog < 0.999 else None,
                    bold=True,
                    family=theme.font_family,
                )

            self.confetti_drawn_last_frame = self._draw_confetti(
                surface, self._landing_pos, prog, scale
            )
        else:
            eos_s = fonts.render_text(
                f'"{self._text_before}"  [End of Sequence]',
                txt_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=max(20, self._text_rect.width - 2 * inner_pad),
            )
            surface.blit(eos_s, (self._text_rect.x + inner_pad, text_y))
            self.confetti_drawn_last_frame = 0
