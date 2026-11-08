"""Title and End presentation cards.

Single responsibility: render opening model overview card and closing pass-highlighted summary.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from tilevis.core.trace import Step
from tilevis.render.animation import ease_out_cubic
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    HitRegion,
    chip_accent,
    draw_panel,
    draw_pill,
    format_token_chip,
)
from tilevis.render.scenes.registry import SCENES


@dataclass(slots=True)
class _StatBadge:
    rect: pygame.Rect
    label: str
    value: str
    hint: str


@dataclass(slots=True)
class _WordChip:
    rect: pygame.Rect
    text: str
    pass_label: str
    is_generated: bool
    accent_idx: int


@SCENES.register("TITLE")
class TitleScene(BaseScene):
    """Opening title card with model summary badges and animated prompt typing."""

    DEFAULT_DURATION_SEC = 1.05

    def __init__(self) -> None:
        super().__init__()
        self._badges: list[_StatBadge] = []
        self._box: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._prompt_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._model_name: str = ""
        self._prompt_text: str = ""

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        meta = step.meta
        m = ctx.trace.model
        self._model_name = str(meta.get("model_name", m.name))
        self._prompt_text = str(meta.get("prompt", ctx.trace.prompt))

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._hit_regions = []
        self._badges = []
        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        meta = step.meta
        m = ctx.trace.model
        n_layers = meta.get("n_layers", m.n_layers)
        n_heads = meta.get("n_heads", m.n_heads)
        emb_dim = meta.get("emb_dim", m.emb_dim)
        n_passes = meta.get("n_passes", len(ctx.trace.passes))

        pad = max(10, min(scale.px(44, minimum=10), stage.width // 12, stage.height // 12))
        self._box = pygame.Rect(
            stage.x + pad,
            stage.y + pad // 2,
            max(1, stage.width - 2 * pad),
            max(1, stage.height - pad),
        )

        stat_defs = [
            ("Layers", str(n_layers), f"{n_layers} stacked transformer blocks"),
            (
                "Attention Heads",
                str(n_heads) if n_heads is not None else "Dynamic",
                f"{n_heads if n_heads is not None else 'auto-detected'} spotlight heads per layer",
            ),
            ("Tiles per Word", str(emb_dim), f"Each token is represented by {emb_dim} numbers"),
            ("Words to Predict", str(n_passes), f"{n_passes} forward pass(es) recorded"),
        ]

        inner_pad = max(10, scale.px(32, minimum=10))
        badge_gap = max(8, scale.px(16, minimum=6))
        row_w = max(40, self._box.width - 2 * inner_pad)
        n_cols = 4 if self._box.width >= 640 else 2
        n_rows = math.ceil(len(stat_defs) / n_cols)
        badge_w = max(30, (row_w - badge_gap * (n_cols - 1)) // n_cols)
        badge_h = max(
            36, min(scale.px(92, minimum=36), self._box.height // (5 if n_rows == 1 else 7))
        )
        badges_top = self._box.y + max(60, int(self._box.height * 0.34))

        for i, (lbl, val, hint) in enumerate(stat_defs):
            r = i // n_cols
            c = i % n_cols
            rect = pygame.Rect(
                self._box.x + inner_pad + c * (badge_w + badge_gap),
                badges_top + r * (badge_h + badge_gap // 2),
                badge_w,
                badge_h,
            )
            self._badges.append(_StatBadge(rect=rect, label=lbl, value=val, hint=hint))
            self._hit_regions.append(
                HitRegion(rect=rect, tooltip_title=lbl, tooltip_lines=(hint, f"value: {val}"))
            )

        prompt_top = badges_top + n_rows * (badge_h + badge_gap // 2) + max(8, scale.px(18))
        prompt_h = max(36, self._box.bottom - inner_pad - prompt_top)
        self._prompt_rect = pygame.Rect(
            self._box.x + inner_pad,
            min(prompt_top, max(self._box.y + 40, self._box.bottom - prompt_h - 6)),
            row_w,
            prompt_h,
        )
        self._hit_regions.append(
            HitRegion(
                rect=self._prompt_rect,
                tooltip_title="Starting Prompt",
                tooltip_lines=(f'"{self._prompt_text}"', f"Model: {self._model_name}"),
            )
        )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render title card layout with typed prompt reveal."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress
        box = self._box
        draw_panel(surface, box, theme, scale)

        inner_pad = max(10, scale.px(32, minimum=10))
        content_w = max(20, box.width - 2 * inner_pad)

        # Progress bar strip across top
        bar_h = max(4, scale.px(6))
        bar_w = max(1, int(content_w * ease_out_cubic(prog)))
        pygame.draw.rect(
            surface,
            theme.accent,
            pygame.Rect(box.x + inner_pad, box.y + inner_pad // 2, bar_w, bar_h),
            border_radius=max(1, bar_h // 2),
        )

        tag_px = max(10, min(scale.font_px(20, minimum=10), box.height // 18))
        tag_rect = pygame.Rect(
            box.x + inner_pad, box.y + inner_pad, min(260, content_w), max(22, tag_px + 12)
        )
        draw_pill(
            surface,
            tag_rect,
            "LLM Forward Pass Tour",
            fonts,
            tag_px,
            theme.background,
            theme.accent,
            family=theme.font_family,
        )

        title_px = max(16, min(scale.font_px(46, minimum=16), box.height // 8))
        title_surf = fonts.render_text(
            self._model_name,
            title_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=content_w,
        )
        surface.blit(title_surf, (box.x + inner_pad, tag_rect.bottom + max(6, scale.px(12))))

        # Stat badges fade/reveal smoothly
        val_px = max(
            13,
            min(
                scale.font_px(30, minimum=13),
                max(13, self._badges[0].rect.height // 2 if self._badges else 24),
            ),
        )
        lbl_px = max(10, min(scale.font_px(18, minimum=10), val_px - 4))
        for idx, b in enumerate(self._badges):
            reveal_t = ease_out_cubic(max(0.0, min(1.0, (prog - idx * 0.08) / 0.35)))
            if reveal_t <= 0.02:
                continue
            draw_panel(
                surface, b.rect, theme, scale, alt=True, highlighted=(idx == 0 and prog < 1.0)
            )
            v_surf = fonts.render_text(
                b.value,
                val_px,
                chip_accent(idx, theme),
                bold=True,
                family=theme.font_family,
                max_width=max(10, b.rect.width - 12),
            )
            l_surf = fonts.render_text(
                b.label,
                lbl_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=max(10, b.rect.width - 12),
            )
            vx = b.rect.x + max(4, (b.rect.width - v_surf.get_width()) // 2)
            lx = b.rect.x + max(4, (b.rect.width - l_surf.get_width()) // 2)
            vy = b.rect.y + max(
                2, (b.rect.height - (v_surf.get_height() + l_surf.get_height() + 4)) // 2
            )
            surface.blit(v_surf, (vx, vy))
            surface.blit(l_surf, (lx, vy + v_surf.get_height() + 4))

        # Prompt typing showcase box
        p_rect = self._prompt_rect
        draw_panel(surface, p_rect, theme, scale, alt=True)
        body_px = max(
            11, min(scale.font_px(theme.heading_font_px, minimum=11), max(11, p_rect.height // 4))
        )
        small_px = max(10, min(scale.font_px(theme.small_font_px, minimum=10), body_px - 2))
        hdr = fonts.render_text(
            "Starting Prompt:",
            small_px,
            theme.accent,
            bold=True,
            family=theme.font_family,
            max_width=max(20, p_rect.width - 20),
        )
        p_pad = max(8, scale.px(18, minimum=8))
        surface.blit(hdr, (p_rect.x + p_pad, p_rect.y + p_pad // 2))

        type_prog = ease_out_cubic(min(1.0, max(0.0, prog / 0.85)))
        n_chars = (
            round(len(self._prompt_text) * type_prog) if prog < 1.0 else len(self._prompt_text)
        )
        shown = self._prompt_text[:n_chars]
        suffix = "|" if 0.0 < prog < 0.999 else ""
        display_prompt = f'"{shown}{suffix}"'

        cur_y = p_rect.y + p_pad // 2 + hdr.get_height() + max(4, scale.px(8))
        line_h = max(12, fonts.get(body_px, family=theme.font_family).get_linesize())
        avail_h = max(1, p_rect.bottom - p_pad // 2 - cur_y)
        lines = fonts.wrap_text(
            display_prompt,
            max(20, p_rect.width - 2 * p_pad),
            body_px,
            family=theme.font_family,
            max_lines=max(1, avail_h // line_h),
        )
        for line in lines:
            ls = fonts.render_text(line, body_px, theme.text_primary, family=theme.font_family)
            surface.blit(ls, (p_rect.x + p_pad, cur_y))
            cur_y += line_h


@SCENES.register("END")
class EndScene(BaseScene):
    """Closing card showcasing original prompt plus pass-highlighted generated words."""

    DEFAULT_DURATION_SEC = 1.15

    def __init__(self) -> None:
        super().__init__()
        self._box: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._chips: list[_WordChip] = []
        self._n_passes: int = 0
        self._model_name: str = ""

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        self._model_name = str(step.meta.get("model_name", ctx.trace.model.name))
        self._n_passes = int(step.meta.get("n_passes", len(ctx.trace.passes)))

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._hit_regions = []
        self._chips = []
        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        pad = max(10, min(scale.px(44, minimum=10), stage.width // 12, stage.height // 12))
        self._box = pygame.Rect(
            stage.x + pad,
            stage.y + pad // 2,
            max(1, stage.width - 2 * pad),
            max(1, stage.height - pad),
        )

        # Build sequence of prompt tokens from pass 0 + generated tokens per pass
        entries: list[tuple[str, str, bool, int, tuple[str, ...]]] = []
        if ctx.trace.passes:
            p0 = ctx.trace.passes[0]
            for t in p0.input_tokens:
                entries.append(
                    (
                        format_token_chip(t),
                        "Prompt",
                        False,
                        0,
                        (f'Prompt token: "{t}"', "Part of your original input text"),
                    )
                )
            for p in ctx.trace.passes:
                if p.prediction is None:
                    continue
                pred = p.prediction
                prob_str = ""
                if pred.chosen_id in pred.candidate_ids:
                    idx = pred.candidate_ids.index(pred.chosen_id)
                    if idx < len(pred.candidate_probs):
                        prob_str = f" ({pred.candidate_probs[idx]:.1%})"
                entries.append(
                    (
                        format_token_chip(pred.chosen_token),
                        f"Pass {p.index + 1}",
                        True,
                        p.index,
                        (
                            f'Predicted on pass {p.index + 1}: "{pred.chosen_token}"{prob_str}',
                            f"Strategy: {pred.strategy}",
                        ),
                    )
                )

        if not entries:
            return

        fonts = ctx.fonts
        theme = ctx.theme
        inner_pad = max(10, scale.px(32, minimum=10))
        chip_px = max(12, min(scale.font_px(28, minimum=12), max(12, self._box.height // 14)))
        lbl_px = max(9, min(scale.font_px(15, minimum=9), chip_px - 2))
        chip_h = chip_px + scale.px(18, minimum=8)
        row_h = chip_h + lbl_px + scale.px(18, minimum=8)
        gap_x = max(6, scale.px(12, minimum=4))
        gap_y = max(8, scale.px(14, minimum=6))

        start_x = self._box.x + inner_pad
        max_x = self._box.right - inner_pad
        cur_x = start_x
        cur_y = self._box.y + max(72, int(self._box.height * 0.28))

        for txt, pass_lbl, is_gen, accent_idx, tt_lines in entries:
            surf = fonts.render_text(txt or " ", chip_px, theme.text_primary, bold=is_gen)
            lbl_surf = fonts.render_text(pass_lbl, lbl_px, theme.text_secondary)
            w = max(surf.get_width(), lbl_surf.get_width()) + scale.px(24, minimum=12)
            w = min(w, max(30, max_x - start_x))

            if cur_x + w > max_x and cur_x > start_x:
                cur_x = start_x
                cur_y += row_h + gap_y

            if cur_y + row_h > self._box.bottom - inner_pad:
                break

            c_rect = pygame.Rect(cur_x, cur_y + lbl_px + 4, w, chip_h)
            self._chips.append(
                _WordChip(
                    rect=c_rect,
                    text=txt,
                    pass_label=pass_lbl,
                    is_generated=is_gen,
                    accent_idx=accent_idx,
                )
            )
            self._hit_regions.append(
                HitRegion(rect=c_rect, tooltip_title=pass_lbl, tooltip_lines=tt_lines)
            )
            cur_x += w + gap_x

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render final highlighted word sequence."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress
        box = self._box
        draw_panel(surface, box, theme, scale)

        inner_pad = max(10, scale.px(32, minimum=10))
        content_w = max(20, box.width - 2 * inner_pad)

        title_px = max(15, min(scale.font_px(40, minimum=15), box.height // 8))
        sub_px = max(11, min(scale.font_px(24, minimum=11), title_px - 4))
        t_surf = fonts.render_text(
            "Complete Generated Story",
            title_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=content_w,
        )
        surface.blit(t_surf, (box.x + inner_pad, box.y + inner_pad))

        n_gen = sum(1 for c in self._chips if c.is_generated)
        sub_txt = (
            f"{self._model_name} . Completed {self._n_passes} pass(es), added {n_gen} word(s)"
            if n_gen > 0
            else f"{self._model_name} . Stopped at End-of-Sequence ({self._n_passes} pass(es))"
        )
        s_surf = fonts.render_text(
            sub_txt,
            sub_px,
            theme.accent,
            family=theme.font_family,
            max_width=content_w,
        )
        surface.blit(s_surf, (box.x + inner_pad, box.y + inner_pad + t_surf.get_height() + 4))

        chip_px = max(12, min(scale.font_px(26, minimum=12), max(12, box.height // 14)))
        lbl_px = max(9, min(scale.font_px(15, minimum=9), chip_px - 2))

        gen_seen = 0
        total_gen = max(1, n_gen)
        for chip in self._chips:
            if not chip.is_generated:
                draw_pill(
                    surface,
                    chip.rect,
                    chip.text,
                    fonts,
                    chip_px,
                    theme.text_secondary,
                    theme.panel_alt,
                    theme.border,
                    bold=False,
                    family=theme.font_family,
                )
                continue

            # Reveal generated words sequentially by pass
            start_t = 0.15 + 0.70 * (gen_seen / total_gen)
            span_t = 0.35 / total_gen + 0.15
            local_t = ease_out_cubic(max(0.0, min(1.0, (prog - start_t) / span_t)))
            gen_seen += 1
            if local_t <= 0.02:
                continue

            accent = chip_accent(chip.accent_idx, theme)
            lbl_surf = fonts.render_text(
                chip.pass_label,
                lbl_px,
                accent,
                bold=True,
                family=theme.font_family,
                max_width=chip.rect.width,
            )
            lx = chip.rect.x + max(0, (chip.rect.width - lbl_surf.get_width()) // 2)
            surface.blit(lbl_surf, (lx, chip.rect.y - lbl_surf.get_height() - 3))

            draw_pill(
                surface,
                chip.rect,
                chip.text,
                fonts,
                chip_px,
                theme.background,
                accent,
                theme.highlight if local_t < 0.99 else None,
                bold=True,
                family=theme.font_family,
            )
