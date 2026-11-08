"""Prompt and Tokenization scenes.

Single responsibility: animate raw text typing (PROMPT) and colored token-ID chips (TOKENIZE).
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from tilevis.core.kinds import StepKind
from tilevis.core.trace import Step
from tilevis.render.animation import ease_out_cubic
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.common import (
    BaseScene,
    HitRegion,
    chip_accent,
    draw_arrow,
    draw_panel,
    draw_pill,
    format_token_chip,
)
from tilevis.render.scenes.registry import SCENES


@dataclass(slots=True)
class _TokenChipLayout:
    chip_rect: pygame.Rect
    id_rect: pygame.Rect
    text: str
    raw_token: str
    token_id: int | None
    color_idx: int


@SCENES.register(StepKind.PROMPT.value)
class PromptScene(BaseScene):
    """Presents the input text so far typing out before entering the model."""

    DEFAULT_DURATION_SEC = 0.85

    def __init__(self) -> None:
        super().__init__()
        self._text: str = ""
        self._n_tokens: int = 0
        self._pass_index: int = 0
        self._card_rect: pygame.Rect = pygame.Rect(0, 0, 1, 1)

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        meta = step.meta
        raw = meta.get("text") or meta.get("prompt") or ctx.pass_record.text_before
        self._text = str(raw) if raw is not None else ""
        n_tok = meta.get("n_tokens")
        self._n_tokens = int(n_tok) if isinstance(n_tok, int) else len(ctx.pass_record.input_tokens)
        p_idx = meta.get("pass_index", meta.get("pass", ctx.pass_record.index))
        self._pass_index = int(p_idx) if isinstance(p_idx, int) else ctx.pass_record.index

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        pad = max(10, min(scale.px(48, minimum=10), stage.width // 10, stage.height // 10))
        self._card_rect = pygame.Rect(
            stage.x + pad,
            stage.y + pad // 2,
            max(1, stage.width - 2 * pad),
            max(1, stage.height - pad),
        )
        self._hit_regions = [
            HitRegion(
                rect=self._card_rect,
                tooltip_title=f"Pass {self._pass_index + 1} Input Text",
                tooltip_lines=(
                    f"{self._n_tokens} tokens, {len(self._text)} characters",
                    "Every pass reads the text so far and predicts 1 more token.",
                ),
            )
        ]

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render typing prompt reveal and pass indicator badge."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress
        box = self._card_rect
        draw_panel(surface, box, theme, scale)

        inner_pad = max(10, scale.px(32, minimum=10))
        content_w = max(20, box.width - 2 * inner_pad)

        pill_px = max(10, min(scale.font_px(20, minimum=10), box.height // 16))
        pill_rect = pygame.Rect(
            box.x + inner_pad,
            box.y + inner_pad,
            min(220, content_w),
            max(22, pill_px + 12),
        )
        draw_pill(
            surface,
            pill_rect,
            f"Pass {self._pass_index + 1} Input",
            fonts,
            pill_px,
            theme.background,
            theme.accent,
            family=theme.font_family,
        )

        # Prompt quote panel
        quote_top = pill_rect.bottom + max(10, scale.px(20))
        footer_h = max(28, scale.px(48, minimum=24))
        quote_h = max(36, box.bottom - inner_pad - footer_h - quote_top)
        q_rect = pygame.Rect(box.x + inner_pad, quote_top, content_w, quote_h)
        draw_panel(surface, q_rect, theme, scale, alt=True)

        body_px = max(13, min(scale.font_px(38, minimum=13), max(13, q_rect.height // 5)))
        q_pad = max(10, scale.px(24, minimum=8))
        type_prog = ease_out_cubic(min(1.0, prog / 0.85))
        n_chars = round(len(self._text) * type_prog) if prog < 1.0 else len(self._text)
        cursor = "|" if 0.0 < prog < 0.999 else ""
        shown = f'"{self._text[:n_chars]}{cursor}"'

        line_h = max(14, fonts.get(body_px, family=theme.font_family).get_linesize())
        max_lines = max(1, (q_rect.height - 2 * q_pad) // line_h)
        cur_y = q_rect.y + q_pad
        for line in fonts.wrap_text(
            shown,
            max(20, q_rect.width - 2 * q_pad),
            body_px,
            family=theme.font_family,
            max_lines=max_lines,
        ):
            surf = fonts.render_text(line, body_px, theme.text_primary, family=theme.font_family)
            surface.blit(surf, (q_rect.x + q_pad, cur_y))
            cur_y += line_h

        # Bottom summary flow indicator
        foot_y = box.bottom - inner_pad - footer_h // 2
        hint_px = max(10, min(scale.font_px(theme.body_font_px, minimum=10), box.height // 16))
        hint_surf = fonts.render_text(
            f"Ready: {self._n_tokens} token piece(s) to examine",
            hint_px,
            theme.text_secondary,
            family=theme.font_family,
            max_width=max(20, content_w - 50),
        )
        surface.blit(hint_surf, (box.x + inner_pad, foot_y - hint_surf.get_height() // 2))
        if prog > 0.3:
            ax0 = box.x + inner_pad + hint_surf.get_width() + 12
            ax1 = min(box.right - inner_pad, ax0 + max(24, scale.px(44)))
            if ax1 > ax0 + 8:
                draw_arrow(surface, (ax0, foot_y), (ax1, foot_y), theme.accent, thickness=3)


@SCENES.register(StepKind.TOKENIZE.value)
class TokenizeScene(BaseScene):
    """Splits input text into colored token chips with vocabulary ID badges underneath."""

    DEFAULT_DURATION_SEC = 1.05

    def __init__(self) -> None:
        super().__init__()
        self._tokens: list[str] = []
        self._token_ids: list[int | None] = []
        self._raw_text: str = ""
        self._chips: list[_TokenChipLayout] = []
        self._top_box: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._overflow_rect: pygame.Rect | None = None
        self._hidden_count: int = 0

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        meta = step.meta
        raw_tokens = meta.get("tokens")
        if isinstance(raw_tokens, (list, tuple)) and raw_tokens:
            self._tokens = [str(t) for t in raw_tokens]
        else:
            self._tokens = [str(t) for t in ctx.pass_record.input_tokens]

        raw_ids = meta.get("token_ids")
        if isinstance(raw_ids, (list, tuple)) and len(raw_ids) == len(self._tokens):
            self._token_ids = [int(i) for i in raw_ids]
        elif len(ctx.pass_record.input_ids) == len(self._tokens):
            self._token_ids = [int(i) for i in ctx.pass_record.input_ids]
        elif step.tensors and step.tensors[0].data.ndim == 1:
            arr = step.tensors[0].data
            self._token_ids = [int(v) for v in arr[: len(self._tokens)]]
        else:
            self._token_ids = [None] * len(self._tokens)

        self._raw_text = ctx.pass_record.text_before or "".join(self._tokens).replace("·", " ")

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._hit_regions = []
        self._chips = []
        self._overflow_rect = None
        self._hidden_count = 0

        if self._ctx is None:
            return

        fonts = self._ctx.fonts
        theme = self._ctx.theme

        outer_pad = max(8, min(scale.px(32, minimum=8), stage.width // 14, stage.height // 14))
        top_h = max(44, min(scale.px(110, minimum=44), stage.height // 4))
        self._top_box = pygame.Rect(
            stage.x + outer_pad,
            stage.y + outer_pad // 2,
            max(1, stage.width - 2 * outer_pad),
            top_h,
        )
        self._hit_regions.append(
            HitRegion(
                rect=self._top_box,
                tooltip_title="Original Text",
                tooltip_lines=(f'"{self._raw_text}"', f"Split into {len(self._tokens)} token(s)"),
            )
        )

        # Chip layout area underneath top text strip
        chips_top = self._top_box.bottom + max(24, scale.px(44, minimum=20))
        chips_area = pygame.Rect(
            stage.x + outer_pad,
            chips_top,
            max(1, stage.width - 2 * outer_pad),
            max(1, stage.bottom - outer_pad // 2 - chips_top),
        )

        n = len(self._tokens)
        if n == 0 or chips_area.height < 24:
            return

        chip_px = max(11, min(scale.font_px(26, minimum=11), max(11, chips_area.height // 10)))
        id_px = max(9, min(scale.font_px(18, minimum=9), chip_px - 2))
        chip_h = max(24, chip_px + scale.px(16, minimum=8))
        id_h = max(18, id_px + scale.px(8, minimum=4))
        pair_h = chip_h + 4 + id_h
        gap_x = max(6, scale.px(14, minimum=4))
        gap_y = max(8, scale.px(16, minimum=6))

        cur_x = chips_area.x
        cur_y = chips_area.y
        max_chip_w = max(40, chips_area.width // 2 if chips_area.width > 240 else chips_area.width)

        for i, tok in enumerate(self._tokens):
            tok_id = self._token_ids[i] if i < len(self._token_ids) else None
            chip_txt = format_token_chip(tok)
            t_surf = fonts.render_text(chip_txt or " ", chip_px, theme.text_primary, bold=True)
            id_txt = f"ID {tok_id}" if tok_id is not None else f"#{i + 1}"
            i_surf = fonts.render_text(id_txt, id_px, theme.text_secondary)

            w = max(t_surf.get_width(), i_surf.get_width()) + scale.px(22, minimum=10)
            w = min(max(scale.px(48, minimum=36), w), max_chip_w)

            if cur_x + w > chips_area.right and cur_x > chips_area.x:
                cur_x = chips_area.x
                cur_y += pair_h + gap_y

            if cur_y + pair_h > chips_area.bottom:
                self._hidden_count = n - i
                overflow_w = min(chips_area.width, scale.px(160, minimum=80))
                self._overflow_rect = pygame.Rect(
                    chips_area.right - overflow_w,
                    max(chips_area.y, chips_area.bottom - pair_h),
                    overflow_w,
                    chip_h,
                )
                break

            c_rect = pygame.Rect(cur_x, cur_y, w, chip_h)
            id_rect = pygame.Rect(cur_x, cur_y + chip_h + 4, w, id_h)
            self._chips.append(
                _TokenChipLayout(
                    chip_rect=c_rect,
                    id_rect=id_rect,
                    text=chip_txt,
                    raw_token=tok,
                    token_id=tok_id,
                    color_idx=i,
                )
            )
            tt_lines = (
                f'Token #{i + 1}: "{tok}"',
                f"Vocabulary ID: {tok_id}" if tok_id is not None else "No vocab ID",
            )
            self._hit_regions.append(
                HitRegion(
                    rect=c_rect.union(id_rect),
                    tooltip_title=f'Token "{tok}"',
                    tooltip_lines=tt_lines,
                )
            )
            cur_x += w + gap_x

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render raw text and popped-out colored token chips with vocab IDs."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        # Top sentence box
        draw_panel(surface, self._top_box, theme, scale)
        pad = max(8, scale.px(18, minimum=6))
        body_px = max(11, min(scale.font_px(26, minimum=11), max(11, self._top_box.height // 3)))
        lbl_px = max(9, min(scale.font_px(16, minimum=9), body_px - 2))
        hdr = fonts.render_text(
            "Input Text to Chop:",
            lbl_px,
            theme.accent,
            bold=True,
            family=theme.font_family,
        )
        surface.blit(hdr, (self._top_box.x + pad, self._top_box.y + pad // 2))
        txt_surf = fonts.render_text(
            f'"{self._raw_text}"',
            body_px,
            theme.text_primary,
            family=theme.font_family,
            max_width=max(20, self._top_box.width - 2 * pad),
        )
        surface.blit(
            txt_surf,
            (self._top_box.x + pad, self._top_box.y + pad // 2 + hdr.get_height() + 4),
        )

        # Connecting arrow from text box down toward chips
        if self._chips:
            mid_x = self._top_box.centerx
            ay0 = self._top_box.bottom + 4
            ay1 = self._chips[0].chip_rect.y - 6
            if ay1 > ay0 + 6:
                draw_arrow(
                    surface, (mid_x, ay0), (mid_x, ay1), theme.border, thickness=2, head_len=8
                )

        chip_px = max(11, min(scale.font_px(24, minimum=11), max(11, stage.height // 18)))
        id_px = max(9, min(scale.font_px(17, minimum=9), chip_px - 2))

        total = max(1, len(self._chips))
        for idx, ch in enumerate(self._chips):
            start_t = 0.08 + 0.72 * (idx / total)
            span_t = max(0.18, 0.30 / total + 0.14)
            local = ease_out_cubic(max(0.0, min(1.0, (prog - start_t) / span_t)))
            if local <= 0.02:
                continue

            # Subtle slide up into resting spot
            y_off = round((1.0 - local) * scale.px(12, minimum=4))
            c_rect = ch.chip_rect.move(0, y_off)
            id_rect = ch.id_rect.move(0, y_off)

            accent = chip_accent(ch.color_idx, theme)
            draw_pill(
                surface,
                c_rect,
                ch.text,
                fonts,
                chip_px,
                theme.background,
                accent,
                theme.highlight if (idx == len(self._chips) - 1 and prog >= 0.85) else None,
                bold=True,
                family=theme.font_family,
            )

            id_txt = f"ID {ch.token_id}" if ch.token_id is not None else f"#{idx + 1}"
            draw_pill(
                surface,
                id_rect,
                id_txt,
                fonts,
                id_px,
                theme.text_secondary,
                theme.panel_alt,
                theme.border,
                bold=False,
                family=theme.font_family,
            )

        if self._overflow_rect is not None and self._hidden_count > 0 and prog >= 0.8:
            draw_pill(
                surface,
                self._overflow_rect,
                f"+{self._hidden_count} more",
                fonts,
                id_px,
                theme.text_primary,
                theme.panel_alt,
                theme.border,
                family=theme.font_family,
            )
