"""Heads-up display (HUD) coordinating header, minimap, footer controls, and legend.

Single responsibility: render status bar, pipeline minimap, footer legend, and buttons.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import pygame

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel
from tilevis.render.colormap import Colormap
from tilevis.render.layout import UiScale
from tilevis.render.player import StepPlayer
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme
from tilevis.render.widgets.buttons import ButtonsWidget
from tilevis.render.widgets.legend import LegendWidget
from tilevis.render.widgets.minimap import MinimapWidget

_DETAIL_LABELS: dict[int, str] = {
    int(DetailLevel.STORY): "Story",
    int(DetailLevel.STANDARD): "Standard",
    int(DetailLevel.DEEP): "Deep",
}


class HudWidget:
    """Renders top header bar (with minimap) and footer bar (with legend and buttons)."""

    def __init__(self) -> None:
        self.minimap: MinimapWidget = MinimapWidget()
        self.legend: LegendWidget = LegendWidget()
        self.buttons: ButtonsWidget = ButtonsWidget()

    def format_progress(self, player: StepPlayer) -> str:
        """Build human-readable pass & step counter string ('Word X of Y · Step A / B')."""
        total_steps = player.total_steps
        if total_steps <= 0 or player.current is None:
            return "Step 0 / 0"

        step_num = player.cursor + 1
        ref = player.current
        total_passes = max(1, len(player.trace.passes))
        if ref.pass_index is not None:
            return f"Word {ref.pass_index + 1} of {total_passes} · Step {step_num} / {total_steps}"
        if ref.kind == "TITLE":
            return f"Intro · Step {step_num} / {total_steps}"
        return f"Summary · Step {step_num} / {total_steps}"

    def draw_header(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        player: StepPlayer,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
        nerd_mode: bool = False,
        esc_armed: bool = False,
    ) -> None:
        """Render header HUD bar containing model badge, pipeline minimap, and progress counter."""
        if rect.width <= 0 or rect.height <= 0:
            return

        pad_x = max(4, min(scale.px(20, minimum=4), rect.width // 16))
        pad_y = max(2, min(scale.px(8, minimum=2), rect.height // 8))
        bar = pygame.Rect(
            rect.x + pad_x,
            rect.y + pad_y,
            max(1, rect.width - 2 * pad_x),
            max(1, rect.height - 2 * pad_y),
        )
        pygame.draw.rect(
            surface,
            theme.panel,
            bar,
            border_radius=max(2, scale.px(theme.corner_radius)),
        )
        pygame.draw.rect(
            surface,
            theme.border,
            bar,
            width=1,
            border_radius=max(2, scale.px(theme.corner_radius)),
        )

        font_px = max(
            10, min(scale.font_px(theme.body_font_px, minimum=10), max(10, bar.height // 2))
        )
        inner_pad = max(4, scale.px(16, minimum=4))

        detail_label = _DETAIL_LABELS.get(player.settings.detail_level, "Custom")
        left_txt = f"{player.trace.model.name} · {detail_label}"
        if nerd_mode:
            left_txt += " · Nerd ON"

        progress_txt = self.format_progress(player)
        if esc_armed:
            right_txt = f"{progress_txt}   |   Press Esc again to exit"
            right_color = theme.warning
        else:
            right_txt = progress_txt
            right_color = theme.text_primary

        right_surf = fonts.render_text(
            right_txt,
            font_px,
            right_color,
            bold=True,
            family=theme.font_family,
            max_width=max(20, bar.width // 3),
        )
        left_max_w = max(20, bar.width // 4)
        left_surf = fonts.render_text(
            left_txt,
            font_px,
            theme.text_secondary,
            family=theme.font_family,
            max_width=left_max_w,
        )

        cy_left = bar.y + max(0, (bar.height - left_surf.get_height()) // 2)
        cy_right = bar.y + max(0, (bar.height - right_surf.get_height()) // 2)
        surface.blit(left_surf, (bar.x + inner_pad, cy_left))
        surface.blit(right_surf, (bar.right - inner_pad - right_surf.get_width(), cy_right))

        # Render pipeline minimap in the center region between left and right text
        left_end = bar.x + inner_pad + left_surf.get_width()
        right_start = bar.right - inner_pad - right_surf.get_width()
        minimap_avail_w = right_start - left_end - 2 * inner_pad
        if minimap_avail_w >= 100:
            minimap_h = max(16, min(scale.px(26, minimum=16), bar.height - 10))
            minimap_rect = pygame.Rect(
                left_end + inner_pad,
                bar.y + max(0, (bar.height - minimap_h) // 2),
                minimap_avail_w,
                minimap_h,
            )
            self.minimap.draw(
                surface,
                minimap_rect,
                player,
                theme=theme,
                fonts=fonts,
                scale=scale,
            )

    def draw_footer(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        player: StepPlayer,
        settings: PresenterSettings,
        *,
        colormap: Colormap,
        scale_val: float,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
    ) -> None:
        """Render footer bar containing colormap legend, hint text, and interactive buttons."""
        if rect.width <= 0 or rect.height <= 0:
            return

        pad_x = max(4, scale.px(16, minimum=4))

        # 1. Legend on the left: width ~140-200px
        legend_w = max(80, min(scale.px(180, minimum=80), rect.width // 5))
        legend_rect = pygame.Rect(rect.x + pad_x, rect.y + 2, legend_w, max(4, rect.height - 4))
        self.legend.draw(
            surface,
            legend_rect,
            colormap,
            scale_val,
            theme=theme,
            fonts=fonts,
            scale=scale,
        )

        # 2. Interactive buttons on the right: width ~320-540px
        buttons_w = max(180, min(scale.px(540, minimum=180), rect.width * 3 // 5))
        buttons_rect = pygame.Rect(
            rect.right - pad_x - buttons_w,
            rect.y + 2,
            buttons_w,
            max(4, rect.height - 4),
        )
        self.buttons.draw(
            surface,
            buttons_rect,
            player,
            settings,
            theme=theme,
            fonts=fonts,
            scale=scale,
        )

        # 3. Navigation hint centered between legend and buttons if space permits
        hint_left = legend_rect.right + pad_x
        hint_right = buttons_rect.x - pad_x
        hint_avail_w = hint_right - hint_left
        if hint_avail_w >= 120:
            hint = "Right/Space: Next · Left: Back · D: Detail · N: Nerd · T: Theme · Esc(x2): Quit"
            font_px = max(9, min(scale.font_px(11, minimum=9), max(9, rect.height - 8)))
            surf = fonts.render_text(
                hint,
                font_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=hint_avail_w,
            )
            hx = hint_left + max(0, (hint_avail_w - surf.get_width()) // 2)
            hy = rect.y + max(0, (rect.height - surf.get_height()) // 2)
            surface.blit(surf, (hx, hy))
