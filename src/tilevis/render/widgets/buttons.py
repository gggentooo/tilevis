"""On-screen clickable navigation buttons (Prev, Next, Detail, Nerd, Help).

Single responsibility: render mouse-clickable control buttons with hover states.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.player import StepPlayer
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme

_DETAIL_NAMES = {
    int(DetailLevel.STORY): "Story",
    int(DetailLevel.STANDARD): "Standard",
    int(DetailLevel.DEEP): "Deep",
}


@dataclass
class ButtonState:
    """State and hit-test rectangle for a single clickable button."""

    cmd: Command
    label: str
    rect: pygame.Rect
    is_enabled: bool = True
    is_active: bool = False
    is_hovered: bool = False


class ButtonsWidget:
    """On-screen interactive navigation and toggle controls."""

    def __init__(self) -> None:
        self.buttons: list[ButtonState] = []

    def layout(
        self,
        rect: pygame.Rect,
        player: StepPlayer,
        settings: PresenterSettings,
        *,
        scale: UiScale,
    ) -> list[ButtonState]:
        """Compute layout of button rects within rect."""
        if rect.width <= 0 or rect.height <= 0:
            self.buttons = []
            return []

        detail_name = _DETAIL_NAMES.get(settings.detail_level, "Detail")
        nerd_label = "Nerd ON" if settings.nerd_mode else "Nerd"

        items = [
            (Command.PREV, "< Prev", player.can_prev, False),
            (Command.NEXT, "Next >", player.can_next, False),
            (Command.CYCLE_DETAIL, f"Detail: {detail_name}", True, False),
            (Command.TOGGLE_NERD, nerd_label, True, settings.nerd_mode),
            (
                Command.TOGGLE_LOCALE,
                f"Lang: {settings.locale.upper()}",
                True,
                settings.locale == "ko",
            ),
            (Command.TOGGLE_HELP, "?", True, False),
        ]

        btn_h = max(18, min(scale.px(30, minimum=18), rect.height - 4))
        btn_y = rect.y + max(0, (rect.height - btn_h) // 2)
        gap = max(4, scale.px(8, minimum=4))

        # Measure button widths using approximate character widths
        # or fixed minimum sizes
        buttons: list[ButtonState] = []
        # Layout from right to left or left to right
        # Let's allocate widths proportionally
        total_w = 0
        btn_specs: list[tuple[Command, str, bool, bool, int]] = []
        for cmd, label, enabled, active in items:
            char_w = scale.px(8, minimum=6)
            pad_x = scale.px(16, minimum=8)
            w = max(
                scale.px(28, minimum=22) if label == "?" else scale.px(64, minimum=44),
                len(label) * char_w + pad_x,
            )
            btn_specs.append((cmd, label, enabled, active, w))
            total_w += w

        total_w += gap * (len(items) - 1)
        # Position at the right edge of rect, or centered if small
        x = rect.right - total_w if total_w <= rect.width else rect.x

        for cmd, label, enabled, active, w in btn_specs:
            btn_rect = pygame.Rect(x, btn_y, max(1, w), btn_h)
            buttons.append(
                ButtonState(
                    cmd=cmd,
                    label=label,
                    rect=btn_rect,
                    is_enabled=enabled,
                    is_active=active,
                    is_hovered=False,
                )
            )
            x += w + gap

        self.buttons = buttons
        return buttons

    def update_hover(self, pos: tuple[int, int]) -> bool:
        """Update hover state on all buttons; return True if pos hits any button."""
        any_hit = False
        for btn in self.buttons:
            if btn.is_enabled and btn.rect.collidepoint(pos):
                btn.is_hovered = True
                any_hit = True
            else:
                btn.is_hovered = False
        return any_hit

    def handle_click(self, pos: tuple[int, int]) -> Command | None:
        """Return the Command of the clicked button if pos hits an enabled button, else None."""
        for btn in self.buttons:
            if btn.is_enabled and btn.rect.collidepoint(pos):
                return btn.cmd
        return None

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        player: StepPlayer,
        settings: PresenterSettings,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
    ) -> None:
        """Render navigation buttons into rect."""
        if rect.width <= 0 or rect.height <= 0:
            return

        self.layout(rect, player, settings, scale=scale)
        font_px = max(9, min(scale.font_px(13, minimum=9), max(9, rect.height - 6)))
        radius = max(2, scale.px(4, minimum=2))

        for btn in self.buttons:
            if btn.rect.width <= 0 or btn.rect.height <= 0:
                continue

            if not btn.is_enabled:
                bg = theme.panel
                border = theme.border
                txt_color = theme.masked_cell
            elif btn.is_active:
                bg = theme.accent
                border = theme.highlight
                txt_color = theme.background if theme.name == "dark" else (255, 255, 255)
            elif btn.is_hovered:
                bg = theme.panel_alt
                border = theme.accent
                txt_color = theme.text_primary
            else:
                bg = theme.panel
                border = theme.border
                txt_color = theme.text_secondary

            pygame.draw.rect(surface, bg, btn.rect, border_radius=radius)
            pygame.draw.rect(surface, border, btn.rect, width=1, border_radius=radius)

            surf = fonts.render_text(
                btn.label,
                font_px,
                txt_color,
                bold=btn.is_active or btn.is_hovered,
                family=theme.font_family,
                max_width=max(4, btn.rect.width - 2),
            )
            tx = btn.rect.x + max(0, (btn.rect.width - surf.get_width()) // 2)
            ty = btn.rect.y + max(0, (btn.rect.height - surf.get_height()) // 2)
            surface.blit(surf, (tx, ty))
