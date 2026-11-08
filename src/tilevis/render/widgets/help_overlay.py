"""Modal help overlay listing keyboard shortcuts and mouse actions.

Single responsibility: display searchable/scannable keybinding reference toggled with H or F1.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import pygame

from tilevis.render.commands import Command
from tilevis.render.input_map import InputMap
from tilevis.render.layout import UiScale
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


class HelpOverlayWidget:
    """Modal dialog displaying active keyboard and mouse shortcut cheat sheet."""

    def __init__(self) -> None:
        self.card_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.close_btn_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

    def draw(
        self,
        surface: pygame.Surface,
        viewport_w: int,
        viewport_h: int,
        input_map: InputMap,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
    ) -> None:
        """Render modal backdrop and centered shortcut reference dialog."""
        vw = max(1, int(viewport_w))
        vh = max(1, int(viewport_h))

        # 1. Dark semi-transparent backdrop
        dim_surf = pygame.Surface((vw, vh), pygame.SRCALPHA)
        dim_surf.fill((0, 0, 0, 180))
        surface.blit(dim_surf, (0, 0))

        # 2. Centered dialog card
        card_w = max(280, min(scale.px(680, minimum=280), vw - 20))
        card_h = max(240, min(scale.px(620, minimum=240), vh - 20))
        card_x = max(0, (vw - card_w) // 2)
        card_y = max(0, (vh - card_h) // 2)
        self.card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

        radius = max(4, scale.px(8, minimum=4))
        pygame.draw.rect(surface, theme.panel, self.card_rect, border_radius=radius)
        pygame.draw.rect(surface, theme.accent, self.card_rect, width=2, border_radius=radius)

        pad_x = max(10, scale.px(20, minimum=10))
        pad_y = max(10, scale.px(16, minimum=10))
        inner_w = card_w - 2 * pad_x
        cur_y = card_y + pad_y

        title_font_px = max(12, scale.font_px(18, minimum=12))
        sec_font_px = max(10, scale.font_px(13, minimum=10))
        text_font_px = max(9, scale.font_px(11, minimum=9))

        # Title and Close [X] button
        title_surf = fonts.render_text(
            "Keyboard & Mouse Shortcuts",
            title_font_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
        )
        surface.blit(title_surf, (card_x + pad_x, cur_y))

        close_surf = fonts.render_text(
            "[X]",
            sec_font_px,
            theme.accent,
            bold=True,
            family=theme.font_family,
        )
        self.close_btn_rect = pygame.Rect(
            self.card_rect.right - pad_x - close_surf.get_width() - 4,
            cur_y,
            close_surf.get_width() + 8,
            close_surf.get_height() + 4,
        )
        surface.blit(close_surf, (self.close_btn_rect.x + 2, self.close_btn_rect.y + 2))

        cur_y += title_surf.get_height() + max(4, scale.px(6, minimum=4))
        pygame.draw.line(
            surface,
            theme.border,
            (card_x + pad_x, cur_y),
            (card_x + card_w - pad_x, cur_y),
            width=1,
        )
        cur_y += max(6, scale.px(10, minimum=6))

        # Helper to format binding keys
        def _fmt(cmd: Command, default_hint: str) -> str:
            keys = input_map.bindings_for(cmd)
            if not keys:
                return default_hint
            return " / ".join(k.upper() for k in keys)

        # Shortcut sections
        sections: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Navigation",
                [
                    ("Next Step / Layer", _fmt(Command.NEXT, "RIGHT / SPACE")),
                    ("Previous Step", _fmt(Command.PREV, "LEFT")),
                    (
                        "Next / Prev Stage",
                        (
                            f"{_fmt(Command.NEXT_STAGE, 'SHIFT+RIGHT')} / "
                            f"{_fmt(Command.PREV_STAGE, 'SHIFT+LEFT')}"
                        ),
                    ),
                    (
                        "First / Last Step",
                        f"{_fmt(Command.FIRST, 'HOME')} / {_fmt(Command.LAST, 'END')}",
                    ),
                    ("Next Pass (Word)", _fmt(Command.NEXT_PASS, "CTRL+RIGHT")),
                ],
            ),
            (
                "Presentation & View Options",
                [
                    ("Detail Level (Story/Standard/Deep)", _fmt(Command.CYCLE_DETAIL, "D")),
                    ("Nerd Mode Overlay", _fmt(Command.TOGGLE_NERD, "N")),
                    ("Theme Toggle (Dark/Light)", _fmt(Command.CYCLE_THEME, "T")),
                    ("Autoplay On / Off", _fmt(Command.TOGGLE_AUTOPLAY, "P")),
                    (
                        "Speed Up / Slow Down",
                        f"{_fmt(Command.SPEED_UP, '+')} / {_fmt(Command.SPEED_DOWN, '-')}",
                    ),
                    ("Fullscreen Toggle", _fmt(Command.TOGGLE_FULLSCREEN, "F11")),
                    ("Help Screen Toggle", _fmt(Command.TOGGLE_HELP, "H / F1")),
                    ("Quit App", "ESC (press twice to confirm)"),
                ],
            ),
            (
                "Attention Scene Controls",
                [
                    (
                        "Cycle Attention Head",
                        f"{_fmt(Command.CYCLE_HEAD, ']')} / {_fmt(Command.CYCLE_HEAD_PREV, '[')}",
                    ),
                    ("Average All Heads", _fmt(Command.AVG_HEADS, "A")),
                ],
            ),
            (
                "Mouse Controls",
                [
                    ("Advance / Retreat", "Left-Click (advance) / Right-Click (retreat)"),
                    ("Interactive Elements", "Click Minimap segments or on-screen buttons"),
                    ("Inspect Values", "Hover over tiles to view exact unreduced numbers"),
                ],
            ),
        ]

        col_w = (inner_w - max(10, scale.px(16, minimum=10))) // 2
        col1_x = card_x + pad_x
        col2_x = col1_x + col_w + max(10, scale.px(16, minimum=10))

        # Render sections across 2 columns
        left_sections = sections[:2]
        right_sections = sections[2:]

        for col_x, sec_list in ((col1_x, left_sections), (col2_x, right_sections)):
            sy = cur_y
            for sec_title, rows in sec_list:
                sec_surf = fonts.render_text(
                    sec_title,
                    sec_font_px,
                    theme.highlight,
                    bold=True,
                    family=theme.font_family,
                    max_width=col_w,
                )
                surface.blit(sec_surf, (col_x, sy))
                sy += sec_surf.get_height() + 3

                for action, shortcut in rows:
                    act_surf = fonts.render_text(
                        action,
                        text_font_px,
                        theme.text_primary,
                        family=theme.font_family,
                        max_width=col_w,
                    )
                    short_surf = fonts.render_text(
                        shortcut,
                        text_font_px,
                        theme.text_secondary,
                        bold=True,
                        family=theme.font_family,
                        max_width=col_w,
                    )
                    surface.blit(act_surf, (col_x, sy))
                    sy += act_surf.get_height() + 1
                    surface.blit(short_surf, (col_x + max(6, scale.px(8, minimum=6)), sy))
                    sy += short_surf.get_height() + max(2, scale.px(3, minimum=2))

                sy += max(4, scale.px(8, minimum=4))

        # Footer hint
        hint_surf = fonts.render_text(
            "Press Esc, H, or click anywhere to close",
            text_font_px,
            theme.text_secondary,
            family=theme.font_family,
        )
        surface.blit(
            hint_surf,
            (
                card_x + max(0, (card_w - hint_surf.get_width()) // 2),
                self.card_rect.bottom - pad_y - hint_surf.get_height(),
            ),
        )
