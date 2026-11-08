"""Command enumeration for high-level presenter actions.

Single responsibility: enumerate discrete presenter commands (NEXT, PREV, JUMP, etc.).
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

from enum import StrEnum


class Command(StrEnum):
    """High-level interactive commands emitted by input controls and widgets."""

    NEXT = "next"
    PREV = "prev"
    NEXT_STAGE = "next_stage"
    PREV_STAGE = "prev_stage"
    FIRST = "first"
    LAST = "last"
    NEXT_PASS = "next_pass"
    JUMP = "jump"
    CYCLE_DETAIL = "cycle_detail"
    TOGGLE_NERD = "toggle_nerd"
    CYCLE_HEAD = "cycle_head"
    CYCLE_HEAD_PREV = "cycle_head_prev"
    AVG_HEADS = "avg_heads"
    TOGGLE_HELP = "toggle_help"
    CYCLE_THEME = "cycle_theme"
    TOGGLE_LOCALE = "toggle_locale"
    SPEED_UP = "speed_up"
    SPEED_DOWN = "speed_down"
    TOGGLE_AUTOPLAY = "toggle_autoplay"
    TOGGLE_FULLSCREEN = "toggle_fullscreen"
    ESCAPE = "escape"
    QUIT = "quit"
