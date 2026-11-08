"""Unit tests for InputMap key/mouse translation and TOML [keys] overrides."""

from __future__ import annotations

import logging

import pygame
import pytest

from tilevis.render.commands import Command
from tilevis.render.input_map import InputMap


def test_default_key_bindings_and_lock_modifiers() -> None:
    """Default shortcuts map accurately and ignore unrelated CapsLock/NumLock modifier bits."""
    imap = InputMap()

    cases = [
        (pygame.K_RIGHT, 0, Command.NEXT),
        (pygame.K_SPACE, pygame.KMOD_CAPS, Command.NEXT),
        (pygame.K_RETURN, pygame.KMOD_NUM, Command.NEXT),
        (pygame.K_PAGEDOWN, 0, Command.NEXT),
        (pygame.K_LEFT, 0, Command.PREV),
        (pygame.K_BACKSPACE, 0, Command.PREV),
        (pygame.K_PAGEUP, 0, Command.PREV),
        (pygame.K_RIGHT, pygame.KMOD_SHIFT, Command.NEXT_STAGE),
        (pygame.K_LEFT, pygame.KMOD_SHIFT, Command.PREV_STAGE),
        (pygame.K_RIGHT, pygame.KMOD_CTRL, Command.NEXT_PASS),
        (pygame.K_HOME, 0, Command.FIRST),
        (pygame.K_END, 0, Command.LAST),
        (pygame.K_d, 0, Command.CYCLE_DETAIL),
        (pygame.K_n, 0, Command.TOGGLE_NERD),
        (pygame.K_RIGHTBRACKET, 0, Command.CYCLE_HEAD),
        (pygame.K_LEFTBRACKET, 0, Command.CYCLE_HEAD_PREV),
        (pygame.K_a, 0, Command.AVG_HEADS),
        (pygame.K_h, 0, Command.TOGGLE_HELP),
        (pygame.K_F1, 0, Command.TOGGLE_HELP),
        (pygame.K_t, 0, Command.CYCLE_THEME),
        (pygame.K_PLUS, 0, Command.SPEED_UP),
        (pygame.K_EQUALS, pygame.KMOD_SHIFT, Command.SPEED_UP),
        (pygame.K_MINUS, 0, Command.SPEED_DOWN),
        (pygame.K_p, 0, Command.TOGGLE_AUTOPLAY),
        (pygame.K_F11, 0, Command.TOGGLE_FULLSCREEN),
        (pygame.K_ESCAPE, 0, Command.ESCAPE),
    ]
    for key, mod, expected in cases:
        ev = pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)
        assert imap.translate_event(ev) == expected


def test_mouse_click_and_widget_consumption() -> None:
    """Unhandled left/right clicks translate to NEXT/PREV; consumed events yield None."""
    imap = InputMap()
    left = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(100, 100))
    right = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(100, 100))
    middle = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=2, pos=(100, 100))

    assert imap.translate_event(left) == Command.NEXT
    assert imap.translate_event(right) == Command.PREV
    assert imap.translate_event(middle) is None
    assert imap.translate_event(left, widget_consumed=True) is None


def test_custom_overrides_and_unrecognized_keys(caplog: pytest.LogCaptureFixture) -> None:
    """Custom [keys] replace defaults while unknown key/command names warn and never raise."""
    with caplog.at_level(logging.WARNING, logger="tilevis.render.input_map"):
        imap = InputMap(
            overrides={
                "next": ["j"],
                "prev": ["not_a_real_key_name", "k"],
                "nonexistent_command": ["x"],
            }
        )

    assert imap.translate_keydown(pygame.K_j) == Command.NEXT
    assert imap.translate_keydown(pygame.K_k) == Command.PREV
    assert imap.translate_keydown(pygame.K_RIGHT) is None  # replaced by override
    assert any("Unrecognized key name" in r.message for r in caplog.records)
    assert any("Unknown command name" in r.message for r in caplog.records)
