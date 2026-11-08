"""Input mapping translating pygame events to high-level Commands.

Single responsibility: map keyboard keys and mouse clicks to Commands with configurable bindings.
Allowed dependencies: pygame, tilevis.render, stdlib.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pygame

from tilevis.render.commands import Command

logger = logging.getLogger("tilevis.render.input_map")

# Modifier tuple: (shift, ctrl, alt)
ModifierState = tuple[bool, bool, bool]
KeySpec = tuple[int, ModifierState]

DEFAULT_KEY_BINDINGS: dict[str, list[str]] = {
    "next": ["right", "space", "return", "page down"],
    "prev": ["left", "backspace", "page up"],
    "next_stage": ["shift+right"],
    "prev_stage": ["shift+left"],
    "first": ["home"],
    "last": ["end"],
    "next_pass": ["ctrl+right"],
    "cycle_detail": ["d"],
    "toggle_nerd": ["n"],
    "cycle_head": ["]"],
    "cycle_head_prev": ["["],
    "avg_heads": ["a"],
    "toggle_help": ["h", "f1"],
    "cycle_theme": ["t"],
    "speed_up": ["+", "=", "kp_plus"],
    "speed_down": ["-", "kp_minus"],
    "toggle_autoplay": ["p"],
    "toggle_fullscreen": ["f11"],
    "escape": ["escape"],
}

_STATIC_KEY_MAP: dict[str, int] = {
    "right": pygame.K_RIGHT,
    "left": pygame.K_LEFT,
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "space": pygame.K_SPACE,
    "return": pygame.K_RETURN,
    "enter": pygame.K_RETURN,
    "kp_enter": pygame.K_KP_ENTER,
    "page down": pygame.K_PAGEDOWN,
    "pagedown": pygame.K_PAGEDOWN,
    "page up": pygame.K_PAGEUP,
    "pageup": pygame.K_PAGEUP,
    "backspace": pygame.K_BACKSPACE,
    "home": pygame.K_HOME,
    "end": pygame.K_END,
    "escape": pygame.K_ESCAPE,
    "esc": pygame.K_ESCAPE,
    "tab": pygame.K_TAB,
    "f1": pygame.K_F1,
    "f2": pygame.K_F2,
    "f3": pygame.K_F3,
    "f4": pygame.K_F4,
    "f5": pygame.K_F5,
    "f6": pygame.K_F6,
    "f7": pygame.K_F7,
    "f8": pygame.K_F8,
    "f9": pygame.K_F9,
    "f10": pygame.K_F10,
    "f11": pygame.K_F11,
    "f12": pygame.K_F12,
    "]": pygame.K_RIGHTBRACKET,
    "rightbracket": pygame.K_RIGHTBRACKET,
    "[": pygame.K_LEFTBRACKET,
    "leftbracket": pygame.K_LEFTBRACKET,
    "+": pygame.K_PLUS,
    "plus": pygame.K_PLUS,
    "=": pygame.K_EQUALS,
    "equals": pygame.K_EQUALS,
    "-": pygame.K_MINUS,
    "minus": pygame.K_MINUS,
    "kp_plus": pygame.K_KP_PLUS,
    "kp plus": pygame.K_KP_PLUS,
    "kp_minus": pygame.K_KP_MINUS,
    "kp minus": pygame.K_KP_MINUS,
    ",": pygame.K_COMMA,
    "comma": pygame.K_COMMA,
    ".": pygame.K_PERIOD,
    "period": pygame.K_PERIOD,
    "/": pygame.K_SLASH,
    "slash": pygame.K_SLASH,
}
for _ch in "abcdefghijklmnopqrstuvwxyz":
    _STATIC_KEY_MAP[_ch] = getattr(pygame, f"K_{_ch}")
for _num in "0123456789":
    _STATIC_KEY_MAP[_num] = getattr(pygame, f"K_{_num}")


def _resolve_keycode(key_name: str) -> int | None:
    """Resolve a pygame key name string into a keycode without requiring display init."""
    norm = key_name.strip().lower()
    if norm in _STATIC_KEY_MAP:
        return _STATIC_KEY_MAP[norm]
    if pygame.get_init():
        try:
            return int(pygame.key.key_code(norm))
        except (ValueError, pygame.error):
            return None
    return None


def parse_binding(binding_str: str) -> list[KeySpec]:
    """Parse a key binding descriptor such as 'shift+right' or 'page down'."""
    raw = binding_str.strip().lower()
    if not raw:
        return []

    shift = False
    ctrl = False
    alt = False

    if raw == "+":
        # Accept both dedicated K_PLUS and US-keyboard Shift+'='
        return [(pygame.K_PLUS, (False, False, False)), (pygame.K_EQUALS, (True, False, False))]

    if raw.endswith("++"):
        parts = [*raw[:-2].split("+"), "+"]
    else:
        parts = [p.strip() for p in raw.split("+") if p.strip()]

    if not parts:
        return []

    *mods, key_token = parts
    for mod in mods:
        if mod in {"shift", "lshift", "rshift"}:
            shift = True
        elif mod in {"ctrl", "control", "lctrl", "rctrl"}:
            ctrl = True
        elif mod in {"alt", "lalt", "ralt"}:
            alt = True
        else:
            logger.warning(
                "Unknown modifier %r in key binding %r; ignoring binding", mod, binding_str
            )
            return []

    code = _resolve_keycode(key_token)
    if code is None:
        logger.warning("Unrecognized key name %r in binding %r; ignoring", key_token, binding_str)
        return []

    mod_tuple: ModifierState = (shift, ctrl, alt)
    specs: list[KeySpec] = [(code, mod_tuple)]
    if key_token == "+" and mod_tuple == (False, False, False):
        specs.append((pygame.K_EQUALS, (True, False, False)))
    return specs


def _normalize_mod(mod_bits: int) -> ModifierState:
    """Extract normalized (shift, ctrl, alt) tuple, ignoring Caps/Num lock bits."""
    return (
        bool(mod_bits & pygame.KMOD_SHIFT),
        bool(mod_bits & pygame.KMOD_CTRL),
        bool(mod_bits & pygame.KMOD_ALT),
    )


class InputMap:
    """Configurable mapping from pygame keyboard and mouse events to presenter Commands."""

    def __init__(self, overrides: Mapping[str, Sequence[str]] | None = None) -> None:
        self._bindings: dict[Command, list[str]] = {}
        self._lookup: dict[KeySpec, Command] = {}

        merged: dict[str, Sequence[str]] = dict(DEFAULT_KEY_BINDINGS)
        if overrides:
            for cmd_name, key_list in overrides.items():
                merged[cmd_name.strip().lower()] = list(key_list)

        for cmd_name, raw_keys in merged.items():
            try:
                cmd = Command(cmd_name.lower())
            except ValueError:
                logger.warning("Unknown command name %r in [keys] config; ignoring", cmd_name)
                continue

            valid_labels: list[str] = []
            for key_str in raw_keys:
                specs = parse_binding(key_str)
                if not specs:
                    continue
                valid_labels.append(key_str)
                for spec in specs:
                    self._lookup[spec] = cmd
            self._bindings[cmd] = valid_labels

    def bindings_for(self, cmd: Command) -> tuple[str, ...]:
        """Return configured key display strings for the given command."""
        return tuple(self._bindings.get(cmd, ()))

    def translate_keydown(self, key: int, mod: int = 0) -> Command | None:
        """Translate a key code and pygame modifier bitmask into a Command, or None."""
        norm_mod = _normalize_mod(mod)
        cmd = self._lookup.get((key, norm_mod))
        if cmd is not None:
            return cmd
        return None

    def translate_event(
        self,
        event: pygame.event.Event,
        *,
        widget_consumed: bool = False,
    ) -> Command | None:
        """Translate a single pygame Event into a Command or None."""
        if event.type == pygame.QUIT:
            return Command.QUIT

        if widget_consumed:
            return None

        if event.type == pygame.KEYDOWN:
            key = int(getattr(event, "key", 0))
            mod = int(getattr(event, "mod", 0))
            cmd = self.translate_keydown(key, mod)
            if cmd is not None:
                return cmd
            unicode_ch = getattr(event, "unicode", "")
            if unicode_ch == "+" and (pygame.K_PLUS, (False, False, False)) in self._lookup:
                return self._lookup[(pygame.K_PLUS, (False, False, False))]
            return None

        if event.type == pygame.MOUSEBUTTONDOWN:
            button = int(getattr(event, "button", 0))
            if button == 1:
                return Command.NEXT
            if button == 3:
                return Command.PREV
            return None

        return None
