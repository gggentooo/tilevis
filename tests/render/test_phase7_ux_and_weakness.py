"""Phase 7 UX, overlay precedence, mouse-only presentation, and weakness tests."""

from __future__ import annotations

from pathlib import Path

import pygame
import pytest
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.config import PresenterSettings
from tilevis.core.serialization import load_trace
from tilevis.render.app import App
from tilevis.render.commands import Command

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "traces" / "samples"


def _click(pos: tuple[int, int], button: int = 1) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def _move(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEMOTION, pos=pos)


def _key(key: int, mod: int = 0) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


def test_overlay_precedence_clicks_and_escape_close_overlay() -> None:
    """While help_overlay is open, clicks and Esc close the overlay (no advance or quit)."""
    trace = load_trace(SAMPLES_DIR / "tiny_demo.trace.npz")
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1920, 1080))
    initial_cursor = app.player.cursor

    # Open help overlay via Command.TOGGLE_HELP
    app.dispatch_command(Command.TOGGLE_HELP)
    assert app._help_open

    # 1. Click anywhere on the backdrop while overlay is open
    app.tick(0.016, [_click((500, 500))])
    assert not app._help_open
    # Cursor did NOT move (stage click was consumed!)
    assert app.player.cursor == initial_cursor
    assert not app.should_quit

    # Re-open help overlay
    app.dispatch_command(Command.TOGGLE_HELP)
    assert app._help_open

    # 2. Escape closes overlay rather than arming quit
    app.tick(0.016, [_key(pygame.K_ESCAPE)])
    assert not app._help_open
    assert app._esc_timer is None
    assert not app.should_quit
    assert app.player.cursor == initial_cursor


def test_buttons_click_event_routing_does_not_advance_stage() -> None:
    """Clicking ?, Nerd, or Detail consumes the event and does NOT advance steps."""
    trace = load_trace(SAMPLES_DIR / "tiny_demo.trace.npz")
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1920, 1080))
    app.draw()
    initial_cursor = app.player.cursor

    # Locate the '?' Help button
    help_btn = next(b for b in app.hud.buttons.buttons if b.cmd == Command.TOGGLE_HELP)
    app.tick(0.016, [_click(help_btn.rect.center)])

    # Help overlay opened, cursor did NOT advance
    assert app._help_open
    assert app.player.cursor == initial_cursor

    # Close help overlay
    app.dispatch_command(Command.TOGGLE_HELP)
    assert not app._help_open

    # Locate the Nerd Mode button
    nerd_btn = next(b for b in app.hud.buttons.buttons if b.cmd == Command.TOGGLE_NERD)
    initial_nerd = app.settings.nerd_mode
    app.tick(0.016, [_click(nerd_btn.rect.center)])

    # Nerd mode toggled, cursor did NOT advance
    assert app.settings.nerd_mode != initial_nerd
    assert app.player.cursor == initial_cursor


def test_mouse_only_full_presentation_walkthrough() -> None:
    """Drive entire presentation from start to end and back using only mouse events."""
    trace = load_trace(SAMPLES_DIR / "tiny_demo.trace.npz")
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1920, 1080))
    app.draw()

    total = app.player.total_steps
    assert app.player.cursor == 0

    # Step all the way to the end using the on-screen Next > button
    for expected_cursor in range(total - 1):
        app.draw()
        next_btn = next(b for b in app.hud.buttons.buttons if b.cmd == Command.NEXT)
        assert next_btn.is_enabled

        # Move mouse over next button
        app.tick(0.016, [_move(next_btn.rect.center)])

        if app.scene.is_animating and app.settings.advance_skips_animation:
            # First click finishes current animation
            app.tick(0.016, [_click(next_btn.rect.center)])
            assert not app.scene.is_animating
            assert app.player.cursor == expected_cursor

        # Click next button advances to next step
        app.draw()
        app.tick(0.016, [_click(next_btn.rect.center)])
        assert app.player.cursor == expected_cursor + 1

    assert app.player.cursor == total - 1

    # Step all the way back using on-screen < Prev button
    for expected_cursor in range(total - 1, 0, -1):
        app.draw()
        prev_btn = next(b for b in app.hud.buttons.buttons if b.cmd == Command.PREV)
        assert prev_btn.is_enabled

        app.tick(0.016, [_move(prev_btn.rect.center)])
        app.tick(0.016, [_click(prev_btn.rect.center)])
        assert app.player.cursor == expected_cursor - 1

    assert app.player.cursor == 0


def test_theme_cache_invalidation_50_toggles() -> None:
    """Toggling theme (T) 50 times mid-step updates surfaces cleanly without errors or crashes."""
    trace = load_trace(SAMPLES_DIR / "tiny_demo.trace.npz")
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1920, 1080))
    app.player.cursor = 4  # mid-presentation layer
    app._enter_current_step(animate=True)

    for _i in range(50):
        prev_theme = app.theme.name
        app.dispatch_command(Command.CYCLE_THEME)
        assert app.theme.name != prev_theme
        assert app.fonts is not None
        # Verify tick and draw succeed without stale surfaces
        app.tick(0.016, [])
        app.draw()


@pytest.mark.parametrize("matrix_name", list(COMPAT_MATRIX.keys()))
def test_compat_matrix_walkthrough_all_buttons_and_toggles(matrix_name: str) -> None:
    """Full walkthrough clicking buttons and toggling N, T, D, H, P across compat_matrix."""
    entry = COMPAT_MATRIX[matrix_name]
    n_layers = 2 if matrix_name == "gpt2_small_shape" else entry.n_layers
    trace = make_fake_trace(
        n_layers=n_layers,
        emb_dim=min(64, entry.emb_dim),
        n_tokens=4,
        n_passes=1,
        n_heads=entry.n_heads,
        seed=42,
    )
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1280, 720))

    # Toggle Help on and off
    app.dispatch_command(Command.TOGGLE_HELP)
    assert app._help_open
    app.draw()
    app.dispatch_command(Command.TOGGLE_HELP)
    assert not app._help_open

    # Toggle Nerd mode on and off
    app.dispatch_command(Command.TOGGLE_NERD)
    assert app.settings.nerd_mode
    app.draw()
    app.dispatch_command(Command.TOGGLE_NERD)
    assert not app.settings.nerd_mode

    # Cycle theme twice
    app.dispatch_command(Command.CYCLE_THEME)
    app.draw()
    app.dispatch_command(Command.CYCLE_THEME)
    app.draw()

    # Cycle detail level
    app.dispatch_command(Command.CYCLE_DETAIL)
    app.draw()

    # Toggle autoplay
    app.dispatch_command(Command.TOGGLE_AUTOPLAY)
    assert app._autoplay
    app.tick(0.1, [])
    app.dispatch_command(Command.TOGGLE_AUTOPLAY)
    assert not app._autoplay

    # Click Minimap segment if available
    app.draw()
    if app.hud.minimap.segments:
        # Click first clickable segment
        for seg in app.hud.minimap.segments:
            if seg.is_clickable:
                app.hud.minimap.handle_click(seg.rect.center, app.player)
                break
        app.draw()
