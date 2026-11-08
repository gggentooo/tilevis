"""Phase 6 regression, compat matrix walkthrough, weakness mashing, and 1920x1080 perf tests."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pygame
import pytest
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel
from tilevis.core.serialization import load_trace
from tilevis.render.app import App
from tilevis.render.scenes.registry import SCENES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "traces" / "samples"


def _key(key: int, mod: int = 0) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


def test_step_player_animation_skip_and_prev_semantics_across_all_scenes() -> None:
    """NEXT during animation finishes first, second NEXT advances; PREV starts finished."""
    trace = load_trace(SAMPLES_DIR / "tiny_demo.trace.npz")
    settings = PresenterSettings(
        detail_level=int(DetailLevel.STANDARD),
        detail_blocks="first",
        later_pass_level=int(DetailLevel.STORY),
        advance_skips_animation=True,
    )
    app = App(trace, settings, headless=True, viewport_size=(1280, 720))

    total = app.player.total_steps
    seen_scene_types: set[str] = set()

    # Walk forward visiting every scene type (TITLE, PROMPT, ..., END)
    for idx in range(total - 1):
        assert app.player.cursor == idx
        seen_scene_types.add(type(app.scene).__name__)
        assert app.scene.is_animating

        # First NEXT finishes animation without moving cursor
        app.tick(0.016, [_key(pygame.K_RIGHT)])
        assert not app.scene.is_animating
        assert app.player.cursor == idx
        app.draw()

        # Second NEXT advances to idx + 1 with fresh animation started
        app.tick(0.016, [_key(pygame.K_RIGHT)])
        assert app.player.cursor == idx + 1
        app.draw()

    seen_scene_types.add(type(app.scene).__name__)
    assert len(seen_scene_types) >= 12  # covers all distinct kinds present in tiny_demo

    # Stepping backwards via PREV skips animation immediately on arrival
    for idx in range(total - 1, max(0, total - 15), -1):
        app.tick(0.016, [_key(pygame.K_LEFT)])
        assert app.player.cursor == idx - 1
        assert not app.scene.is_animating
        app.draw()


@pytest.mark.parametrize("matrix_name", list(COMPAT_MATRIX.keys()))
@pytest.mark.parametrize("detail", [DetailLevel.STORY, DetailLevel.STANDARD, DetailLevel.DEEP])
def test_compat_matrix_synthetic_three_frames_per_step(
    matrix_name: str, detail: DetailLevel
) -> None:
    """Render t=0, mid, and end frames for every step across every compat_matrix shape."""
    entry = COMPAT_MATRIX[matrix_name]
    n_layers = 2 if matrix_name == "gpt2_small_shape" else entry.n_layers
    trace = make_fake_trace(
        n_layers=n_layers,
        emb_dim=min(64, entry.emb_dim),
        n_tokens=4,
        n_passes=1,
        n_heads=entry.n_heads,
        seed=13,
    )
    settings = PresenterSettings(
        detail_level=int(detail),
        detail_blocks="first",
        advance_skips_animation=True,
    )
    app = App(trace, settings, headless=True, viewport_size=(1280, 720))

    total = app.player.total_steps
    max_visit = min(total, 28) if detail == DetailLevel.DEEP else total
    for step_i in range(max_visit):
        assert app.player.cursor == step_i
        # Frame 1: t=0
        app.tick(0.0, [])
        app.draw()
        # Frame 2: mid-animation (~0.5s)
        app.tick(0.5, [])
        app.draw()
        # Frame 3: completed end state
        app.scene.finish()
        app.draw()
        assert not app.scene.is_animating
        if app.player.can_next:
            app.tick(0.016, [_key(pygame.K_RIGHT)])


def test_rapid_input_mashing_200_commands_no_stale_state() -> None:
    """200 alternating navigation, head cycling, detail, and theme commands with random dt."""
    trace = make_fake_trace(n_layers=3, emb_dim=16, n_tokens=5, n_passes=2, n_heads=4, seed=99)
    rng = random.Random(2026)
    keys = [
        pygame.K_RIGHT,
        pygame.K_LEFT,
        pygame.K_RIGHTBRACKET,
        pygame.K_LEFTBRACKET,
        pygame.K_a,
        pygame.K_d,
        pygame.K_t,
        pygame.K_n,
    ]
    dts = [0.0, 0.001, 0.5]

    for vw, vh in ((800, 600), (1920, 1080), (3440, 1440)):
        app = App(trace, PresenterSettings(), headless=True, viewport_size=(vw, vh))
        prev_scene = app.scene
        prev_step_id = app.player.current.step_id if app.player.current else ""
        for _ in range(200):
            ev = _key(rng.choice(keys))
            dt = rng.choice(dts)
            app.tick(dt, [ev])
            app.draw()
            assert 0 <= app.player.cursor < app.player.total_steps
            cur_id = app.player.current.step_id if app.player.current else ""
            if cur_id != prev_step_id:
                assert app.scene is not prev_scene
                prev_scene = app.scene
                prev_step_id = cur_id


@pytest.mark.perf
def test_phase6_1920x1080_scene_enter_and_frame_time_budgets() -> None:
    """Every registered StepKind enters in < 100 ms and animates at < 16 ms/frame at 1920x1080."""
    # Large gpt2_small-like reduced trace (24x64 features, 64x64 weights, 4 heads)
    trace = make_fake_trace(
        n_layers=2,
        emb_dim=64,
        n_tokens=24,
        n_passes=1,
        n_heads=4,
        ff_mult=1,  # 64x64 weights + 24x64 activations (heavy display tile grid stress)
        seed=77,
    )
    settings = PresenterSettings(
        detail_level=int(DetailLevel.DEEP),
        detail_blocks="first",
        include_title=True,
        include_end=True,
    )
    app = App(trace, settings, headless=True, viewport_size=(1920, 1080))

    # Pick first occurrence of every registered StepKind (plus TITLE & END)
    kind_to_cursor: dict[str, int] = {}
    for idx, ref in enumerate(app.player.playlist):
        kind_to_cursor.setdefault(ref.kind, idx)

    assert len(kind_to_cursor) >= len(SCENES) - 1  # all except GENERIC covered directly

    worst_enter_ms = 0.0
    worst_avg_frame_ms = 0.0

    for kind, cursor_idx in sorted(kind_to_cursor.items()):
        app.player.cursor = cursor_idx

        t0 = time.perf_counter()
        app._enter_current_step(animate=True)
        # Include initial layout + TileGrid surface cache creation in enter budget check
        app.draw()
        enter_ms = (time.perf_counter() - t0) * 1000.0
        worst_enter_ms = max(worst_enter_ms, enter_ms)
        assert enter_ms < 100.0, f"{kind} initial enter+layout took {enter_ms:.1f} ms (>= 100 ms)"

        # Measure 20 animating frames across the step
        n_frames = 20
        t1 = time.perf_counter()
        for _ in range(n_frames):
            app.tick(0.016, [])
            app.draw()
        avg_frame_ms = ((time.perf_counter() - t1) * 1000.0) / n_frames
        worst_avg_frame_ms = max(worst_avg_frame_ms, avg_frame_ms)
        assert avg_frame_ms < 16.0, (
            f"{kind} animating avg frame took {avg_frame_ms:.2f} ms (>= 16.0 ms)"
        )

    assert worst_enter_ms < 100.0
    assert worst_avg_frame_ms < 16.0
