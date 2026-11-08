"""Headless App integration, compatibility matrix, weakness probes, and torch-free tests."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pytest
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.fake_traces import make_fake_trace, make_snapshot

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel, Stage, TensorRole
from tilevis.core.serialization import load_trace
from tilevis.core.trace import Fidelity, Step
from tilevis.render.app import DOUBLE_ESC_WINDOW_SEC, App
from tilevis.render.scenes.embedding import EmbeddingScene
from tilevis.render.scenes.generic import GenericScene

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "traces" / "samples"


def _key_event(key: int, mod: int = 0) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


@pytest.mark.parametrize("matrix_name", list(COMPAT_MATRIX.keys()))
def test_headless_app_walkthrough_compat_matrix(matrix_name: str) -> None:
    """App steps start -> end -> start cleanly on synthetic traces matching every compat entry."""
    entry = COMPAT_MATRIX[matrix_name]
    n_passes = 1 if matrix_name == "gpt2_small_shape" else 2
    trace = make_fake_trace(
        n_layers=entry.n_layers,
        emb_dim=entry.emb_dim,
        n_tokens=4,
        n_passes=n_passes,
        n_heads=entry.n_heads,
        seed=42,
    )
    settings = PresenterSettings(
        detail_level=int(DetailLevel.STANDARD),
        detail_blocks="first",
        advance_skips_animation=True,
    )
    app = App(trace, settings, headless=True, viewport_size=(1280, 720))

    total = app.player.total_steps
    assert total >= 3

    # Step start -> end using NEXT (skipping active animation then advancing)
    for idx in range(total - 1):
        assert app.player.cursor == idx
        if app.scene.is_animating:
            app.tick(0.016, [_key_event(pygame.K_RIGHT)])
            assert not app.scene.is_animating
            assert app.player.cursor == idx
        app.tick(0.016, [_key_event(pygame.K_RIGHT)])
        assert app.player.cursor == idx + 1
        app.draw()

    # Bounded at end
    app.scene.finish()
    app.tick(0.016, [_key_event(pygame.K_RIGHT)])
    assert app.player.cursor == total - 1

    # Step end -> back to start via PREV (immediate end state, no animation)
    for idx in range(total - 1, 0, -1):
        app.tick(0.016, [_key_event(pygame.K_LEFT)])
        assert app.player.cursor == idx - 1
        assert not app.scene.is_animating
        app.draw()

    assert app.player.cursor == 0


@pytest.mark.parametrize("sample_name", ["tiny_demo.trace.npz", "tiny_ch07.trace.npz"])
def test_headless_app_phase4_sample_traces(sample_name: str) -> None:
    """Committed Phase 4 real traces step cleanly end-to-end inside headless App."""
    trace_path = SAMPLES_DIR / sample_name
    assert trace_path.exists(), f"Missing committed sample trace: {trace_path}"
    trace = load_trace(trace_path)

    app = App(trace, PresenterSettings(detail_level=2), headless=True, viewport_size=(1920, 1080))
    total = app.player.total_steps
    assert total > 5

    # Walk full playlist cleanly and test runtime toggles along the way
    steps_visited = 1
    while app.player.cursor < total - 1:
        app.tick(2.0, [])  # complete 0.8-1.5s animation naturally
        assert not app.scene.is_animating
        app.draw()
        app.tick(0.016, [_key_event(pygame.K_SPACE)])
        steps_visited += 1

    assert steps_visited == total
    # Toggle detail, nerd, and theme cleanly
    app.tick(0.016, [_key_event(pygame.K_n)])
    assert app.settings.nerd_mode is True
    app.draw()
    app.tick(0.016, [_key_event(pygame.K_t)])
    assert app.theme.name == "light"
    app.draw()
    app.tick(0.016, [_key_event(pygame.K_d)])
    assert 0 <= app.player.cursor < app.player.total_steps
    app.draw()


@pytest.mark.parametrize(
    ("vw", "vh"),
    [
        (800, 600),
        (1280, 720),
        (1366, 768),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (3440, 1440),
    ],
)
def test_extreme_viewports_and_tight_downsampling(vw: int, vh: int) -> None:
    """All standard/extreme viewports render cleanly and trigger 'showing N of M' when tight."""
    # Wide 24x64 tensors will exceed 8px tile minimum at narrow viewport widths like 800x600
    trace = make_fake_trace(n_layers=1, emb_dim=64, n_tokens=24, n_passes=1, n_heads=4)
    app = App(trace, PresenterSettings(include_title=False), headless=True, viewport_size=(vw, vh))

    # Jump to embedding step (has 3 wide tensors side by side -> tight on 800x600)
    app.player.goto("p0.embed")
    app._enter_current_step(animate=False)
    app.tick(0.016, [])
    app.draw()

    assert isinstance(app.scene, EmbeddingScene)
    if vw == 800:
        notes = app.scene.downsample_notes
        assert notes, "Expected automatic display downsampling in 800x600 tight stage"
        assert any("showing" in n and " of " in n for n in notes)

        # Also verify GenericScene tight downsampling via an unknown 3-tensor step
        embed_step = trace.step_by_id("p0.embed")
        generic_wide = replace(embed_step, id="p0.future_wide", kind="future_wide")
        mod_pass = replace(trace.passes[0], steps=(generic_wide,))
        g_app = App(
            replace(trace, passes=(mod_pass,)),
            PresenterSettings(include_title=False, include_end=False),
            headless=True,
            viewport_size=(vw, vh),
        )
        g_app.draw()
        assert isinstance(g_app.scene, GenericScene)
        assert g_app.scene.downsample_notes
        assert any("showing" in n and " of " in n for n in g_app.scene.downsample_notes)


def test_rapid_resize_mid_playback_and_animation() -> None:
    """VIDEORESIZE and WINDOWSIZECHANGED preserve player cursor and animation state."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1)
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1600, 900))
    app.tick(0.016, [_key_event(pygame.K_RIGHT)])
    app.tick(0.016, [_key_event(pygame.K_RIGHT)])
    saved_cursor = app.player.cursor
    assert app.scene.is_animating

    resizes = [
        pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600),
        pygame.event.Event(pygame.WINDOWSIZECHANGED, x=1920, y=1080),
        pygame.event.Event(pygame.VIDEORESIZE, w=3440, h=1440),
        pygame.event.Event(pygame.WINDOWSIZECHANGED, x=1024, y=768),
    ]
    for ev in resizes:
        app.tick(0.05, [ev])
        app.draw()
        assert app.player.cursor == saved_cursor

    assert app.surface.get_size() == (1024, 768)
    assert app.scale.stage.width > 0 and app.scale.stage.height > 0


def test_unknown_stepkind_and_semantic_renders_safely() -> None:
    """Unknown forward-compatible StepKind/Semantic renders cleanly via GenericScene fallback."""
    base = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=3, n_passes=1)
    snap = make_snapshot(
        "future_tensor",
        TensorRole.OUTPUT,
        np.ones((3, 8), dtype=np.float32),
        ("token", "model_dim"),
        token_labels=("a", "b", "c"),
    )
    future_step = Step(
        id="p0.future_op",
        kind="future_op",
        semantic="future.experimental_layer",
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=int(DetailLevel.STORY),
        module_path="future.mod",
        tensors=(snap,),
        fidelity=Fidelity("unverified", note="Future custom op"),
    )
    mod_pass = replace(base.passes[0], steps=(future_step,), prediction=None)
    mod_trace = replace(base, passes=(mod_pass,))

    app = App(
        mod_trace,
        PresenterSettings(include_title=False, include_end=False),
        headless=True,
        viewport_size=(1280, 720),
    )
    assert isinstance(app.scene, GenericScene)
    app.tick(0.2, [])
    app.draw()
    assert app.player.total_steps == 1
    assert app.player.current is not None
    assert app.player.current.step_id == "p0.future_op"

    # Bounded navigation commands on 1-step EOS-only trace do not crash or wrap
    app.tick(
        0.016,
        [
            _key_event(pygame.K_RIGHT),
            _key_event(pygame.K_LEFT),
            _key_event(pygame.K_RIGHT, pygame.KMOD_SHIFT),
            _key_event(pygame.K_LEFT, pygame.KMOD_SHIFT),
            _key_event(pygame.K_RIGHT, pygame.KMOD_CTRL),
            _key_event(pygame.K_HOME),
            _key_event(pygame.K_END),
        ],
    )
    app.draw()
    assert app.player.cursor == 0


def test_single_step_eos_empty_blocks_app_navigation() -> None:
    """App handles single-step EOS pass with prediction=None and zero block steps gracefully."""
    base = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=2, n_passes=1)
    only_prompt = replace(base.passes[0], steps=(base.passes[0].steps[0],), prediction=None)
    minimal_trace = replace(base, passes=(only_prompt,))

    app = App(
        minimal_trace,
        PresenterSettings(include_title=False, include_end=False),
        headless=True,
        viewport_size=(1280, 720),
    )
    assert app.player.total_steps == 1
    for cmd_key, cmd_mod in [
        (pygame.K_RIGHT, 0),
        (pygame.K_LEFT, 0),
        (pygame.K_d, 0),
        (pygame.K_p, 0),
    ]:
        app.tick(0.5, [_key_event(cmd_key, cmd_mod)])
        app.draw()
    assert app.player.cursor == 0


def test_double_esc_quit_timing_state_machine() -> None:
    """First Esc arms quit; Esc after > 2.0s re-arms instead of quitting; <= 2.0s exits."""
    trace = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=2, n_passes=1)
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1280, 720))

    # 1. First Esc -> armed, does NOT quit
    app.tick(0.016, [_key_event(pygame.K_ESCAPE)])
    assert not app.should_quit
    assert app._esc_timer == pytest.approx(DOUBLE_ESC_WINDOW_SEC)

    # 2. Elapsed > 2.0s -> expires, second Esc only re-arms
    app.tick(2.1, [])
    assert app._esc_timer is None
    assert not app.should_quit

    app.tick(0.016, [_key_event(pygame.K_ESCAPE)])
    assert not app.should_quit
    assert app._esc_timer is not None

    # 3. Any other command cancels armed timer
    app.tick(0.1, [_key_event(pygame.K_n)])
    assert app._esc_timer is None
    assert not app.should_quit

    # 4. Arm and confirm within 2.0s -> quits cleanly
    app.tick(0.016, [_key_event(pygame.K_ESCAPE)])
    assert not app.should_quit
    app.tick(0.5, [_key_event(pygame.K_ESCAPE)])
    assert app.should_quit


def test_attention_head_cycling_and_hit_test_in_generic_scene() -> None:
    """GenericScene handles head cycling on unknown 3-D tensors and produces valid tooltips."""
    base = make_fake_trace(n_layers=1, emb_dim=16, n_tokens=4, n_passes=1, n_heads=4)
    attn_step = base.step_by_id("p0.b0.att.scores")
    generic_3d = replace(attn_step, id="p0.future_3d", kind="future_3d")
    mod_trace = replace(base, passes=(replace(base.passes[0], steps=(generic_3d,)),))
    app = App(
        mod_trace,
        PresenterSettings(include_title=False, include_end=False),
        headless=True,
        viewport_size=(1280, 720),
    )
    app.draw()

    assert isinstance(app.scene, GenericScene)
    assert app.scene._head_index == 0
    app.tick(0.016, [_key_event(pygame.K_RIGHTBRACKET)])
    assert app.scene._head_index == 1
    app.tick(0.016, [_key_event(pygame.K_LEFTBRACKET)])
    assert app.scene._head_index == 0
    app.tick(0.016, [_key_event(pygame.K_a)])
    assert app.scene._average_heads is True
    app.draw()

    # Hit test over first tile returns a populated Tooltip
    panel = app.scene._panels[0]
    tt = app.scene.hit_test(panel.grid.cell_center(0, 0))
    assert tt is not None
    assert tt.lines
    assert app.scene.hit_test((0, 0)) is None


def test_subprocess_torch_free_playback_execution() -> None:
    """End-to-end headless playback runs with torch completely blocked from sys.meta_path."""
    script = """
import os, sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

class TorchBlocker:
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch is intentionally blocked in playback verification')
        return None

sys.meta_path.insert(0, TorchBlocker())

import pygame
from tilevis.core.config import PresenterSettings
from tilevis.core.serialization import load_trace
from tilevis.render.app import App

trace = load_trace('traces/samples/tiny_demo.trace.npz')
app = App(trace, PresenterSettings(detail_level=1), headless=True, viewport_size=(1280, 720))
while app.player.next():
    app._enter_current_step(animate=False)
    app.draw()
app.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
app.tick(0.2, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
assert app.should_quit
assert 'torch' not in sys.modules
"""
    res = subprocess.run(
        [sys.executable, "-s", "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Torch-free playback subprocess failed:\n{res.stderr}"
