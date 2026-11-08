"""Phase 6 integration: record every compat_matrix model and render all dedicated scenes."""

from __future__ import annotations

from pathlib import Path

import pygame
import pytest

torch = pytest.importorskip("torch")

from tests.compat_matrix import COMPAT_MATRIX  # noqa: E402
from tests.fixtures.tiny_models import save_matrix_checkpoint  # noqa: E402

from tilevis.core.config import PresenterSettings, RecordSettings  # noqa: E402
from tilevis.core.kinds import DetailLevel  # noqa: E402
from tilevis.probe.loading.factory import build_model  # noqa: E402
from tilevis.probe.loading.profile import load_profile  # noqa: E402
from tilevis.probe.recorder import TraceRecorder  # noqa: E402
from tilevis.render.app import App  # noqa: E402

FAST_MATRIX_KEYS = [k for k in COMPAT_MATRIX if k != "gpt2_small_shape"]


def _verify_recorded_playback(matrix_key: str, tmp_path: Path) -> None:
    """Save checkpoint, load via factory, record trace, and render 3 frames/step across details."""
    entry = COMPAT_MATRIX[matrix_key]
    _ckpt_path, profile_path = save_matrix_checkpoint(
        entry,
        tmp_path / matrix_key,
        vocab_size=50257,
        seed=42,
    )
    profile = load_profile(profile_path)
    built = build_model(profile)
    recorder = TraceRecorder(built, RecordSettings(new_tokens=1, strategy="greedy"))
    trace = recorder.record("Every effort moves you")

    next_ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    for detail in (DetailLevel.STORY, DetailLevel.STANDARD, DetailLevel.DEEP):
        settings = PresenterSettings(
            detail_level=int(detail),
            detail_blocks="first",
            advance_skips_animation=True,
        )
        app = App(trace, settings, headless=True, viewport_size=(1280, 720))
        total = app.player.total_steps
        assert total >= 3
        for idx in range(total):
            assert app.player.cursor == idx
            # t=0
            app.tick(0.0, [])
            app.draw()
            # mid-animation
            app.tick(0.55, [])
            app.draw()
            # completed end state
            app.scene.finish()
            app.draw()
            assert not app.scene.is_animating
            if app.player.can_next:
                app.tick(0.016, [next_ev])


@pytest.mark.parametrize("matrix_key", FAST_MATRIX_KEYS)
def test_real_recorded_compat_models_render_three_frames_per_step(
    matrix_key: str, tmp_path: Path
) -> None:
    """Real recorded traces from 7 compat_matrix variants play cleanly at t=0, mid, and end."""
    _verify_recorded_playback(matrix_key, tmp_path)


@pytest.mark.slow
def test_real_recorded_gpt2_small_shape_renders_three_frames_per_step(tmp_path: Path) -> None:
    """GPT-2 small-shaped recorded trace plays cleanly across STORY, STANDARD, and DEEP."""
    _verify_recorded_playback("gpt2_small_shape", tmp_path)
