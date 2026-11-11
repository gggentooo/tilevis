"""Phase 9 tests: High-DPI resolutions across compat matrix and sample traces."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel
from tilevis.core.serialization import load_trace
from tilevis.render.app import App

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "traces" / "samples"
SAMPLE_NAMES = [
    "tiny_ch07.trace.npz",
    "tiny_demo.trace.npz",
    "big_en.trace.npz",
    "big_es.trace.npz",
    "big_ja.trace.npz",
    "big_ko.trace.npz",
]

RESOLUTIONS = [
    (1920, 1080),
    (1512, 982),
    (2560, 1440),
    (3024, 1964),
]


@pytest.mark.parametrize("res", RESOLUTIONS)
@pytest.mark.parametrize("matrix_name", list(COMPAT_MATRIX.keys()))
def test_compat_matrix_resolutions(res: tuple[int, int], matrix_name: str) -> None:
    """Render each model in compat_matrix at standard 1080p, Retina, and 1440p/4K resolutions."""
    entry = COMPAT_MATRIX[matrix_name]
    trace = make_fake_trace(
        n_layers=min(2, entry.n_layers),
        emb_dim=min(32, entry.emb_dim),
        n_tokens=4,
        n_passes=1,
        n_heads=entry.n_heads,
        seed=42,
    )
    app = App(trace, headless=True, viewport_size=res)
    assert app.surface.get_size() == res
    app.draw()

    # Step forward through a few steps
    for _ in range(min(5, app.player.total_steps - 1)):
        app.player.next()
        app._enter_current_step(animate=False)
        app.draw()


@pytest.mark.parametrize("res", RESOLUTIONS)
@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_sample_traces_resolutions(res: tuple[int, int], sample_name: str) -> None:
    """Render all 6 sample traces headlessly across all 4 target drawable resolutions."""
    sample_path = SAMPLES_DIR / sample_name
    assert sample_path.is_file(), f"Missing sample trace: {sample_name}"

    trace = load_trace(sample_path)
    locale = "ja" if "ja" in sample_name else ("ko" if "ko" in sample_name else "en")
    settings = PresenterSettings(locale=locale, detail_level=int(DetailLevel.STANDARD))

    app = App(trace, settings=settings, headless=True, viewport_size=res)
    assert app.surface.get_size() == res
    app.draw()

    # Walk 3 steps to exercise title, tokenize, and embedding/norm
    for _ in range(min(3, app.player.total_steps - 1)):
        app.player.next()
        app._enter_current_step(animate=False)
        app.draw()


def test_ch07_remains_untouched() -> None:
    """Enforce Ground Rule 1 & R6: ch07/ remains strictly unmodified."""
    res = subprocess.run(
        ["git", "status", "--porcelain", "ch07/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", f"ch07/ has modified files: {res.stdout}"
