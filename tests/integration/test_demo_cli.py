"""Integration test for `tilevis demo` subcommand."""

from __future__ import annotations

import pytest

from tilevis.cli import main


def test_tilevis_demo_headless_smoke() -> None:
    """tilevis demo --headless runs in-memory recording and one headless playback tick."""
    pytest.importorskip("torch")

    # Run default demo headlessly
    ret = main(["demo", "--headless", "--prompt", "A journey of a thousand miles"])
    assert ret == 0


def test_tilevis_demo_with_custom_settings() -> None:
    """tilevis demo accepts --theme and --detail overrides."""
    pytest.importorskip("torch")

    ret = main(["demo", "--headless", "--theme", "light", "--detail", "1"])
    assert ret == 0
