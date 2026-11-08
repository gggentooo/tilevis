"""Integration tests for `tilevis play` CLI subcommand."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tilevis.cli import main

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_TRACE = PROJECT_ROOT / "traces" / "samples" / "tiny_demo.trace.npz"


def test_cli_play_headless_sample_trace() -> None:
    """`tilevis play` loads sample trace, applies overrides, and exits cleanly in headless mode."""
    code = main(
        [
            "play",
            str(SAMPLE_TRACE),
            "--presenter",
            str(PROJECT_ROOT / "presenter" / "default.toml"),
            "--window",
            "windowed",
            "--theme",
            "light",
            "--detail",
            "1",
            "--headless",
        ]
    )
    assert code == 0


def test_cli_play_errors_exit_code_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Missing trace or bad presenter path returns clean error and exit code 1."""
    code1 = main(["play", str(tmp_path / "missing.trace.npz"), "--headless"])
    assert code1 == 1
    assert "not found" in capsys.readouterr().err.lower()

    code2 = main(
        [
            "play",
            str(SAMPLE_TRACE),
            "--presenter",
            str(tmp_path / "missing_presenter.toml"),
            "--headless",
        ]
    )
    assert code2 == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_cli_play_subprocess_without_torch() -> None:
    """Invoke `python -m tilevis play --headless` in subprocess with torch blocked."""
    wrapper = (
        "import os, sys\n"
        "os.environ['SDL_VIDEODRIVER'] = 'dummy'\n"
        "os.environ['SDL_AUDIODRIVER'] = 'dummy'\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from tilevis.cli import main\n"
        f"rc = main(['play', {str(SAMPLE_TRACE)!r}, '--headless'])\n"
        "assert 'torch' not in sys.modules\n"
        "sys.exit(rc)\n"
    )
    res = subprocess.run(
        [sys.executable, "-s", "-c", wrapper],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
