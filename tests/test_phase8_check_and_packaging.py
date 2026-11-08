"""Phase 8 tests: Preflight diagnostics, packaging, clean-room paths, and weakness probes."""

from __future__ import annotations

import io
import shutil
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pygame
import pytest

from tilevis.check import run_checks
from tilevis.cli import main
from tilevis.core.config import PresenterSettings
from tilevis.core.serialization import load_trace
from tilevis.render.app import App
from tilevis.render.commands import Command

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "traces" / "samples"
SAMPLE_TRACE = SAMPLES_DIR / "tiny_demo.trace.npz"


def test_tilevis_check_valid_environment_returns_zero() -> None:
    """tilevis check returns exit code 0 and reports all passing diagnostics on sample traces."""
    buf = io.StringIO()
    code = run_checks([str(SAMPLE_TRACE)], out=buf)
    output = buf.getvalue()

    assert code == 0
    assert "[PASS]" in output
    assert "Result: All preflight checks PASSED" in output
    assert "tiny_demo.trace.npz" in output


def test_tilevis_check_missing_trace_returns_one() -> None:
    """tilevis check returns exit code 1 with actionable message when trace path does not exist."""
    buf = io.StringIO()
    code = run_checks(["nonexistent_path.trace.npz"], out=buf)
    output = buf.getvalue()

    assert code == 1
    assert "[FAIL]" in output
    assert "Trace file not found" in output


def test_tilevis_check_corrupt_trace_returns_one(tmp_path: Path) -> None:
    """tilevis check returns exit code 1 with clear diagnostic when archive is corrupted."""
    corrupt_file = tmp_path / "corrupted.trace.npz"
    corrupt_file.write_bytes(b"PK\x03\x04corrupted_zip_data_not_valid_npz")

    buf = io.StringIO()
    code = run_checks([str(corrupt_file)], out=buf)
    output = buf.getvalue()

    assert code == 1
    assert "[FAIL]" in output
    assert "Failed to load trace" in output or "Corrupted" in output


def test_tilevis_check_missing_font_asset_simulation() -> None:
    """Simulated missing font asset causes preflight check to fail with actionable error."""
    with patch("importlib.resources.files") as mock_files:
        mock_dir = MagicMock()
        mock_file = MagicMock()
        mock_file.read_bytes.return_value = b""  # empty corrupted font
        mock_dir.joinpath.return_value = mock_file
        mock_files.return_value = mock_dir

        buf = io.StringIO()
        code = run_checks([str(SAMPLE_TRACE)], out=buf)
        output = buf.getvalue()

        assert code == 1
        assert "Bundled font files are corrupted or unexpectedly small" in output


def test_tilevis_check_pygame_collision_simulation() -> None:
    """Simulated collision between legacy pygame and pygame-ce causes preflight check to fail."""
    dist_legacy = MagicMock()
    dist_legacy.metadata = {"Name": "pygame"}
    dist_ce = MagicMock()
    dist_ce.metadata = {"Name": "pygame-ce"}

    with patch("importlib.metadata.distributions", return_value=[dist_legacy, dist_ce]):
        buf = io.StringIO()
        code = run_checks([str(SAMPLE_TRACE)], out=buf)
        output = buf.getvalue()

        assert code == 1
        assert "Package collision" in output
        assert "pip uninstall" in output


def test_pyinstaller_spec_structure_and_exclusions() -> None:
    """PyInstaller spec exists and excludes heavy libraries (torch, matplotlib, etc.)."""
    spec_path = PROJECT_ROOT / "packaging" / "tilevis_player.spec"
    assert spec_path.is_file(), f"Missing PyInstaller spec at {spec_path}"

    content = spec_path.read_text(encoding="utf-8")

    # Mandatory exclusions from agents.md §9.8 Task 3
    mandatory_excludes = ["torch", "tiktoken", "matplotlib", "pytest", "hypothesis", "mypy"]
    for exc in mandatory_excludes:
        assert f"'{exc}'" in content or f'"{exc}"' in content, (
            f"Mandatory PyInstaller exclusion '{exc}' missing from spec"
        )

    # Verification of entry script
    assert "__main__.py" in content


def test_clean_room_korean_spaces_path_playback(tmp_path: Path) -> None:
    """Play back traces located in paths containing Korean characters and spaces."""
    korean_dir = tmp_path / "2026 발표 자료" / "발표용 traces"
    korean_dir.mkdir(parents=True)
    korean_trace_path = korean_dir / "작은_데모.trace.npz"

    # Copy sample trace
    shutil.copyfile(SAMPLE_TRACE, korean_trace_path)

    # 1. Preflight check
    buf = io.StringIO()
    ret_check = run_checks([str(korean_trace_path)], out=buf)
    assert ret_check == 0
    assert "작은_데모.trace.npz" in buf.getvalue()

    # 2. Headless App playback
    trace = load_trace(korean_trace_path)
    app = App(trace, PresenterSettings(), headless=True)
    app.draw()
    assert app.player.total_steps > 0
    app.scene.finish()
    app.dispatch_command(Command.NEXT)
    assert app.player.cursor == 1


def test_readonly_directory_playback(tmp_path: Path) -> None:
    """Playback from a read-only directory (locked USB drive) succeeds with zero write attempts."""
    ro_dir = tmp_path / "locked_usb_drive"
    ro_dir.mkdir()
    ro_trace = ro_dir / "demo.trace.npz"
    shutil.copyfile(SAMPLE_TRACE, ro_trace)

    # Set directory and file read-only (0o555 for dir, 0o444 for file)
    ro_trace.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    ro_dir.chmod(
        stat.S_IREAD | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )

    try:
        # Preflight check on read-only trace
        buf = io.StringIO()
        ret = run_checks([str(ro_trace)], out=buf)
        assert ret == 0

        # Load and play trace without write attempts
        trace = load_trace(ro_trace)
        app = App(trace, PresenterSettings(), headless=True)
        app.draw()
        app.scene.finish()
        app.dispatch_command(Command.NEXT)
        assert app.player.cursor == 1
    finally:
        # Restore write permissions for tmp_path cleanup
        ro_dir.chmod(stat.S_IRWXU)
        ro_trace.chmod(stat.S_IRWXU)


def test_focus_loss_and_f11_mode_toggling() -> None:
    """Focus loss and F11 mode toggling handle modifier resetting and cursor preservation."""
    trace = load_trace(SAMPLE_TRACE)
    app = App(trace, PresenterSettings(), headless=True, viewport_size=(1920, 1080))
    app.player.cursor = 3
    app._enter_current_step(animate=False)

    # 1. Simulate WINDOWFOCUSLOST
    ev_lost = pygame.event.Event(getattr(pygame, "WINDOWFOCUSLOST", pygame.USEREVENT))
    app.tick(0.016, [ev_lost])
    assert not app.window.focused

    # 2. Simulate WINDOWFOCUSGAINED
    ev_gained = pygame.event.Event(getattr(pygame, "WINDOWFOCUSGAINED", pygame.USEREVENT + 1))
    app.tick(0.016, [ev_gained])
    assert app.window.focused
    assert app.player.cursor == 3

    # 3. Simulate F11 fullscreen toggle command
    app.dispatch_command(Command.TOGGLE_FULLSCREEN)
    assert app.player.cursor == 3
    app.draw()


def test_cli_user_error_hygiene_no_raw_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid CLI invocations return code 1 with clean messages and no tracebacks."""
    # 1. Missing trace file on play
    code1 = main(["play", "definitely_nonexistent_trace.npz", "--headless"])
    assert code1 == 1
    err1 = capsys.readouterr().err
    assert "Error:" in err1
    assert "Traceback" not in err1

    # 2. Missing profile on record
    code2 = main(["record", "--profile", "nonexistent.toml", "--prompt", "hi", "-o", "out.npz"])
    assert code2 == 1
    err2 = capsys.readouterr().err
    assert "Error:" in err2
    assert "Traceback" not in err2
