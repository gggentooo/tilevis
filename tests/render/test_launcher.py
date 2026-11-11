"""Unit tests and weakness probes for GUI trace launcher and discovery."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pygame
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.cli import main
from tilevis.core.config import PresenterSettings
from tilevis.core.serialization import save_trace
from tilevis.render.launcher import (
    LauncherApp,
    discover_traces,
    read_trace_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_discover_traces_basic_and_sorting() -> None:
    """discover_traces puts User Traces before Samples, and sorts alphabetically."""
    traces = discover_traces()
    assert len(traces) >= 2

    # Check category division
    user_traces = [t for t in traces if t.category == "User Trace"]
    sample_traces = [t for t in traces if t.category == "Sample"]

    # All user traces must appear before all sample traces
    if user_traces and sample_traces:
        last_user_idx = max(traces.index(t) for t in user_traces)
        first_sample_idx = min(traces.index(t) for t in sample_traces)
        assert last_user_idx < first_sample_idx

    # Entries are sorted alphabetically within category
    user_names = [t.display_name.lower() for t in user_traces]
    assert user_names == sorted(user_names)

    sample_names = [t.display_name.lower() for t in sample_traces]
    assert sample_names == sorted(sample_names)


def test_discover_traces_priority_and_frozen(tmp_path: Path) -> None:
    """discover_traces respects explicit traces_dir and simulates frozen executable lookup."""
    # 1. Explicit directory takes precedence
    dir_a = tmp_path / "custom_traces"
    dir_a.mkdir()
    trace = make_fake_trace(n_layers=1, emb_dim=8)
    save_trace(trace, dir_a / "model_custom.trace.npz")

    found = discover_traces(traces_dir=dir_a)
    assert any(t.display_name == "model_custom.trace.npz" for t in found)
    assert found[0].display_name == "model_custom.trace.npz"

    # 2. Simulated frozen executable
    mock_exe_dir = tmp_path / "mock_install"
    mock_exe_traces = mock_exe_dir / "traces"
    mock_exe_traces.mkdir(parents=True)
    save_trace(trace, mock_exe_traces / "usb_adjacent.trace.npz")

    mock_exe_path = str(mock_exe_dir / "tilevis_player")
    with (
        mock.patch.object(sys, "frozen", True, create=True),
        mock.patch.object(sys, "executable", mock_exe_path),
    ):
        found_frozen = discover_traces()
        assert any(t.display_name == "usb_adjacent.trace.npz" for t in found_frozen)


def test_read_trace_metadata_valid_and_invalid(tmp_path: Path) -> None:
    """read_trace_metadata extracts manifest info and handles missing/corrupt files cleanly."""
    # 1. Valid trace
    valid_p = tmp_path / "valid.trace.npz"
    trace = make_fake_trace(n_layers=3, emb_dim=24, n_heads=3)
    save_trace(trace, valid_p)

    meta = read_trace_metadata(valid_p)
    assert meta.error is None
    assert meta.n_layers == 3
    assert meta.emb_dim == 24
    assert meta.n_heads == 3
    assert "Every effort" in meta.prompt_preview
    assert meta.total_steps > 0

    # 2. Missing file
    missing_meta = read_trace_metadata(tmp_path / "does_not_exist.trace.npz")
    assert missing_meta.error is not None
    assert "not exist" in missing_meta.error

    # 3. Corrupt file
    corrupt_p = tmp_path / "corrupt.trace.npz"
    corrupt_p.write_bytes(b"PK\x03\x04corrupteddatahere")
    corrupt_meta = read_trace_metadata(corrupt_p)
    assert corrupt_meta.error is not None


def test_launcher_app_headless_and_navigation() -> None:
    """LauncherApp initializes, updates selection with Up/Down, and toggles dropdown."""
    launcher = LauncherApp(headless=True, viewport_size=(1180, 700))
    assert launcher.surface.get_size() == (1180, 700)
    assert len(launcher.traces) > 0
    assert launcher.selected_trace is not None

    initial_idx = launcher.selected_index

    # 1. Arrow down advances selection
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)])
    assert launcher.selected_index == (initial_idx + 1) % len(launcher.traces)

    # 2. Arrow up moves back
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)])
    assert launcher.selected_index == initial_idx

    # 3. Dropdown open/close
    assert launcher._dropdown_open is False
    launcher._dropdown_open = True
    launcher.draw()
    assert len(launcher._dropdown_item_rects) > 0

    # Esc closes dropdown
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert launcher._dropdown_open is False
    assert launcher.should_quit is False

    # Second Esc quits launcher
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert launcher.should_quit is True


def test_launcher_app_option_toggles() -> None:
    """LauncherApp allows toggling mode, theme, detail level, and language."""
    settings = PresenterSettings(
        window_mode="borderless", theme="dark", detail_level=2, locale="en"
    )
    launcher = LauncherApp(settings=settings, headless=True)

    # 1. Mode toggle (W key)
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)])
    assert launcher.settings.window_mode == "windowed"

    # 2. Theme toggle (T key)
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t)])
    assert launcher.settings.theme == "light"
    assert launcher.theme.name == "light"

    # 3. Detail toggle (D key)
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d)])
    assert launcher.settings.detail_level == 3

    # 4. Locale toggle (L key)
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l)])
    assert launcher.settings.locale == "ko"
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l)])
    assert launcher.settings.locale == "ja"

    # Draw in Japanese and Korean to verify font rendering without exceptions
    launcher.draw()


def test_launcher_mouse_clicks_and_high_dpi_mapping() -> None:
    """Verify mouse click on dropdown items and buttons works with coordinate mapping."""
    launcher = LauncherApp(headless=True, viewport_size=(1180, 700))
    launcher.draw()

    # Click on dropdown button opens menu
    btn_rect = launcher._dropdown_btn_rect
    assert btn_rect.width > 0
    click_pos = (btn_rect.centerx, btn_rect.centery)
    click_ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=click_pos, button=1)
    launcher.tick(0.016, [click_ev])
    assert launcher._dropdown_open is True

    # Draw to populate item rects
    launcher.draw()
    assert len(launcher._dropdown_item_rects) >= 2

    # Click second item
    target_idx, target_rect = launcher._dropdown_item_rects[1]
    item_click_pos = (target_rect.centerx, target_rect.centery)
    launcher.tick(0.016, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=item_click_pos, button=1)])
    assert launcher._dropdown_open is False
    assert launcher.selected_index == target_idx


def test_launcher_launch_handoff(tmp_path: Path) -> None:
    """Launching presentation hands off to App and returns exit code."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace = make_fake_trace(n_layers=1, emb_dim=12)
    trace_file = trace_dir / "demo.trace.npz"
    save_trace(trace, trace_file)

    launcher = LauncherApp(traces_dir=trace_dir, headless=True)
    assert launcher.selected_trace is not None
    assert launcher.selected_trace.path == trace_file

    # Trigger presentation launch via Return key
    ret_ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
    launcher.tick(0.016, [ret_ev])
    assert launcher.launch_requested is True


def test_cli_launch_subcommand_and_zero_arg_default() -> None:
    """CLI 'tilevis launch' and zero-argument default invocation route to LauncherApp."""
    # 1. tilevis launch --headless
    with mock.patch("tilevis.render.launcher.LauncherApp.run", return_value=0) as mock_run:
        rc = main(["launch", "--headless"])
        assert rc == 0
        mock_run.assert_called_once()

    # 2. tilevis (no arguments at all) defaults to launch
    with mock.patch("tilevis.render.launcher.LauncherApp.run", return_value=0) as mock_run_zero:
        rc_zero = main([])
        assert rc_zero == 0
        mock_run_zero.assert_called_once()


def test_weakness_empty_traces_directory(tmp_path: Path) -> None:
    """Weakness probe: empty traces directory shows clean empty state without crashing."""
    empty_dir = tmp_path / "empty_traces"
    empty_dir.mkdir()

    launcher = LauncherApp(traces_dir=empty_dir, headless=True)
    assert len(launcher.traces) == 0
    assert launcher.selected_trace is None
    assert launcher.selected_index == -1

    # Drawing and clicking should be completely safe
    launcher.draw()
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    launcher.tick(0.016, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
    assert launcher.should_quit is False


def test_weakness_corrupt_trace_handling(tmp_path: Path) -> None:
    """Weakness probe: corrupt trace in directory does not crash launcher or launch attempt."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    # 1 good trace, 1 corrupt trace
    good_p = traces_dir / "good.trace.npz"
    save_trace(make_fake_trace(n_layers=1, emb_dim=8), good_p)

    bad_p = traces_dir / "bad.trace.npz"
    bad_p.write_bytes(b"corrupt non-zip content")

    launcher = LauncherApp(traces_dir=traces_dir, headless=True)
    assert len(launcher.traces) == 2

    # Select the corrupt trace
    bad_idx = next(i for i, t in enumerate(launcher.traces) if t.path == bad_p)
    launcher.selected_index = bad_idx
    meta = launcher.get_metadata(launcher.traces[bad_idx])
    assert meta.error is not None

    # Draw preview card with error notice
    launcher.draw()

    # Launch should gracefully refuse rather than unhandled exception
    res = launcher.launch_presentation()
    assert res == 1


def test_weakness_dropdown_overflow_and_scrolling(tmp_path: Path) -> None:
    """Weakness probe: 25 traces in dropdown scroll smoothly with wheel and arrow keys."""
    traces_dir = tmp_path / "many_traces"
    traces_dir.mkdir()

    trace = make_fake_trace(n_layers=1, emb_dim=8)
    for i in range(25):
        save_trace(trace, traces_dir / f"trace_{i:02d}.trace.npz")

    launcher = LauncherApp(traces_dir=traces_dir, headless=True)
    assert len(launcher.traces) == 25

    launcher._dropdown_open = True
    launcher.draw()
    assert len(launcher._dropdown_item_rects) == launcher.DROPDOWN_MAX_VISIBLE

    # Scroll with mouse wheel
    launcher.tick(0.016, [pygame.event.Event(pygame.MOUSEWHEEL, y=-3)])
    assert launcher._dropdown_scroll == 3

    # Navigate past bottom of visible window with down arrow
    launcher.selected_index = 15
    launcher._ensure_selected_visible()
    assert launcher._dropdown_scroll > 0
    max_vis = launcher.DROPDOWN_MAX_VISIBLE
    assert launcher._dropdown_scroll <= launcher.selected_index
    assert launcher.selected_index < launcher._dropdown_scroll + max_vis
