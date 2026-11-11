"""Unit tests for WindowManager, platform utilities, and scene registry dispatch."""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace
from unittest import mock

import pygame
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.check import check_display_modes
from tilevis.render import platform as plat
from tilevis.render.app import App
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.scenes.generic import GenericScene
from tilevis.render.scenes.registry import SCENES, create_scene
from tilevis.render.window import WindowManager, build_window_icon


def test_build_window_icon_dimensions() -> None:
    """Window icon creates a valid 32x32 RGBA Surface."""
    icon = build_window_icon()
    assert isinstance(icon, pygame.Surface)
    assert icon.get_size() == (32, 32)


def test_enable_high_dpi_sdl_hints_and_win32_hierarchy() -> None:
    """enable_high_dpi sets SDL hints and tries Context -> Awareness -> Aware hierarchy."""
    # Ensure SDL environment hints are populated
    with mock.patch.object(plat.sys, "platform", "linux"):
        assert plat.enable_high_dpi() is False
        assert os.environ.get("SDL_WINDOWS_DPI_AWARENESS") == "permonitorv2"
        assert os.environ.get("SDL_VIDEO_HIGHDPI_DISABLED") == "0"

    # 1. Win10 1703+: user32.SetProcessDpiAwarenessContext(-4) succeeds
    mock_user32_ctx = SimpleNamespace(SetProcessDpiAwarenessContext=mock.Mock(return_value=True))
    mock_windll_ctx = SimpleNamespace(user32=mock_user32_ctx)
    with (
        mock.patch.object(plat.sys, "platform", "win32"),
        mock.patch.object(ctypes, "windll", mock_windll_ctx, create=True),
    ):
        assert plat.enable_high_dpi() is True
        mock_user32_ctx.SetProcessDpiAwarenessContext.assert_called_once()

    # 2. Win8.1: SetProcessDpiAwarenessContext fails -> fallback to shcore.SetProcessDpiAwareness(2)
    mock_user32_fail = SimpleNamespace(
        SetProcessDpiAwarenessContext=mock.Mock(side_effect=AttributeError("no context"))
    )
    mock_shcore = SimpleNamespace(SetProcessDpiAwareness=mock.Mock(return_value=0))
    mock_windll_sh = SimpleNamespace(user32=mock_user32_fail, shcore=mock_shcore)
    with (
        mock.patch.object(plat.sys, "platform", "win32"),
        mock.patch.object(ctypes, "windll", mock_windll_sh, create=True),
    ):
        assert plat.enable_high_dpi() is True
        mock_shcore.SetProcessDpiAwareness.assert_called_once_with(2)

    # 3. Vista: shcore fails -> fallback to user32.SetProcessDPIAware()
    mock_user32_aware = SimpleNamespace(
        SetProcessDpiAwarenessContext=mock.Mock(side_effect=AttributeError),
        SetProcessDPIAware=mock.Mock(return_value=1),
    )
    mock_shcore_fail = SimpleNamespace(
        SetProcessDpiAwareness=mock.Mock(side_effect=OSError("no shcore"))
    )
    mock_windll_aware = SimpleNamespace(user32=mock_user32_aware, shcore=mock_shcore_fail)
    with (
        mock.patch.object(plat.sys, "platform", "win32"),
        mock.patch.object(ctypes, "windll", mock_windll_aware, create=True),
    ):
        assert plat.enable_high_dpi() is True
        mock_user32_aware.SetProcessDPIAware.assert_called_once()


def test_default_windowed_size_and_detect_display_info() -> None:
    """Verify smart default windowed size across 768p, 982p, 1080p, 1440p, and 4K displays."""
    assert plat.default_windowed_size(3840, 2160) == (1920, 1080)
    assert plat.default_windowed_size(2560, 1440) == (1920, 1080)
    assert plat.default_windowed_size(1920, 1080) == (1632, 918)
    assert plat.default_windowed_size(1512, 982) == (1280, 720)
    assert plat.default_windowed_size(1366, 768) == (1152, 648)

    info = plat.detect_display_info(0)
    assert isinstance(info, plat.DisplayInfo)
    assert info.desktop_w > 0 and info.desktop_h > 0
    assert info.default_windowed[0] > 0 and info.default_windowed[1] > 0


def test_window_manager_dpi_tracking_and_map_point() -> None:
    """WindowManager tracks logical/drawable sizes and maps mouse points accurately."""
    wm = WindowManager(mode="windowed", windowed_size=(1280, 720))
    # Standard 1.0x scale
    wm.logical_size = (1920, 1080)
    wm.drawable_size = (1920, 1080)
    wm._update_dpi_scale()
    assert wm.dpi_scale == (1.0, 1.0)
    assert wm.map_point((100, 200)) == (100, 200)

    # 2.0x Retina scale (e.g. 1512x982 window -> 3024x1964 surface)
    wm.logical_size = (1512, 982)
    wm.drawable_size = (3024, 1964)
    wm._update_dpi_scale()
    assert wm.dpi_scale == (2.0, 2.0)
    assert wm.map_point((100, 200)) == (200, 400)

    # 1.5x Windows 150% HiDPI scale (e.g. 1280x720 window -> 1920x1080 surface)
    wm.logical_size = (1280, 720)
    wm.drawable_size = (1920, 1080)
    wm._update_dpi_scale()
    assert wm.dpi_scale == (1.5, 1.5)
    assert wm.map_point((100, 200)) == (150, 300)


def test_window_manager_modes_toggle_and_resize() -> None:
    """WindowManager switches modes, handles resize events, and toggles F11 cleanly."""
    wm = WindowManager(mode="windowed", windowed_size=(1024, 600))
    surf_headless = wm.create(headless=True)
    assert surf_headless.get_size() == (1024, 600)
    assert wm.logical_size == (1024, 600)
    assert wm.drawable_size == (1024, 600)

    surf_toggled = wm.toggle_fullscreen(headless=True)
    assert wm.mode == "borderless"
    assert surf_toggled.get_width() >= 320

    surf_resized = wm.handle_resize(800, 500, headless=True)
    assert surf_resized.get_size() == (800, 500)
    assert wm.logical_size == (800, 500)
    assert wm.drawable_size == (800, 500)

    # Exercise dummy SDL video display lifecycle
    try:
        surf_real = wm.create(headless=False)
        assert surf_real.get_width() >= 320
        wm.present()
        surf_fs = wm.toggle_fullscreen(headless=False)
        assert surf_fs.get_width() >= 320
        wm.handle_resize(960, 540, headless=False)
        assert wm.size[0] >= 320
    finally:
        wm.destroy()


def test_check_display_modes_reports_high_dpi_status() -> None:
    """check_display_modes reports desktop size and High-DPI backing capability."""
    ok, msg = check_display_modes()
    assert ok is True
    assert "High-DPI backing:" in msg
    assert "default windowed:" in msg


def test_weakness_hidpi_click_mapping_retina_and_windows() -> None:
    """Weakness probe: verify mouse clicks in logical space hit physical buttons accurately."""
    trace = make_fake_trace(n_passes=1, n_layers=2)
    app = App(trace, headless=True, viewport_size=(1920, 1080))
    app.draw()

    # 1. Simulate 2.0x Retina scaling: logical (1512, 982) -> physical (3024, 1964)
    app.window.logical_size = (1512, 982)
    app.window.drawable_size = (3024, 1964)
    app.window._update_dpi_scale()
    app.surface = pygame.Surface(app.window.drawable_size)
    app.scale = UiScale.from_viewport(3024, 1964)
    app.draw()

    next_btn = next(b for b in app.hud.buttons.buttons if b.cmd == Command.NEXT)
    next_rect = next_btn.rect
    assert next_rect.width > 0
    # Center of next button in physical coordinates
    phys_cx, phys_cy = next_rect.centerx, next_rect.centery
    # Map back to logical coordinates: pos / dpi_scale
    logical_cx = round(phys_cx / app.window.dpi_scale[0])
    logical_cy = round(phys_cy / app.window.dpi_scale[1])

    app.scene.finish()
    assert not app.scene.is_animating
    cur_idx = app.player.cursor
    # Fire synthetic MOUSEBUTTONDOWN at logical coords
    click_ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(logical_cx, logical_cy), button=1)
    app.tick(0.016, [click_ev])
    assert app.player.cursor == cur_idx + 1

    # 2. Simulate 1.5x Windows HiDPI scaling: logical (1280, 720) -> physical (1920, 1080)
    app.window.logical_size = (1280, 720)
    app.window.drawable_size = (1920, 1080)
    app.window._update_dpi_scale()
    app.surface = pygame.Surface(app.window.drawable_size)
    app.scale = UiScale.from_viewport(1920, 1080)
    app.draw()

    next_btn_15 = next(b for b in app.hud.buttons.buttons if b.cmd == Command.NEXT)
    next_rect_15 = next_btn_15.rect
    phys_cx_15, phys_cy_15 = next_rect_15.centerx, next_rect_15.centery
    logical_cx_15 = round(phys_cx_15 / app.window.dpi_scale[0])
    logical_cy_15 = round(phys_cy_15 / app.window.dpi_scale[1])

    app.scene.finish()
    assert not app.scene.is_animating
    cur_idx_15 = app.player.cursor
    click_ev_15 = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, pos=(logical_cx_15, logical_cy_15), button=1
    )
    app.tick(0.016, [click_ev_15])
    assert app.player.cursor == cur_idx_15 + 1


def test_weakness_live_resize_and_f11_mode_switch_crispness() -> None:
    """Weakness probe: live resize across resolutions preserves animation and updates scale."""
    trace = make_fake_trace(n_passes=1, n_layers=2)
    app = App(trace, headless=True, viewport_size=(1280, 720))

    # Move to a step with cards / TileGrids
    app.player.next()
    app._enter_current_step(animate=True)
    if hasattr(app.scene, "_timeline"):
        app.scene._timeline.elapsed = 0.5  # mid-animation

    # 1. Resize to 1280x720
    app.resize(1280, 720)
    assert app.scale.viewport_w == 1280
    assert app.scene._ctx is not None
    assert app.scene._ctx.scale.viewport_w == 1280

    # 2. Resize to 1920x1080 -> exact 1.0 snapping
    app.resize(1920, 1080)
    assert app.scale.s == 1.0
    assert app.scene._ctx.scale.s == 1.0

    # 3. Resize to 3024x1964 -> Retina high resolution
    app.resize(3024, 1964)
    assert app.scale.viewport_w == 3024
    assert app.scene._ctx.scale.viewport_w == 3024
    assert getattr(app.scene, "_timeline", None) is not None
    # Progress preserved
    assert app.scene._timeline.progress > 0.0

    app.draw()
    assert app.surface.get_size() == (3024, 1964)


def test_scene_registry_fallback_and_custom_registration() -> None:
    """create_scene returns GenericScene fallback or registered custom Scene subclass."""
    assert isinstance(create_scene("UNKNOWN_FUTURE_KIND"), GenericScene)

    key = "CUSTOM_TEST_SCENE_KIND"
    assert key not in SCENES

    @SCENES.register(key)
    class _DummyScene(GenericScene):
        pass

    try:
        scene = create_scene(key)
        assert isinstance(scene, _DummyScene)
    finally:
        SCENES._entries.pop(key, None)
