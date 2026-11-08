"""Unit tests for WindowManager, platform utilities, and scene registry dispatch."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest import mock

import pygame

from tilevis.render import platform as plat
from tilevis.render.scenes.generic import GenericScene
from tilevis.render.scenes.registry import SCENES, create_scene
from tilevis.render.window import WindowManager, build_window_icon


def test_build_window_icon_dimensions() -> None:
    """Window icon creates a valid 32x32 RGBA Surface."""
    icon = build_window_icon()
    assert isinstance(icon, pygame.Surface)
    assert icon.get_size() == (32, 32)


def test_enable_high_dpi_noop_on_non_windows_and_mocked_win32() -> None:
    """enable_high_dpi handles non-Windows no-op, shcore, and user32 fallback safely."""
    with mock.patch.object(plat.sys, "platform", "linux"):
        assert plat.enable_high_dpi() is False

    mock_shcore = SimpleNamespace(SetProcessDpiAwareness=mock.Mock())
    mock_windll = SimpleNamespace(shcore=mock_shcore, user32=SimpleNamespace())
    with (
        mock.patch.object(plat.sys, "platform", "win32"),
        mock.patch.object(ctypes, "windll", mock_windll, create=True),
    ):
        assert plat.enable_high_dpi() is True
        mock_shcore.SetProcessDpiAwareness.assert_called_once_with(2)

    # Fallback when shcore fails -> user32.SetProcessDPIAware()
    mock_user32 = SimpleNamespace(SetProcessDPIAware=mock.Mock())
    bad_shcore = SimpleNamespace(SetProcessDpiAwareness=mock.Mock(side_effect=OSError("no shcore")))
    mock_windll_fb = SimpleNamespace(shcore=bad_shcore, user32=mock_user32)
    with (
        mock.patch.object(plat.sys, "platform", "win32"),
        mock.patch.object(ctypes, "windll", mock_windll_fb, create=True),
    ):
        assert plat.enable_high_dpi() is True
        mock_user32.SetProcessDPIAware.assert_called_once()


def test_get_desktop_size_fallback_and_dummy() -> None:
    """get_desktop_size returns positive dimensions cleanly."""
    w, h = plat.get_desktop_size(0)
    assert w >= 320 and h >= 240


def test_window_manager_modes_toggle_and_resize() -> None:
    """WindowManager switches modes, handles resize events, and toggles F11 cleanly."""
    wm = WindowManager(mode="windowed", windowed_size=(1024, 600))
    surf_headless = wm.create(headless=True)
    assert surf_headless.get_size() == (1024, 600)

    surf_toggled = wm.toggle_fullscreen(headless=True)
    assert wm.mode == "borderless"
    assert surf_toggled.get_width() >= 320

    surf_resized = wm.handle_resize(800, 500, headless=True)
    assert surf_resized.get_size() == (800, 500)

    # Exercise dummy SDL video display lifecycle
    try:
        surf_real = wm.create(headless=False)
        assert surf_real.get_width() >= 320
        surf_fs = wm.toggle_fullscreen(headless=False)
        assert surf_fs.get_width() >= 320
        wm.handle_resize(960, 540, headless=False)
        assert wm.size[0] >= 320
    finally:
        if pygame.display.get_init():
            pygame.display.quit()


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
