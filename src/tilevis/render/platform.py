"""Platform-specific utilities (Windows DPI awareness, display queries).

Single responsibility: configure High-DPI scaling and query monitor desktop geometries.
Allowed dependencies: pygame, stdlib.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass

import pygame

logger = logging.getLogger("tilevis.render.platform")


@dataclass(frozen=True, slots=True)
class DisplayInfo:
    """Detected desktop resolution and computed smart default windowed geometry."""

    desktop_size: tuple[int, int]
    default_windowed: tuple[int, int]

    @property
    def desktop_w(self) -> int:
        return self.desktop_size[0]

    @property
    def desktop_h(self) -> int:
        return self.desktop_size[1]


def default_windowed_size(desktop_w: int, desktop_h: int) -> tuple[int, int]:
    """Compute smart default windowed resolution fitting desktop with 16:9 aspect ratio.

    Returns exact (1920, 1080) when the desktop exceeds 1080p in both dimensions,
    or a crisp 16:9 fit (~85% of desktop bounds, minimum 800x450) when running on
    1080p or smaller displays.
    """
    dw = max(1, int(desktop_w))
    dh = max(1, int(desktop_h))

    # Desktop strictly exceeds 1080p in both dimensions (e.g. 2560x1440, 3840x2160)
    if dw > 1920 and dh > 1080:
        return (1920, 1080)

    # 16:9 aspect ratio unit: width = 16 * k, height = 9 * k
    target_w = dw * 0.85
    target_h = dh * 0.85
    k = int(min(target_w / 16.0, target_h / 9.0))
    # Minimum 800x450 (k=50)
    k = max(50, k)

    w = 16 * k
    h = 9 * k

    # If minimum (800x450) exceeds desktop bounds, clamp to desktop bounds
    if w > dw or h > dh:
        k_fit = int(min(dw / 16.0, dh / 9.0))
        w = max(16, 16 * k_fit) if k_fit > 0 else dw
        h = max(9, 9 * k_fit) if k_fit > 0 else dh

    return (min(1920, w), min(1080, h))


def enable_high_dpi() -> bool:
    """Enable Per-Monitor High-DPI awareness and SDL environment hints before display init.

    On non-Windows platforms, sets SDL High-DPI hints and returns False.
    """
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")

    if sys.platform != "win32":
        return False

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False

    # 1. Try Windows 10 1703+ DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (-4)
    try:
        user32 = getattr(windll, "user32", None)
        if (
            user32 is not None
            and hasattr(user32, "SetProcessDpiAwarenessContext")
            and user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        ):
            return True
    except Exception as exc:
        logger.debug("SetProcessDpiAwarenessContext(-4) failed: %s", exc)

    # 2. Try Windows 8.1+ PROCESS_PER_MONITOR_DPI_AWARE (2)
    try:
        shcore = getattr(windll, "shcore", None)
        if shcore is not None and hasattr(shcore, "SetProcessDpiAwareness"):
            shcore.SetProcessDpiAwareness(2)
            return True
    except Exception as exc:
        logger.debug("SetProcessDpiAwareness(2) failed: %s", exc)

    # 3. Fallback to Windows Vista+ SetProcessDPIAware()
    try:
        user32 = getattr(windll, "user32", None)
        if user32 is not None and hasattr(user32, "SetProcessDPIAware"):
            user32.SetProcessDPIAware()
            return True
    except Exception as exc:
        logger.warning("Failed to enable Windows High-DPI awareness: %s", exc)
        return False

    return False


def get_desktop_size(display: int = 0) -> tuple[int, int]:
    """Query desktop monitor resolution with a safe fallback when unavailable."""
    try:
        if not pygame.display.get_init():
            pygame.display.init()
        sizes = pygame.display.get_desktop_sizes()
        if sizes:
            idx = display if 0 <= display < len(sizes) else 0
            w, h = sizes[idx]
            if w >= 64 and h >= 64:
                return int(w), int(h)
    except Exception as exc:
        logger.debug("Desktop size query failed (%s); using default 1920x1080", exc)
    return (1920, 1080)


def detect_display_info(display: int = 0) -> DisplayInfo:
    """Detect desktop resolution and compute default windowed size."""
    dw, dh = get_desktop_size(display)
    default_win = default_windowed_size(dw, dh)
    return DisplayInfo(desktop_size=(dw, dh), default_windowed=default_win)
