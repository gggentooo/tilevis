"""Platform-specific utilities (Windows DPI awareness, display queries).

Single responsibility: configure High-DPI scaling and query monitor desktop geometries.
Allowed dependencies: pygame, stdlib.
"""

from __future__ import annotations

import ctypes
import logging
import sys

import pygame

logger = logging.getLogger("tilevis.render.platform")


def enable_high_dpi() -> bool:
    """Enable Per-Monitor High-DPI awareness on Windows before pygame display init.

    On non-Windows platforms this is a safe no-op returning False.
    """
    if sys.platform != "win32":
        return False

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False

    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        windll.shcore.SetProcessDpiAwareness(2)
        return True
    except Exception:
        try:
            windll.user32.SetProcessDPIAware()
            return True
        except Exception as exc:
            logger.warning("Failed to enable Windows High-DPI awareness: %s", exc)
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
