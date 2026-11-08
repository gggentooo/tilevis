"""Logging configuration for tilevis.

Single responsibility: set up console logging with safe encoding on all platforms.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

import contextlib
import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging for tilevis.

    Args:
        verbose: If True, set log level to DEBUG; otherwise set to INFO.
    """
    # Safe reconfigure on Windows / cp949 consoles to avoid encoding crashes
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(errors="replace")

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("tilevis")
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
