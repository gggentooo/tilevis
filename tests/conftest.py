"""Pytest configuration and global fixtures.

Sets up headless dummy audio/video drivers for SDL before pygame can be imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set dummy drivers before any potential import of pygame or SDL
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

# Ensure src/ and project root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
