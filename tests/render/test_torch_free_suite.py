"""Verify the entire render and narration test suites run cleanly with torch blocked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_full_render_and_narration_suite_runs_without_torch() -> None:
    """Execute all tests/render and tests/narration under a hard torch import blocker."""
    script = """
import os, sys, pytest

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

class TorchBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch is intentionally blocked for render/narration verification')
        return None

sys.meta_path.insert(0, TorchBlocker())

rc = pytest.main([
    'tests/render',
    'tests/narration',
    '-k', 'not test_full_render_and_narration_suite_runs_without_torch',
    '-m', 'not manual',
    '-p', 'no:cov',
    '-p', 'no:cacheprovider',
    '-q',
])
assert 'torch' not in sys.modules, 'torch leaked into sys.modules!'
sys.exit(rc)
"""
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYTHONNOUSERSITE"] = "1"
    res = subprocess.run(
        [sys.executable, "-s", "-c", script],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, (
        f"Render/narration suite failed under torch blocker:\n{res.stdout}\n{res.stderr}"
    )
