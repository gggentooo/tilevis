"""Weakness and environment compatibility tests for Phase 1 scaffolding."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

from tests.compat_matrix import COMPAT_MATRIX


def test_compat_matrix_integrity() -> None:
    """Verify all 8 standard compatibility configurations are defined and valid."""
    expected = [
        "tiny_ch07",
        "small_bias",
        "heads_8_of_64",
        "head_dim_1",
        "deep_narrow",
        "lora_small",
        "wrapped_ckpt",
        "gpt2_small_shape",
    ]
    for key in expected:
        assert key in COMPAT_MATRIX, f"Missing matrix key {key}"
        cfg = COMPAT_MATRIX[key].to_config_dict()
        assert cfg["emb_dim"] > 0
        assert cfg["n_layers"] > 0
        assert cfg["n_heads"] > 0
        assert cfg["vocab_size"] == 50257


def test_no_pygame_and_pygame_ce_conflict() -> None:
    """Verify pygame and pygame-ce are not both installed simultaneously."""
    installed_distributions = {
        dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()
    }
    has_legacy_pygame = "pygame" in installed_distributions
    has_pygame_ce = "pygame-ce" in installed_distributions

    # They must never be installed concurrently as both claim `import pygame`
    assert not (has_legacy_pygame and has_pygame_ce), (
        "Both 'pygame' and 'pygame-ce' distributions are installed! "
        "Uninstall 'pygame' and keep only 'pygame-ce'."
    )


def test_non_ascii_working_directory_execution(tmp_path: Path) -> None:
    """Weakness probe: verify execution works smoothly in a non-ASCII directory path."""
    korean_dir = tmp_path / "학생_모델_디렉토리"
    korean_dir.mkdir(parents=True, exist_ok=True)

    # Run `python -m tilevis --version` with korean_dir as working directory
    result = subprocess.run(
        [sys.executable, "-s", "-m", "tilevis", "--version"],
        cwd=str(korean_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "tilevis" in result.stdout
