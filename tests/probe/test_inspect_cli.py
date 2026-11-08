"""Unit tests for tilevis inspect CLI command."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import save_matrix_checkpoint

from tilevis.cli import main
from tilevis.probe.inspect import inspect_model


def test_inspect_model_with_tiny_random_profile() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    profile_path = root / "profiles" / "tiny_random.toml"

    stream = io.StringIO()
    code = inspect_model(profile_path=profile_path, stream=stream)  # type: ignore[arg-type]
    assert code == 0

    output = stream.getvalue()
    assert "Tilevis Model Inspection: Tiny Random GPT" in output
    assert "Adapter:          gpt_scratch" in output
    assert "Transformer Blocks (1 total)" in output
    assert "Total Parameters:" in output


def test_inspect_model_with_raw_checkpoint(tmp_path: Path) -> None:
    entry = COMPAT_MATRIX["small_bias"]
    ckpt_path, _ = save_matrix_checkpoint(entry, tmp_path, vocab_size=200)

    stream = io.StringIO()
    code = inspect_model(checkpoint_path=ckpt_path, stream=stream)  # type: ignore[arg-type]
    assert code == 0

    output = stream.getvalue()
    assert "Adapter:          gpt_scratch" in output
    assert "small_bias" in output
    assert "Transformer Blocks (3 total)" in output


def test_cli_inspect_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parent.parent.parent
    profile_path = root / "profiles" / "tiny_random.toml"

    code = main(["inspect", "--profile", str(profile_path)])
    assert code == 0

    captured = capsys.readouterr()
    assert "Tilevis Model Inspection" in captured.out
