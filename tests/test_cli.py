"""Unit tests for command-line interface."""

from __future__ import annotations

import pytest

import tilevis
from tilevis.cli import build_parser, main


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"tilevis {tilevis.__version__}" in captured.out


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "LLM Tile Visualizer" in captured.out
    for sub in ["record", "play", "inspect", "check", "demo"]:
        assert sub in captured.out


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_cli_record_error_on_missing_profile(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "record",
            "--profile",
            "nonexistent_profile.toml",
            "--prompt",
            "Hello",
            "-o",
            "out.trace.npz",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_cli_play_missing_file_returns_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["play", "traces/nonexistent_sample.trace.npz", "--headless"])
    assert code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err.lower()


def test_cli_inspect_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["inspect", "--profile", "profiles/tiny_random.toml"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Tilevis Model Inspection" in captured.out


def test_cli_check_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["check"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Preflight Check" in captured.out
    assert "Result: All preflight checks PASSED" in captured.out


def test_cli_demo_implemented() -> None:
    pytest.importorskip("torch")
    code = main(["demo", "--headless"])
    assert code == 0


def test_main_module_execution() -> None:
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-s", "-m", "tilevis", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert f"tilevis {tilevis.__version__}" in res.stdout
