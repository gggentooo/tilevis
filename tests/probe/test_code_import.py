"""Unit tests for probe dynamic code importing and sys.path safety."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tilevis.core.errors import ModelLoadError
from tilevis.probe.loading.code_import import clear_module_cache, import_model_module


def test_import_model_module_preserves_sys_path(tmp_path: Path) -> None:
    original_path = list(sys.path)

    # Create dummy user module
    mod_file = tmp_path / "custom_model.py"
    mod_file.write_text("CUSTOM_VAR = 42\n", encoding="utf-8")

    mod = import_model_module(mod_file)
    assert mod.CUSTOM_VAR == 42
    # Verify sys.path is completely restored
    assert sys.path == original_path


def test_import_model_module_preserves_sys_path_on_error(tmp_path: Path) -> None:
    original_path = list(sys.path)

    # Create invalid Python module that raises during execution
    bad_file = tmp_path / "syntax_err.py"
    bad_file.write_text("raise RuntimeError('Simulated execution failure')\n", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="Simulated execution failure"):
        import_model_module(bad_file, reload=True)

    # Verify sys.path is still completely restored
    assert sys.path == original_path


def test_missing_file_raises_model_load_error(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.py"
    with pytest.raises(ModelLoadError, match="Model code file not found"):
        import_model_module(missing)


def test_two_different_files_with_same_name_do_not_clash(tmp_path: Path) -> None:
    clear_module_cache()
    dir1 = tmp_path / "team_a"
    dir2 = tmp_path / "team_b"
    dir1.mkdir()
    dir2.mkdir()

    file1 = dir1 / "model.py"
    file2 = dir2 / "model.py"

    file1.write_text("VERSION = 'team_a_v1'\n", encoding="utf-8")
    file2.write_text("VERSION = 'team_b_v2'\n", encoding="utf-8")

    mod1 = import_model_module(file1)
    mod2 = import_model_module(file2)

    assert mod1.__name__ != mod2.__name__
    assert mod1.VERSION == "team_a_v1"
    assert mod2.VERSION == "team_b_v2"
