"""Tests for architectural import boundaries and module isolation rules.

Enforces:
- tilevis.core and tilevis.narration never import torch or pygame
- tilevis.render never imports torch or tilevis.probe
- tilevis.probe never imports pygame or tilevis.render
- tilevis.cli does not eagerly import probe or render at top level
- Subprocess verification that importing render/core/narration does not load torch
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "tilevis"


class ImportViolation(NamedTuple):
    file_path: Path
    line_number: int
    imported_module: str
    rule_violated: str


def _get_top_level_module(module_name: str) -> str:
    """Extract top-level package name from an import string."""
    return module_name.split(".")[0]


def check_ast_imports(package_dir: Path) -> list[ImportViolation]:
    """Scan all Python files in a directory and enforce architectural import rules."""
    violations: list[ImportViolation] = []

    for py_file in package_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError as exc:
            violations.append(ImportViolation(py_file, exc.lineno or 1, "<syntax_error>", str(exc)))
            continue

        rel_parts = py_file.relative_to(package_dir).parts
        layer = rel_parts[0] if len(rel_parts) > 1 else ""

        # Collect top-level imports vs function-level imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Determine names imported
                mod_names: list[str] = []
                if isinstance(node, ast.Import):
                    mod_names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod_names = [node.module]

                for mod in mod_names:
                    top = _get_top_level_module(mod)

                    # Rules for core
                    if layer == "core":
                        if top in {"torch", "pygame", "tiktoken", "matplotlib"}:
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "core must never import torch, pygame, tiktoken, or matplotlib",
                                )
                            )
                        if mod.startswith("tilevis.") and not mod.startswith("tilevis.core"):
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "core must only import within tilevis.core",
                                )
                            )

                    # Rules for narration
                    elif layer == "narration":
                        if top in {"torch", "pygame", "tiktoken", "matplotlib"}:
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "narration must never import torch, pygame, or matplotlib",
                                )
                            )
                        if mod.startswith("tilevis.") and not (
                            mod.startswith("tilevis.core") or mod.startswith("tilevis.narration")
                        ):
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "narration can only import core and narration",
                                )
                            )

                    # Rules for render
                    elif layer == "render":
                        if top in {"torch", "matplotlib"}:
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "render must never import torch or matplotlib",
                                )
                            )
                        if mod.startswith("tilevis.probe"):
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "render must never import tilevis.probe",
                                )
                            )

                    # Rules for probe
                    elif layer == "probe":
                        if top == "pygame":
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "probe must never import pygame",
                                )
                            )
                        if mod.startswith("tilevis.render") or mod.startswith("tilevis.narration"):
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    node.lineno,
                                    mod,
                                    "probe must never import render or narration",
                                )
                            )

        # Special check for cli.py: top-level imports must not include probe or render
        if py_file.name == "cli.py":
            for stmt in tree.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    imported = []
                    if isinstance(stmt, ast.Import):
                        imported = [a.name for a in stmt.names]
                    elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                        imported = [stmt.module]
                    for imp in imported:
                        if imp.startswith("tilevis.probe") or imp.startswith("tilevis.render"):
                            violations.append(
                                ImportViolation(
                                    py_file,
                                    stmt.lineno,
                                    imp,
                                    "cli.py must import probe and render lazily, not at top level",
                                )
                            )

    return violations


def test_codebase_import_boundaries() -> None:
    """Verify that every file in src/tilevis complies with boundary rules."""
    violations = check_ast_imports(SRC_ROOT)
    if violations:
        msg = "\n".join(
            f"  {v.file_path.name}:{v.line_number} imports {v.imported_module} -> {v.rule_violated}"
            for v in violations
        )
        raise AssertionError(f"Architecture import violations detected:\n{msg}")


def test_subprocess_torch_free_playback_import() -> None:
    """Verify in a subprocess that importing render/narration/core/cli does not load torch."""
    code = (
        "import sys, os\n"
        "os.environ['PYTHONNOUSERSITE'] = '1'\n"
        "import tilevis.core\n"
        "import tilevis.narration\n"
        "import tilevis.render\n"
        "import tilevis.cli\n"
        "assert 'torch' not in sys.modules, f'torch was imported: {sys.modules.get(\"torch\")}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-s", "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Subprocess failed with stderr:\n{result.stderr}"


def test_boundary_checker_catches_bad_imports(tmp_path: Path) -> None:
    """Weakness probe: verify check_ast_imports accurately catches illegal imports."""
    bad_core = tmp_path / "core"
    bad_core.mkdir()
    (bad_core / "bad_mod.py").write_text("import torch\n", encoding="utf-8")

    violations = check_ast_imports(tmp_path)
    assert any("core must never import torch" in v.rule_violated for v in violations)

    bad_render = tmp_path / "render"
    bad_render.mkdir()
    (bad_render / "bad_render.py").write_text("import tilevis.probe\n", encoding="utf-8")

    violations2 = check_ast_imports(tmp_path)
    assert any("render must never import tilevis.probe" in v.rule_violated for v in violations2)

    bad_probe = tmp_path / "probe"
    bad_probe.mkdir()
    (bad_probe / "bad_probe.py").write_text("import pygame\n", encoding="utf-8")

    violations3 = check_ast_imports(tmp_path)
    assert any("probe must never import pygame" in v.rule_violated for v in violations3)
