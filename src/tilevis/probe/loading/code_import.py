"""Dynamic user code importing by file path.

Single responsibility: import user model code from file paths safely without sys.path leakage.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from tilevis.core.errors import ModelLoadError

# Suppress matplotlib font cache warning on headless environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("MPLBACKEND", "Agg")

_MODULE_CACHE: dict[Path, ModuleType] = {}


def import_model_module(path: Path | str, reload: bool = False) -> ModuleType:
    """Import a Python module from a filesystem path safely.

    Uses a unique module name (tilevis_user_<sha1>) and temporarily prepends the module's
    parent directory to sys.path so sibling imports succeed. Restores sys.path in a finally
    block to avoid sys.path pollution.

    Args:
        path: Path to user Python file.
        reload: If True, bypass cache and re-import module.

    Returns:
        Loaded ModuleType.

    Raises:
        ModelLoadError: If file is missing or import execution fails.
    """
    resolved_path = Path(path).resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise ModelLoadError(f"Model code file not found: {resolved_path}")

    if not reload and resolved_path in _MODULE_CACHE:
        return _MODULE_CACHE[resolved_path]

    module_hash = hashlib.sha1(str(resolved_path).encode("utf-8")).hexdigest()[:8]
    module_name = f"tilevis_user_{module_hash}"

    original_sys_path = list(sys.path)
    parent_dir = str(resolved_path.parent)

    try:
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        elif sys.path[0] != parent_dir:
            sys.path.remove(parent_dir)
            sys.path.insert(0, parent_dir)

        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
        if spec is None or spec.loader is None:
            raise ModelLoadError(f"Could not load module specification from {resolved_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _MODULE_CACHE[resolved_path] = module
        return module

    except Exception as exc:
        if module_name in sys.modules:
            del sys.modules[module_name]
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"Failed to import model code from {resolved_path}: {exc}") from exc

    finally:
        sys.path[:] = original_sys_path


def clear_module_cache() -> None:
    """Clear internal module cache (primarily for tests)."""
    _MODULE_CACHE.clear()
