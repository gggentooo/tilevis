"""Checkpoint loading, unwrapping, and prefix normalization.

Single responsibility: load PyTorch checkpoint files safely, unwrap nested containers,
and strip DataParallel / torch.compile module prefixes.
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tilevis.core.errors import ModelLoadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Loaded and normalized model checkpoint data."""

    state_dict: dict[str, torch.Tensor]
    module: nn.Module | None
    source_path: Path


def strip_tensor_prefixes(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Strip common wrapper prefixes ('module.', '_orig_mod.') from state dict keys."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        new_key = key
        # Handle multiple nested prefixes, e.g. module._orig_mod.
        while True:
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            elif new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod.") :]
            else:
                break
        cleaned[new_key] = val
    return cleaned


def unwrap_checkpoint_container(obj: Any) -> tuple[dict[str, Any], nn.Module | None]:
    """Extract state dict and optional full module from various checkpoint formats.

    Supports:
    - Pure state dict mapping
    - Containers with 'model_state_dict', 'state_dict', or 'model' keys
    - Pickled nn.Module instances
    """
    if isinstance(obj, nn.Module):
        return obj.state_dict(), obj

    if isinstance(obj, dict):
        for candidate_key in ("model_state_dict", "state_dict", "model"):
            if candidate_key in obj and isinstance(obj[candidate_key], (dict, nn.Module)):
                inner = obj[candidate_key]
                if isinstance(inner, nn.Module):
                    return inner.state_dict(), inner
                return inner, None
        return obj, None

    raise ModelLoadError(
        f"Unsupported checkpoint payload of type {type(obj).__name__}; expected dict or nn.Module"
    )


def load_checkpoint(path: Path | str, allow_pickle: bool = False) -> LoadedCheckpoint:
    """Load and normalize a PyTorch checkpoint file.

    Args:
        path: Path to .pth / .pt checkpoint file.
        allow_pickle: If True, allow unpickling arbitrary objects (with security warning).
            Defaults to False (weights_only=True).

    Returns:
        LoadedCheckpoint containing cleaned state_dict and optional full module.

    Raises:
        ModelLoadError: If checkpoint file is missing, corrupt, or contains pickled
            objects when allow_pickle=False.
    """
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise ModelLoadError(f"Checkpoint file not found: {file_path}")

    if allow_pickle:
        logger.warning(
            "Loading checkpoint %s with allow_pickle=True; only load trusted files!",
            file_path.name,
        )

    try:
        raw_obj = torch.load(
            file_path,
            map_location="cpu",
            weights_only=not allow_pickle,
        )
    except Exception as exc:
        exc_str = str(exc)
        if not allow_pickle and any(
            hint in exc_str
            for hint in ("Weights only load failed", "Unsupported class", "Unpickler", "pickle")
        ):
            raise ModelLoadError(
                f"Checkpoint {file_path.name} contains pickled objects or full module code. "
                "If you trust this file, load with --allow-pickle "
                "(or set allow_pickle = true in profile). "
                f"Original error: {exc}"
            ) from exc
        raise ModelLoadError(f"Failed to load checkpoint from {file_path}: {exc}") from exc

    raw_dict, module_or_none = unwrap_checkpoint_container(raw_obj)
    cleaned_state_dict = strip_tensor_prefixes(raw_dict)

    if not cleaned_state_dict:
        raise ModelLoadError(
            f"Checkpoint {file_path.name} contains no valid tensor weights in state dict."
        )

    return LoadedCheckpoint(
        state_dict=cleaned_state_dict,
        module=module_or_none,
        source_path=file_path,
    )
