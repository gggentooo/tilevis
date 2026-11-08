"""Execution capture buffer storing unbatched PyTorch activation tensors.

Single responsibility: retain detached float32 CPU tensors keyed by (module_path, call_idx, role).
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _extract_first_tensor(obj: Any) -> torch.Tensor | None:
    """Extract the first torch.Tensor from a tensor, tuple, or list."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            extracted = _extract_first_tensor(item)
            if extracted is not None:
                return extracted
    return None


def _to_unbatched_cpu_float32(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to detached, unbatched float32 on CPU.

    Removes batch dimension 0 if ndim >= 2 and shape[0] == 1.
    """
    detached = t.detach().to(device="cpu", dtype=torch.float32)
    if detached.ndim >= 2 and detached.shape[0] == 1:
        return detached[0].clone()
    return detached.clone()


class CaptureBuffer:
    """In-memory buffer capturing tensors from a single forward pass."""

    def __init__(self) -> None:
        self._storage: dict[tuple[str, int, str], torch.Tensor] = {}

    def store(
        self,
        module_path: str,
        call_idx: int,
        role: str,
        tensor_or_args: Any,
    ) -> None:
        """Store a tensor in the capture buffer.

        Args:
            module_path: Dot-separated submodule path within model.
            call_idx: Invocation index (0-based) for modules called multiple times.
            role: Semantic capture position ('in' or 'out').
            tensor_or_args: Raw tensor or container from pre/post hook.
        """
        t = _extract_first_tensor(tensor_or_args)
        if t is None:
            return
        processed = _to_unbatched_cpu_float32(t)
        self._storage[(module_path, call_idx, role)] = processed

    def get(
        self,
        module_path: str,
        call_idx: int = 0,
        role: str = "out",
    ) -> torch.Tensor | None:
        """Retrieve a captured tensor, or None if not captured."""
        return self._storage.get((module_path, call_idx, role))

    def get_required(
        self,
        module_path: str,
        call_idx: int = 0,
        role: str = "out",
    ) -> torch.Tensor:
        """Retrieve a captured tensor, raising KeyError if not found."""
        key = (module_path, call_idx, role)
        if key not in self._storage:
            raise KeyError(f"Captured tensor for {key} not found in buffer.")
        return self._storage[key]

    def has(
        self,
        module_path: str,
        call_idx: int = 0,
        role: str = "out",
    ) -> bool:
        """Check whether a specific tensor key is present in the buffer."""
        return (module_path, call_idx, role) in self._storage

    def clear(self) -> None:
        """Reset and empty all captured tensors."""
        self._storage.clear()

    def keys(self) -> list[tuple[str, int, str]]:
        """Return list of captured keys."""
        return list(self._storage.keys())

    def __len__(self) -> int:
        return len(self._storage)

    def __contains__(self, key: tuple[str, int, str]) -> bool:
        return key in self._storage

    def __getitem__(self, key: tuple[str, int, str]) -> torch.Tensor:
        return self._storage[key]
