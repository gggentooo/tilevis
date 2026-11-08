"""Forward hook context manager for non-invasive PyTorch execution capture.

Single responsibility: register and cleanly remove PyTorch module hooks with execution counting.
Allowed dependencies: torch, tilevis.core, stdlib.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from tilevis.probe.capture import CaptureBuffer

logger = logging.getLogger(__name__)


class HookManager:
    """Context manager attaching non-invasive forward hooks to PyTorch modules.

    Guarantees handle cleanup on exit even when exceptions occur, and restores the
    original model training state.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer: CaptureBuffer,
        pre_hook_paths: Sequence[str] = (),
        post_hook_paths: Sequence[str] = (),
    ) -> None:
        """Initialize HookManager.

        Args:
            model: Target root PyTorch model.
            buffer: CaptureBuffer to receive captured tensors.
            pre_hook_paths: Module paths to attach forward pre-hooks to (capturing input).
            post_hook_paths: Module paths to attach forward hooks to (capturing output).
        """
        self.model = model
        self.buffer = buffer
        self.pre_hook_paths = tuple(pre_hook_paths)
        self.post_hook_paths = tuple(post_hook_paths)

        self._handles: list[RemovableHandle] = []
        self._pre_counts: dict[str, int] = defaultdict(int)
        self._post_counts: dict[str, int] = defaultdict(int)
        self._original_training: bool = model.training

    @property
    def call_counts(self) -> dict[str, int]:
        """Return combined call counts per module path."""
        all_paths = set(self._pre_counts.keys()) | set(self._post_counts.keys())
        return {p: max(self._pre_counts.get(p, 0), self._post_counts.get(p, 0)) for p in all_paths}

    def reset_counts(self) -> None:
        """Reset per-module invocation counters."""
        self._pre_counts.clear()
        self._post_counts.clear()

    def _resolve_submodule(self, path: str) -> nn.Module | None:
        """Resolve dot-separated module path on root model."""
        if not path:
            return self.model
        try:
            return self.model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            logger.warning("Could not resolve submodule %r: %s", path, exc)
            return None

    def __enter__(self) -> HookManager:
        """Attach hooks and put model into eval mode."""
        self._original_training = self.model.training
        self.model.eval()
        self.reset_counts()

        # Attach pre-hooks
        for path in self.pre_hook_paths:
            submod = self._resolve_submodule(path)
            if submod is None:
                continue

            def make_pre_hook(mod_path: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
                def _pre_hook(module: nn.Module, args: tuple[Any, ...]) -> None:
                    idx = self._pre_counts[mod_path]
                    self._pre_counts[mod_path] += 1
                    self.buffer.store(mod_path, idx, "in", args)

                return _pre_hook

            handle = submod.register_forward_pre_hook(make_pre_hook(path))
            self._handles.append(handle)

        # Attach post-hooks
        for path in self.post_hook_paths:
            submod = self._resolve_submodule(path)
            if submod is None:
                continue

            def make_post_hook(mod_path: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
                def _post_hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
                    idx = self._post_counts[mod_path]
                    self._post_counts[mod_path] += 1
                    self.buffer.store(mod_path, idx, "out", output)

                return _post_hook

            handle = submod.register_forward_hook(make_post_hook(path))
            self._handles.append(handle)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Guaranteed removal of all hooks and restoration of training state."""
        try:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
        finally:
            self.model.train(self._original_training)
