"""Unit tests for HookManager and CaptureBuffer.

Single responsibility: test non-invasive PyTorch forward hook attachment, handle cleanup,
call counting, and unbatched CPU tensor capture.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.hooks import HookManager


class DummySubmodule(nn.Module):
    """Submodule for testing hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyModel(nn.Module):
    """Simple model with multiple submodules and repeated calls."""

    def __init__(self) -> None:
        super().__init__()
        self.sub1 = DummySubmodule()
        self.sub2 = DummySubmodule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call sub1 twice to verify call counting
        h1 = self.sub1(x)
        h2 = self.sub1(h1)
        out = self.sub2(h2)
        return out


def test_capture_buffer_operations() -> None:
    """Test CaptureBuffer storing, retrieval, unbatching, and clear."""
    buf = CaptureBuffer()

    # Store 3-D batched tensor (batch_size=1, seq_len=3, dim=4)
    t = torch.randn(1, 3, 4)
    buf.store("sub1", 0, "in", t)

    assert buf.has("sub1", 0, "in")
    assert not buf.has("sub1", 0, "out")

    retrieved = buf.get_required("sub1", 0, "in")
    # Batch dimension 0 should be stripped
    assert retrieved.shape == (3, 4)
    assert retrieved.dtype == torch.float32
    assert retrieved.device == torch.device("cpu")

    # Storing tuple of tensors extracts the first tensor
    buf.store("sub2", 0, "out", (torch.randn(1, 4), "extra"))
    assert buf.get("sub2", 0, "out").shape == (4,)

    # Clear
    buf.clear()
    assert len(buf) == 0
    with pytest.raises(KeyError):
        buf.get_required("sub1", 0, "in")


def test_hook_manager_normal_lifecycle() -> None:
    """Test HookManager registers hooks, counts calls, and cleans handles on normal exit."""
    model = DummyModel()
    model.train()  # Start in training mode
    buf = CaptureBuffer()

    pre_paths = ["sub1", "sub2"]
    post_paths = ["sub1", "sub2"]

    mgr = HookManager(model, buf, pre_hook_paths=pre_paths, post_hook_paths=post_paths)

    with mgr:
        # Model should be in eval mode inside context
        assert not model.training
        # Handles attached
        assert len(mgr._handles) == 4

        x = torch.randn(1, 2, 4)
        out = model(x)
        assert out.shape == (1, 2, 4)

        # sub1 was called twice
        assert mgr.call_counts["sub1"] == 2
        assert mgr.call_counts["sub2"] == 1

        # Check captured tensors
        assert buf.has("sub1", 0, "in")
        assert buf.has("sub1", 1, "in")
        assert buf.has("sub1", 0, "out")
        assert buf.has("sub1", 1, "out")
        assert buf.has("sub2", 0, "in")
        assert buf.has("sub2", 0, "out")

    # After exit: handles removed and original training mode restored
    assert model.training
    assert len(mgr._handles) == 0
    for mod in model.modules():
        assert len(mod._forward_hooks) == 0
        assert len(mod._forward_pre_hooks) == 0


def test_hook_manager_exception_cleanup() -> None:
    """Test that HookManager removes all hooks even when an exception is raised inside context."""
    model = DummyModel()
    buf = CaptureBuffer()

    mgr = HookManager(model, buf, pre_hook_paths=["sub1"], post_hook_paths=["sub1"])

    with pytest.raises(RuntimeError, match="Simulated crash"), mgr:
        assert len(mgr._handles) > 0
        raise RuntimeError("Simulated crash")

    # Handles must be cleanly removed
    assert len(mgr._handles) == 0
    for mod in model.modules():
        assert len(mod._forward_hooks) == 0
        assert len(mod._forward_pre_hooks) == 0
