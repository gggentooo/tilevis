"""Unit tests for dimensionality reduction, IndexPlan, and snapshot construction.

Single responsibility: test stride indexing, deterministic coordinate alignment,
tensor statistics, and non-finite sanitization.
"""

from __future__ import annotations

import numpy as np

from tilevis.core.kinds import TensorRole
from tilevis.probe.adapters.base import ModelSpec
from tilevis.probe.reduce import (
    IndexPlan,
    compute_stride_indices,
    compute_tensor_stats,
    create_tensor_snapshot,
)


def test_compute_stride_indices_invariants() -> None:
    """Test stride index computation bounds and ordering invariants."""
    # When full_size <= limit, all indices returned
    idx = compute_stride_indices(16, 32)
    assert np.array_equal(idx, np.arange(16))

    # When full_size > limit, strictly increasing and within bounds
    idx = compute_stride_indices(100, 16)
    assert len(idx) <= 16
    assert idx[0] == 0
    assert idx[-1] == 99
    assert np.all(np.diff(idx) > 0)
    assert np.all((idx >= 0) & (idx < 100))

    # Small limit
    idx1 = compute_stride_indices(50, 1)
    assert len(idx1) == 1
    assert 0 <= idx1[0] < 50


def test_index_plan_alignment() -> None:
    """Test IndexPlan builds deterministic axes and token rolling window."""
    spec = ModelSpec(
        adapter="gpt_scratch",
        token_embedding="tok_emb",
        position_embedding="pos_emb",
        blocks=(),
        final_norm="final_norm",
        lm_head="out_head",
        emb_dim=48,
        vocab_size=1000,
        context_length=64,
        n_heads=4,
    )

    plan = IndexPlan.from_model_spec(spec, max_features=32, max_tokens=10, max_heads=6)

    assert len(plan.model_dim) <= 32
    assert len(plan.ff_dim) <= 32
    assert len(plan.head_dim) <= 32
    assert len(plan.head) <= 6

    # Token indices: when T <= max_tokens
    t_short = plan.get_token_indices(8)
    assert np.array_equal(t_short, np.arange(8))

    # Token indices: when T > max_tokens, keeps the LAST max_tokens
    t_long = plan.get_token_indices(25)
    assert len(t_long) == 10
    assert np.array_equal(t_long, np.arange(15, 25))
    assert np.all(np.diff(t_long) > 0)


def test_compute_tensor_stats() -> None:
    """Test full tensor statistics calculation."""
    # Empty array
    empty = np.array([], dtype=np.float32)
    stats_empty = compute_tensor_stats(empty)
    assert stats_empty.min == 0.0
    assert stats_empty.max == 0.0

    # Populated array
    arr = np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    stats = compute_tensor_stats(arr)
    assert stats.min == -5.0
    assert stats.max == 10.0
    assert stats.mean == 2.5
    assert stats.abs_max == 10.0
    assert stats.abs_p99 <= 10.0


def test_create_tensor_snapshot_sanitization_and_slicing() -> None:
    """Test snapshot creation, label slicing, and NaN/Inf replacement."""
    spec = ModelSpec(
        adapter="gpt_scratch",
        token_embedding="tok_emb",
        position_embedding=None,
        blocks=(),
        final_norm=None,
        lm_head="out_head",
        emb_dim=8,
        vocab_size=100,
        context_length=32,
        n_heads=2,
    )
    plan = IndexPlan.from_model_spec(spec, max_features=4, max_tokens=3)

    # 2D array with NaN and Inf
    arr = np.array(
        [
            [1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, float("inf"), 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        ],
        dtype=np.float32,
    )

    tokens = ["token0", "token1", "token2", "token3"]
    snap, had_nf = create_tensor_snapshot(
        name="test_tensor",
        role=TensorRole.OUTPUT,
        tensor=arr,
        axis_names=["token", "model_dim"],
        plan=plan,
        labels_by_axis={0: tokens},
    )

    assert had_nf is True
    # Non-finite values were replaced with 0.0
    assert np.all(np.isfinite(snap.data))
    assert snap.full_shape == (4, 8)
    assert snap.data.ndim == 2
    # Reduced according to plan: max_tokens=3, max_features=4
    assert snap.data.shape[0] <= 3
    assert snap.data.shape[1] <= 4

    # Axes checks
    assert snap.axes[0].name == "token"
    assert snap.axes[0].full_size == 4
    assert snap.axes[0].labels is not None
    assert len(snap.axes[0].labels) == len(snap.axes[0].indices)

    assert snap.axes[1].name == "model_dim"
    assert snap.axes[1].full_size == 8
