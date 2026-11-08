"""Unit tests for Trace dataclass immutability, helpers, and orientation conventions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tests.fixtures.fake_traces import make_fake_trace
from tilevis.core.kinds import StepKind, TensorRole, parse_kind
from tilevis.core.trace import AxisInfo, TensorSnapshot, TensorStats


def test_axis_info_immutability() -> None:
    indices = np.array([0, 2, 4], dtype=np.int64)
    axis = AxisInfo(name="token", full_size=10, indices=indices, labels=("a", "b", "c"))

    # Field reassignment must raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        axis.name = "new_name"  # type: ignore[misc]

    # Writing to the underlying array must raise ValueError (read-only array)
    with pytest.raises(ValueError, match="read-only"):
        axis.indices[0] = 99


def test_tensor_snapshot_immutability() -> None:
    data = np.ones((2, 3), dtype=np.float32)
    axis0 = AxisInfo("token", 2, np.array([0, 1], dtype=np.int64), ("a", "b"))
    axis1 = AxisInfo("model_dim", 3, np.array([0, 1, 2], dtype=np.int64))
    stats = TensorStats(1.0, 1.0, 1.0, 0.0, 1.0, 1.0)

    snap = TensorSnapshot(
        name="test_snap",
        role=TensorRole.OUTPUT,
        data=data,
        axes=(axis0, axis1),
        full_shape=(2, 3),
        reduction="none",
        stats=stats,
    )

    with pytest.raises(FrozenInstanceError):
        snap.name = "modified"  # type: ignore[misc]

    with pytest.raises(ValueError, match="read-only"):
        snap.data[0, 0] = 42.0


def test_orientation_convention_matrix_multiplication() -> None:
    """Validate that (token x in) @ (in x out) = (token x out) holds cleanly."""
    t_count = 3
    in_features = 4
    out_features = 5

    input_row = np.arange(t_count * in_features, dtype=np.float32).reshape(t_count, in_features)
    weight = np.ones((in_features, out_features), dtype=np.float32)

    # Orientation convention: output = input @ weight
    output = input_row @ weight
    assert output.shape == (t_count, out_features)


def test_trace_convenience_methods() -> None:
    trace = make_fake_trace(n_layers=2, emb_dim=8, n_tokens=3, n_passes=2, seed=42)

    # step_by_id
    step_0 = trace.step_by_id("p0.prompt")
    assert step_0.id == "p0.prompt"
    assert step_0.kind == StepKind.PROMPT

    with pytest.raises(KeyError, match="Step with id 'nonexistent' not found in trace"):
        trace.step_by_id("nonexistent")

    # PassRecord.steps_for_block
    p0 = trace.passes[0]
    b0_steps = p0.steps_for_block(0)
    assert len(b0_steps) > 0
    assert all(s.block_index == 0 for s in b0_steps)

    global_steps = p0.steps_for_block(None)
    assert any(s.kind == StepKind.PROMPT for s in global_steps)

    # Step.by_role
    embed_step = trace.step_by_id("p0.embed")
    tok_vecs = embed_step.by_role(TensorRole.TOKEN_VECTORS)
    assert len(tok_vecs) == 1
    assert tok_vecs[0].role == TensorRole.TOKEN_VECTORS.value


def test_parse_kind_helper() -> None:
    assert parse_kind("prompt") == StepKind.PROMPT
    assert parse_kind("PROMPT") == StepKind.PROMPT
    assert parse_kind("next_token") == StepKind.NEXT_TOKEN
    assert parse_kind("FUTURE_UNSEEN_OP") == "FUTURE_UNSEEN_OP"
