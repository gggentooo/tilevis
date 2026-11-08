"""Comprehensive tests for Trace validation invariants."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.fixtures.fake_traces import make_fake_trace, make_snapshot
from tilevis.core.errors import TraceFormatError
from tilevis.core.kinds import Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import AxisInfo, ModelSpecInfo, PassRecord, Step, Trace
from tilevis.core.validation import validate_trace


def test_valid_trace_passes_validation() -> None:
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=2)
    issues = validate_trace(trace, strict=True)
    assert issues == []


def test_invalid_schema_version() -> None:
    trace = make_fake_trace()
    bad_trace = dataclasses.replace(trace, schema_version=0)
    issues = validate_trace(bad_trace)
    assert any("schema_version" in i for i in issues)

    with pytest.raises(TraceFormatError, match="schema_version"):
        validate_trace(bad_trace, strict=True)


def test_empty_metadata_strings() -> None:
    trace = make_fake_trace()
    bad_trace = dataclasses.replace(trace, tilevis_version="  ", created_at="")
    issues = validate_trace(bad_trace)
    assert any("tilevis_version" in i for i in issues)
    assert any("created_at" in i for i in issues)


def test_invalid_model_parameters() -> None:
    trace = make_fake_trace()
    bad_model = dataclasses.replace(trace.model, n_layers=0, emb_dim=-4, vocab_size=0, n_heads=-1)
    bad_trace = dataclasses.replace(trace, model=bad_model)
    issues = validate_trace(bad_trace)
    assert any("n_layers must be > 0" in i for i in issues)
    assert any("emb_dim must be > 0" in i for i in issues)
    assert any("vocab_size must be > 0" in i for i in issues)
    assert any("n_heads must be > 0" in i for i in issues)


def test_duplicate_step_id() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]
    duplicate_step = dataclasses.replace(p0.steps[0], id=p0.steps[1].id)
    new_steps = (duplicate_step, *p0.steps[1:])
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("Duplicate step id" in i for i in issues)


def test_empty_step_fields() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]
    bad_step = dataclasses.replace(p0.steps[0], kind="", semantic="  ", stage="")
    new_steps = (bad_step, *p0.steps[1:])
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("kind must be a non-empty string" in i for i in issues)
    assert any("semantic must be a non-empty string" in i for i in issues)
    assert any("stage must be a non-empty string" in i for i in issues)


def test_block_index_out_of_bounds() -> None:
    trace = make_fake_trace(n_layers=2, n_passes=1)
    p0 = trace.passes[0]
    bad_step = dataclasses.replace(p0.steps[0], block_index=5)
    new_steps = (bad_step, *p0.steps[1:])
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("block_index 5 outside range [0, 2)" in i for i in issues)


def test_invalid_detail_level() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]
    bad_step = dataclasses.replace(p0.steps[0], detail_level=99)
    new_steps = (bad_step, *p0.steps[1:])
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("detail_level 99 must be 1" in i for i in issues)


def test_nan_or_infinite_tensor_data() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]
    # find step with tensor
    step_with_tensor = next(s for s in p0.steps if s.tensors)
    bad_data = np.copy(step_with_tensor.tensors[0].data)
    bad_data[0, 0] = np.nan
    bad_snapshot = dataclasses.replace(step_with_tensor.tensors[0], data=bad_data)
    bad_step = dataclasses.replace(step_with_tensor, tensors=(bad_snapshot,))

    new_steps = tuple(bad_step if s.id == step_with_tensor.id else s for s in p0.steps)
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("contains NaN or infinite values" in i for i in issues)


def test_tensor_axes_mismatch() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]
    step_with_tensor = next(s for s in p0.steps if s.tensors)
    # Drop one axis
    bad_snapshot = dataclasses.replace(
        step_with_tensor.tensors[0], axes=step_with_tensor.tensors[0].axes[:1]
    )
    bad_step = dataclasses.replace(step_with_tensor, tensors=(bad_snapshot,))

    new_steps = tuple(bad_step if s.id == step_with_tensor.id else s for s in p0.steps)
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("!= axes count" in i for i in issues)


def test_axis_indices_not_strictly_increasing_or_out_of_bounds() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]
    step_with_tensor = next(s for s in p0.steps if s.tensors)

    # Non-increasing indices
    bad_axis = AxisInfo("model_dim", 100, np.array([5, 5, 10], dtype=np.int64))
    bad_snapshot = make_snapshot(
        "bad", TensorRole.OUTPUT, np.ones((3,), dtype=np.float32), ("model_dim",)
    )
    object.__setattr__(bad_snapshot, "axes", (bad_axis,))

    bad_step = dataclasses.replace(step_with_tensor, tensors=(bad_snapshot,))
    new_steps = tuple(bad_step if s.id == step_with_tensor.id else s for s in p0.steps)
    bad_pass = dataclasses.replace(p0, steps=new_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("not strictly increasing" in i for i in issues)


def test_token_axis_missing_or_mismatched_labels() -> None:
    # Token axis with missing labels
    bad_axis = AxisInfo("token", 10, np.array([0, 1], dtype=np.int64), labels=None)
    data = np.ones((2,), dtype=np.float32)
    snap = make_snapshot("test", TensorRole.OUTPUT, data, ("token",))
    object.__setattr__(snap, "axes", (bad_axis,))

    step = Step("s1", StepKind.PROMPT, Semantic.PROMPT, Stage.INPUT, None, 1, None, tensors=(snap,))
    pass_rec = PassRecord(0, (1, 2), ("a", "b"), "a b", steps=(step,))
    model = ModelSpecInfo("m", "gpt_scratch", 1, 10, 2, 100, 100, 1000)
    trace = Trace(1, "0.2.0", "2026-01-01T00:00:00Z", model, "prompt", {}, (pass_rec,))

    issues = validate_trace(trace)
    assert any("token axis must have labels" in i for i in issues)


def test_next_token_step_not_last_or_multiple() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]

    # Place NEXT_TOKEN step at index 0
    next_step = next(s for s in p0.steps if s.kind == StepKind.NEXT_TOKEN)
    bad_steps = (next_step, *[s for s in p0.steps if s.id != next_step.id])
    bad_pass = dataclasses.replace(p0, steps=bad_steps)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("NEXT_TOKEN step must be the pass's last step" in i for i in issues)


def test_prediction_probabilities_not_descending_or_out_of_bounds() -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    p0 = trace.passes[0]

    # Non-descending probs
    bad_pred = dataclasses.replace(p0.prediction, candidate_probs=(0.1, 0.9))  # type: ignore[arg-type]
    bad_pass = dataclasses.replace(p0, prediction=bad_pred)
    bad_trace = dataclasses.replace(trace, passes=(bad_pass,))

    issues = validate_trace(bad_trace)
    assert any("candidate_probs not descending" in i for i in issues)
