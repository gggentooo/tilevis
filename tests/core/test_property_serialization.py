"""Property-based tests for Trace serialization and reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.fake_traces import make_fake_trace
from tilevis.core.serialization import load_trace, save_trace


@settings(max_examples=15, deadline=None)
@given(
    n_layers=st.integers(min_value=1, max_value=3),
    emb_dim=st.sampled_from([8, 12, 16]),
    n_tokens=st.integers(min_value=2, max_value=4),
    n_passes=st.integers(min_value=1, max_value=2),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_hypothesis_trace_roundtrip(
    tmp_path_factory: pytest.TempPathFactory,
    n_layers: int,
    emb_dim: int,
    n_tokens: int,
    n_passes: int,
    seed: int,
) -> None:
    """Hypothesis property: random valid fake traces round-trip losslessly."""
    tmp_dir = tmp_path_factory.mktemp("hypo")
    trace_file = tmp_dir / f"trace_{seed}.trace.npz"

    original = make_fake_trace(
        n_layers=n_layers,
        emb_dim=emb_dim,
        n_tokens=n_tokens,
        n_passes=n_passes,
        seed=seed,
    )

    save_trace(original, trace_file)
    loaded = load_trace(trace_file)

    assert loaded.schema_version == original.schema_version
    assert loaded.model.n_layers == original.model.n_layers
    assert loaded.model.emb_dim == original.model.emb_dim
    assert len(loaded.passes) == len(original.passes)

    for p_orig, p_load in zip(original.passes, loaded.passes, strict=True):
        assert p_orig.input_ids == p_load.input_ids
        for s_orig, s_load in zip(p_orig.steps, p_load.steps, strict=True):
            assert s_orig.id == s_load.id
            for t_orig, t_load in zip(s_orig.tensors, s_load.tensors, strict=True):
                assert np.array_equal(t_orig.data, t_load.data)
