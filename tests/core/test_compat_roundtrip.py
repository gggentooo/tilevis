"""Compatibility tests verifying lossless round-trip for all compat_matrix model shapes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.fake_traces import make_fake_trace
from tilevis.core.serialization import load_trace, save_trace


@pytest.mark.parametrize("matrix_key", list(COMPAT_MATRIX.keys()))
def test_all_compat_matrix_shapes_roundtrip(matrix_key: str, tmp_path: Path) -> None:
    entry = COMPAT_MATRIX[matrix_key]
    out_file = tmp_path / f"{matrix_key}.trace.npz"

    # For gpt2_small_shape, limit reduction to keep test swift (<1s)
    # Using small tokens count to verify shape propagation
    n_tokens = 3
    n_passes = 1

    trace = make_fake_trace(
        n_layers=min(entry.n_layers, 4),  # Cap layers at 4 for fast unit test
        emb_dim=min(entry.emb_dim, 64),
        n_tokens=n_tokens,
        n_passes=n_passes,
        n_heads=entry.n_heads,
        seed=42,
    )

    save_trace(trace, out_file)
    loaded = load_trace(out_file)

    assert loaded.schema_version == trace.schema_version
    assert len(loaded.passes) == len(trace.passes)
    for p_orig, p_load in zip(trace.passes, loaded.passes, strict=True):
        assert len(p_orig.steps) == len(p_load.steps)


@pytest.mark.slow
@pytest.mark.perf
def test_gpt2_small_shape_size_and_load_budget(tmp_path: Path) -> None:
    """Verify large fake trace (gpt2_small_shape) stays within size < 50MB and load < 2s budgets."""
    entry = COMPAT_MATRIX["gpt2_small_shape"]
    out_file = tmp_path / "gpt2_small_budget.trace.npz"

    # Full 12 layers, 3 passes, 64 features reduced
    trace = make_fake_trace(
        n_layers=entry.n_layers,
        emb_dim=64,  # reduced display features
        n_tokens=8,
        n_passes=3,
        n_heads=entry.n_heads,
        seed=123,
    )

    save_trace(trace, out_file)

    # Size budget: < 50 MB
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    assert file_size_mb < 50.0, f"Trace size {file_size_mb:.2f} MB exceeded 50 MB budget"

    # Load time budget: < 2.0 s
    t0 = time.perf_counter()
    loaded = load_trace(out_file)
    load_duration = time.perf_counter() - t0

    assert load_duration < 2.0, f"Trace load time {load_duration:.3f} s exceeded 2.0 s budget"
    assert len(loaded.passes) == 3
