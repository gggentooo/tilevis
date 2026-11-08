"""Unit tests for Trace serialization, deserialization, migrations, and error handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from tests.fixtures.fake_traces import make_fake_trace
from tilevis.core.errors import TraceFormatError
from tilevis.core.serialization import MIGRATIONS, load_trace, save_trace
from tilevis.core.trace import TRACE_SCHEMA_VERSION


def test_save_and_load_roundtrip_lossless(tmp_path: Path) -> None:
    original = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=2, seed=123)
    trace_path = tmp_path / "test.trace.npz"

    save_trace(original, trace_path)
    assert trace_path.exists()

    loaded = load_trace(trace_path)

    assert loaded.schema_version == original.schema_version
    assert loaded.tilevis_version == original.tilevis_version
    assert loaded.model.name == original.model.name
    assert loaded.model.n_layers == original.model.n_layers
    assert loaded.model.emb_dim == original.model.emb_dim
    assert len(loaded.passes) == len(original.passes)

    # Check array fidelity on all tensors
    for p_orig, p_load in zip(original.passes, loaded.passes, strict=True):
        assert p_orig.input_ids == p_load.input_ids
        assert p_orig.input_tokens == p_load.input_tokens
        assert len(p_orig.steps) == len(p_load.steps)
        for s_orig, s_load in zip(p_orig.steps, p_load.steps, strict=True):
            assert s_orig.id == s_load.id
            assert len(s_orig.tensors) == len(s_load.tensors)
            for t_orig, t_load in zip(s_orig.tensors, s_load.tensors, strict=True):
                assert t_orig.name == t_load.name
                assert np.array_equal(t_orig.data, t_load.data)
                for a_orig, a_load in zip(t_orig.axes, t_load.axes, strict=True):
                    assert np.array_equal(a_orig.indices, a_load.indices)
                    assert a_orig.labels == a_load.labels


def test_unicode_and_korean_roundtrip(tmp_path: Path) -> None:
    trace = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=3, n_passes=1)
    korean_tokens = ("시작", "·노력은", "·성공🚀\n")

    p0 = trace.passes[0]
    new_p0 = p0
    object.__setattr__(new_p0, "input_tokens", korean_tokens)

    korean_trace = trace
    object.__setattr__(korean_trace, "prompt", "모든 노력은 앞으로 나아간다 🌟")

    save_path = tmp_path / "한국어_경로" / "테스트.trace.npz"
    save_trace(korean_trace, save_path)
    loaded = load_trace(save_path)

    assert loaded.prompt == "모든 노력은 앞으로 나아간다 🌟"
    assert loaded.passes[0].input_tokens == korean_tokens


def test_corrupt_or_truncated_archive(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.trace.npz"
    bad_file.write_bytes(b"PK\x03\x04truncated_corrupt_content")

    with pytest.raises(TraceFormatError, match="Corrupt or invalid trace archive"):
        load_trace(bad_file)


def test_missing_manifest(tmp_path: Path) -> None:
    bad_file = tmp_path / "no_manifest.trace.npz"
    # save zip without __manifest__
    np.savez(bad_file, a0=np.array([1, 2, 3]))

    with pytest.raises(TraceFormatError, match="Missing __manifest__ in trace archive"):
        load_trace(bad_file)


def test_manifest_referencing_missing_array(tmp_path: Path) -> None:
    bad_file = tmp_path / "missing_array.trace.npz"
    manifest = {
        "schema_version": 1,
        "tilevis_version": "0.2.0",
        "created_at": "2026-01-01T00:00:00Z",
        "model": {
            "name": "m",
            "adapter": "gpt_scratch",
            "n_layers": 1,
            "emb_dim": 8,
            "vocab_size": 100,
            "n_params": 100,
        },
        "prompt": "hi",
        "passes": [
            {
                "index": 0,
                "input_ids": [1],
                "input_tokens": ["hi"],
                "text_before": "hi",
                "steps": [
                    {
                        "id": "s0",
                        "kind": "LINEAR",
                        "semantic": "generic",
                        "stage": "BLOCK",
                        "detail_level": 2,
                        "tensors": [
                            {
                                "name": "t0",
                                "role": "output",
                                "data": "@a99",  # missing array
                                "axes": [],
                                "full_shape": [1],
                                "reduction": "none",
                                "stats": {
                                    "min": 0,
                                    "max": 0,
                                    "mean": 0,
                                    "std": 0,
                                    "abs_max": 0,
                                    "abs_p99": 0,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    manifest_arr = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    np.savez(bad_file, __manifest__=manifest_arr)

    with pytest.raises(TraceFormatError, match="Array key 'a99' not found"):
        load_trace(bad_file)


def test_newer_schema_version_rejected(tmp_path: Path) -> None:
    bad_file = tmp_path / "future_schema.trace.npz"
    manifest = {
        "schema_version": TRACE_SCHEMA_VERSION + 1,
        "tilevis_version": "9.9.9",
        "created_at": "2030-01-01T00:00:00Z",
        "model": {
            "name": "m",
            "adapter": "a",
            "n_layers": 1,
            "emb_dim": 8,
            "vocab_size": 100,
            "n_params": 100,
        },
        "prompt": "hi",
        "passes": [],
    }
    manifest_arr = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    np.savez(bad_file, __manifest__=manifest_arr)

    with pytest.raises(TraceFormatError, match="was produced by a newer version of tilevis"):
        load_trace(bad_file)


def test_schema_migration_path(tmp_path: Path) -> None:
    """Test applying sequential migrations from an older schema version."""
    migrated_file = tmp_path / "v0_schema.trace.npz"

    # Define a temporary mock migration from version 0 to 1
    def migrate_v0_to_v1(d: dict) -> dict:
        d["schema_version"] = 1
        d["tilevis_version"] = "migrated-0.2.0"
        return d

    MIGRATIONS[0] = migrate_v0_to_v1

    try:
        trace = make_fake_trace(n_layers=1, n_passes=1)
        save_trace(trace, migrated_file)

        # Tamper manifest to have schema_version = 0
        data = dict(np.load(migrated_file, allow_pickle=False))
        manifest = json.loads(bytes(data["__manifest__"]).decode("utf-8"))
        manifest["schema_version"] = 0
        data["__manifest__"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(migrated_file, **data)

        loaded = load_trace(migrated_file)
        assert loaded.schema_version == 1
        assert loaded.tilevis_version == "migrated-0.2.0"
    finally:
        MIGRATIONS.pop(0, None)


def test_pickle_rejection(tmp_path: Path) -> None:
    """Verify that a pickle object inside an archive is rejected."""
    bad_file = tmp_path / "pickled.trace.npz"
    manifest = {
        "schema_version": 1,
        "tilevis_version": "0.2.0",
        "created_at": "2026-01-01T00:00:00Z",
        "model": {
            "name": "m",
            "adapter": "a",
            "n_layers": 1,
            "emb_dim": 8,
            "vocab_size": 100,
            "n_params": 100,
        },
        "prompt": "hi",
        "passes": [],
    }
    manifest_arr = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    # Write a pickle object array
    obj_arr = np.array([object()], dtype=object)
    np.savez(bad_file, __manifest__=manifest_arr, a0=obj_arr)

    with pytest.raises(TraceFormatError, match=r"Corrupt or invalid trace archive|pickle"):
        load_trace(bad_file)


def test_atomic_write_leaves_original_intact_on_failure(tmp_path: Path) -> None:
    trace = make_fake_trace(n_layers=1, n_passes=1)
    target_file = tmp_path / "important.trace.npz"

    save_trace(trace, target_file)
    original_bytes = target_file.read_bytes()

    # Simulate an error during np.savez_compressed
    with (
        mock.patch("numpy.savez_compressed", side_effect=OSError("Disk write simulated failure")),
        pytest.raises(TraceFormatError, match="Failed to write trace archive"),
    ):
        save_trace(trace, target_file)

    # Verify original file content is intact
    assert target_file.read_bytes() == original_bytes


def test_unknown_kind_string_loads_fine(tmp_path: Path) -> None:
    """Verify that an unknown kind string found in a newer trace loads without crashing."""
    trace = make_fake_trace(n_layers=1, n_passes=1)
    file_path = tmp_path / "unknown_kind.trace.npz"
    save_trace(trace, file_path)

    # Tamper manifest to have a step with an unknown kind string
    data = dict(np.load(file_path, allow_pickle=False))
    manifest = json.loads(bytes(data["__manifest__"]).decode("utf-8"))
    manifest["passes"][0]["steps"][0]["kind"] = "FUTURE_QUANTUM_ATTENTION"
    data["__manifest__"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(file_path, **data)

    loaded = load_trace(file_path)
    assert loaded.passes[0].steps[0].kind == "FUTURE_QUANTUM_ATTENTION"
