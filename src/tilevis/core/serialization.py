"""Trace serialization and deserialization (*.trace.npz).

Single responsibility: save and load compressed NumPy/JSON Trace archives with schema migrations.
Allowed dependencies: stdlib and numpy only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile

import numpy as np

from tilevis.core.errors import TraceFormatError
from tilevis.core.trace import (
    TRACE_SCHEMA_VERSION,
    AxisInfo,
    Fidelity,
    ModelSpecInfo,
    PassRecord,
    Prediction,
    Step,
    TensorSnapshot,
    TensorStats,
    Trace,
)
from tilevis.core.validation import validate_trace

# Schema migration transformers: version_from -> transformer_func
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback converting numpy scalar types to native Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_trace(trace: Trace, path: Path | str) -> None:
    """Serialize a Trace object to a compressed *.trace.npz file atomically.

    Args:
        trace: Valid Trace instance to serialize.
        path: Destination file path.

    Raises:
        TraceFormatError: If the trace fails strict validation.
    """
    validate_trace(trace, strict=True)

    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f"{dest_path.name}.tmp.{os.getpid()}.npz")

    collected_arrays: list[np.ndarray] = []

    def _extract_array(arr: np.ndarray) -> str:
        ref = f"@a{len(collected_arrays)}"
        collected_arrays.append(arr)
        return ref

    def _serialize_axis(axis: AxisInfo) -> dict[str, Any]:
        return {
            "name": axis.name,
            "full_size": axis.full_size,
            "indices": _extract_array(axis.indices),
            "labels": list(axis.labels) if axis.labels is not None else None,
        }

    def _serialize_stats(stats: TensorStats) -> dict[str, Any]:
        return {
            "min": float(stats.min),
            "max": float(stats.max),
            "mean": float(stats.mean),
            "std": float(stats.std),
            "abs_max": float(stats.abs_max),
            "abs_p99": float(stats.abs_p99),
        }

    def _serialize_tensor(tensor: TensorSnapshot) -> dict[str, Any]:
        return {
            "name": tensor.name,
            "role": tensor.role,
            "data": _extract_array(tensor.data),
            "axes": [_serialize_axis(a) for a in tensor.axes],
            "full_shape": list(tensor.full_shape),
            "reduction": tensor.reduction,
            "stats": _serialize_stats(tensor.stats),
        }

    def _serialize_step(step: Step) -> dict[str, Any]:
        return {
            "id": step.id,
            "kind": str(step.kind),
            "semantic": str(step.semantic),
            "stage": str(step.stage),
            "block_index": step.block_index,
            "detail_level": int(step.detail_level),
            "module_path": step.module_path,
            "tensors": [_serialize_tensor(t) for t in step.tensors],
            "meta": dict(step.meta),
            "fidelity": {
                "status": step.fidelity.status,
                "max_abs_err": step.fidelity.max_abs_err,
                "note": step.fidelity.note,
            },
        }

    def _serialize_prediction(pred: Prediction | None) -> dict[str, Any] | None:
        if pred is None:
            return None
        return {
            "candidate_ids": list(pred.candidate_ids),
            "candidate_tokens": list(pred.candidate_tokens),
            "candidate_probs": list(pred.candidate_probs),
            "chosen_id": pred.chosen_id,
            "chosen_token": pred.chosen_token,
            "strategy": pred.strategy,
        }

    def _serialize_pass(pass_rec: PassRecord) -> dict[str, Any]:
        return {
            "index": pass_rec.index,
            "input_ids": list(pass_rec.input_ids),
            "input_tokens": list(pass_rec.input_tokens),
            "text_before": pass_rec.text_before,
            "steps": [_serialize_step(s) for s in pass_rec.steps],
            "prediction": _serialize_prediction(pass_rec.prediction),
        }

    manifest: dict[str, Any] = {
        "schema_version": trace.schema_version,
        "tilevis_version": trace.tilevis_version,
        "created_at": trace.created_at,
        "model": {
            "name": trace.model.name,
            "adapter": trace.model.adapter,
            "n_layers": trace.model.n_layers,
            "emb_dim": trace.model.emb_dim,
            "n_heads": trace.model.n_heads,
            "vocab_size": trace.model.vocab_size,
            "context_length": trace.model.context_length,
            "n_params": trace.model.n_params,
            "extras": dict(trace.model.extras),
        },
        "prompt": trace.prompt,
        "settings": dict(trace.settings),
        "passes": [_serialize_pass(p) for p in trace.passes],
    }

    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=None, default=_json_default
    ).encode("utf-8")
    manifest_arr = np.frombuffer(manifest_bytes, dtype=np.uint8)

    save_dict: dict[str, np.ndarray] = {
        "__manifest__": manifest_arr,
    }
    for i, arr in enumerate(collected_arrays):
        save_dict[f"a{i}"] = arr

    try:
        np.savez_compressed(temp_path, **save_dict)  # type: ignore[arg-type]
        # Atomic replace
        temp_path.replace(dest_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise TraceFormatError(f"Failed to write trace archive: {exc}") from exc


def load_trace(path: Path | str) -> Trace:
    """Load, migrate (if needed), reconstruct, and validate a Trace from disk.

    Args:
        path: Path to .trace.npz archive.

    Returns:
        Reconstructed and validated Trace instance.

    Raises:
        TraceFormatError: If the file is missing, corrupt, pickled, or invalid.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise TraceFormatError(f"Trace file not found: {file_path}")

    try:
        npz = np.load(file_path, allow_pickle=False)
    except (BadZipFile, OSError, ValueError) as exc:
        raise TraceFormatError(f"Corrupt or invalid trace archive {file_path.name}: {exc}") from exc

    try:
        if "__manifest__" not in npz.files:
            raise TraceFormatError("Missing __manifest__ in trace archive")

        # Verify all archive entries are valid arrays without pickled objects
        for key in npz.files:
            try:
                _ = npz[key]
            except ValueError as exc:
                raise TraceFormatError(
                    f"Corrupt or invalid trace archive (pickled data not allowed): {exc}"
                ) from exc

        manifest_raw = bytes(npz["__manifest__"])
        manifest = json.loads(manifest_raw.decode("utf-8"))

        schema_v = manifest.get("schema_version", 1)
        if schema_v > TRACE_SCHEMA_VERSION:
            raise TraceFormatError(
                f"Trace schema version {schema_v} was produced by a newer version of tilevis "
                f"(current supported: {TRACE_SCHEMA_VERSION}). Please update tilevis."
            )

        # Apply sequential migrations if trace is from an older schema version
        while schema_v < TRACE_SCHEMA_VERSION:
            if schema_v not in MIGRATIONS:
                raise TraceFormatError(
                    f"No migration path available from schema {schema_v} to {TRACE_SCHEMA_VERSION}"
                )
            manifest = MIGRATIONS[schema_v](manifest)
            schema_v = manifest.get("schema_version", schema_v + 1)

        def _get_array(ref: str) -> np.ndarray:
            if not isinstance(ref, str) or not ref.startswith("@a"):
                raise TraceFormatError(f"Invalid array reference: {ref!r}")
            key = ref[1:]
            if key not in npz.files:
                raise TraceFormatError(f"Array key {key!r} not found in trace archive")
            return cast(np.ndarray, npz[key])

        def _reconstruct_axis(d: dict[str, Any]) -> AxisInfo:
            arr = _get_array(d["indices"])
            labels = tuple(d["labels"]) if d.get("labels") is not None else None
            return AxisInfo(
                name=d["name"],
                full_size=d["full_size"],
                indices=arr,
                labels=labels,
            )

        def _reconstruct_stats(d: dict[str, Any]) -> TensorStats:
            return TensorStats(
                min=d["min"],
                max=d["max"],
                mean=d["mean"],
                std=d["std"],
                abs_max=d["abs_max"],
                abs_p99=d["abs_p99"],
            )

        def _reconstruct_tensor(d: dict[str, Any]) -> TensorSnapshot:
            arr = _get_array(d["data"])
            axes = tuple(_reconstruct_axis(a) for a in d["axes"])
            return TensorSnapshot(
                name=d["name"],
                role=d["role"],
                data=arr,
                axes=axes,
                full_shape=tuple(d["full_shape"]),
                reduction=d["reduction"],
                stats=_reconstruct_stats(d["stats"]),
            )

        def _reconstruct_step(d: dict[str, Any]) -> Step:
            tensors = tuple(_reconstruct_tensor(t) for t in d.get("tensors", []))
            fid_d = d.get("fidelity", {})
            fidelity = Fidelity(
                status=fid_d.get("status", "unverified"),
                max_abs_err=fid_d.get("max_abs_err"),
                note=fid_d.get("note", ""),
            )
            return Step(
                id=d["id"],
                kind=d["kind"],
                semantic=d["semantic"],
                stage=d["stage"],
                block_index=d.get("block_index"),
                detail_level=d.get("detail_level", 2),
                module_path=d.get("module_path"),
                tensors=tensors,
                meta=d.get("meta", {}),
                fidelity=fidelity,
            )

        def _reconstruct_prediction(d: dict[str, Any] | None) -> Prediction | None:
            if d is None:
                return None
            return Prediction(
                candidate_ids=tuple(d["candidate_ids"]),
                candidate_tokens=tuple(d["candidate_tokens"]),
                candidate_probs=tuple(d["candidate_probs"]),
                chosen_id=d["chosen_id"],
                chosen_token=d["chosen_token"],
                strategy=d["strategy"],
            )

        def _reconstruct_pass(d: dict[str, Any]) -> PassRecord:
            steps = tuple(_reconstruct_step(s) for s in d.get("steps", []))
            pred = _reconstruct_prediction(d.get("prediction"))
            return PassRecord(
                index=d["index"],
                input_ids=tuple(d["input_ids"]),
                input_tokens=tuple(d["input_tokens"]),
                text_before=d["text_before"],
                steps=steps,
                prediction=pred,
            )

        m_d = manifest["model"]
        model = ModelSpecInfo(
            name=m_d["name"],
            adapter=m_d["adapter"],
            n_layers=m_d["n_layers"],
            emb_dim=m_d["emb_dim"],
            n_heads=m_d.get("n_heads"),
            vocab_size=m_d["vocab_size"],
            context_length=m_d.get("context_length"),
            n_params=m_d["n_params"],
            extras=m_d.get("extras", {}),
        )

        passes = tuple(_reconstruct_pass(p) for p in manifest.get("passes", []))

        trace = Trace(
            schema_version=manifest["schema_version"],
            tilevis_version=manifest.get("tilevis_version", "unknown"),
            created_at=manifest["created_at"],
            model=model,
            prompt=manifest["prompt"],
            settings=manifest.get("settings", {}),
            passes=passes,
        )

        validate_trace(trace, strict=True)
        return trace

    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TraceFormatError(f"Failed to parse trace data in {file_path.name}: {exc}") from exc
    finally:
        npz.close()
