"""Color normalization policies (per-tensor, shared-step, global-pass).

Single responsibility: scale float arrays to [-1, 1] or [0, 1] using outlier-robust stats.
Allowed dependencies: numpy, tilevis.core, stdlib.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from tilevis.core.kinds import TensorRole
from tilevis.core.registry import Registry
from tilevis.core.trace import PassRecord, Step, TensorSnapshot, TensorStats

logger = logging.getLogger("tilevis.render.normalization")

_PARAMETER_ROLES: frozenset[str] = frozenset(
    {
        TensorRole.WEIGHT.value,
        TensorRole.BIAS.value,
        TensorRole.SCALE.value,
        TensorRole.SHIFT.value,
    }
)

_PROBABILITY_ROLES: frozenset[str] = frozenset(
    {
        TensorRole.ATTN_WEIGHTS.value,
        TensorRole.PROBS.value,
    }
)


def safe_stat_scale(stats: TensorStats) -> float:
    """Compute outlier-robust positive scale from TensorStats, never returning 0."""
    if np.isfinite(stats.abs_p99) and stats.abs_p99 > 1e-12:
        return float(stats.abs_p99)
    if np.isfinite(stats.abs_max) and stats.abs_max > 1e-12:
        return float(stats.abs_max)
    return 1.0


class Normalizer(Protocol):
    """Protocol for computing snapshot color normalization scales."""

    name: str

    def compute_scale(
        self,
        snapshot: TensorSnapshot,
        step: Step,
        pass_record: PassRecord | None = None,
    ) -> float:
        """Return a positive floating-point normalization scale for the snapshot."""
        ...

    def normalize(
        self,
        snapshot: TensorSnapshot,
        step: Step,
        pass_record: PassRecord | None = None,
        *,
        signed: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Scale snapshot.data into [-1, 1] (signed) or [0, 1] (unsigned) and return scale."""
        ...


class _BaseNormalizer:
    """Shared array normalization implementation."""

    name: str

    def compute_scale(
        self,
        snapshot: TensorSnapshot,
        step: Step,
        pass_record: PassRecord | None = None,
    ) -> float:
        return safe_stat_scale(snapshot.stats)

    def normalize(
        self,
        snapshot: TensorSnapshot,
        step: Step,
        pass_record: PassRecord | None = None,
        *,
        signed: bool = True,
    ) -> tuple[np.ndarray, float]:
        scale = self.compute_scale(snapshot, step, pass_record)
        arr = np.nan_to_num(snapshot.data.astype(np.float32, copy=False), nan=0.0)
        scaled = arr / np.float32(scale)
        clipped = np.clip(scaled, -1.0, 1.0) if signed else np.clip(scaled, 0.0, 1.0)
        return np.asarray(clipped, dtype=np.float32), float(scale)


class PerTensorNormalizer(_BaseNormalizer):
    """Normalize each tensor independently using its own abs_p99."""

    name = "per_tensor"


class SharedStepNormalizer(_BaseNormalizer):
    """Share one activation scale across a step; parameter and prob tensors keep own scale."""

    name = "shared_step"

    def compute_scale(
        self,
        snapshot: TensorSnapshot,
        step: Step,
        pass_record: PassRecord | None = None,
    ) -> float:
        if snapshot.role in _PROBABILITY_ROLES:
            return max(1.0, float(snapshot.stats.max)) if np.isfinite(snapshot.stats.max) else 1.0
        if snapshot.role in _PARAMETER_ROLES:
            return safe_stat_scale(snapshot.stats)

        shared_scales = [
            safe_stat_scale(t.stats)
            for t in step.tensors
            if t.role not in _PARAMETER_ROLES and t.role not in _PROBABILITY_ROLES
        ]
        if not shared_scales:
            return safe_stat_scale(snapshot.stats)
        return max(shared_scales)


class GlobalPassNormalizer(_BaseNormalizer):
    """Share one residual-stream scale across all steps of the current pass."""

    name = "global_pass"

    def compute_scale(
        self,
        snapshot: TensorSnapshot,
        step: Step,
        pass_record: PassRecord | None = None,
    ) -> float:
        if snapshot.role in _PROBABILITY_ROLES:
            return max(1.0, float(snapshot.stats.max)) if np.isfinite(snapshot.stats.max) else 1.0
        if snapshot.role in _PARAMETER_ROLES:
            return safe_stat_scale(snapshot.stats)

        if pass_record is None:
            return SharedStepNormalizer().compute_scale(snapshot, step, None)

        scales = [
            safe_stat_scale(t.stats)
            for s in pass_record.steps
            for t in s.tensors
            if t.role not in _PARAMETER_ROLES and t.role not in _PROBABILITY_ROLES
        ]
        if not scales:
            return safe_stat_scale(snapshot.stats)
        return max(scales)


NORMALIZERS: Registry[str, Normalizer] = Registry("NORMALIZERS")
NORMALIZERS.register("per_tensor")(PerTensorNormalizer())
NORMALIZERS.register("shared_step")(SharedStepNormalizer())
NORMALIZERS.register("global_pass")(GlobalPassNormalizer())


def get_normalizer(name: str) -> Normalizer:
    """Retrieve a Normalizer by name, falling back to 'shared_step' if unknown."""
    key = name.strip().lower()
    if key in NORMALIZERS:
        return NORMALIZERS.get(key)
    logger.warning("Unknown normalizer %r; falling back to 'shared_step'", name)
    return NORMALIZERS.get("shared_step")
