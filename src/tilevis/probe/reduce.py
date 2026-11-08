"""Dimensionality reduction, indexing plans, and tensor snapshot construction.

Single responsibility: downsample large activations and weights into displayable
snapshots with consistent cross-step coordinate alignment.
Allowed dependencies: stdlib, numpy, torch, tilevis.core.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from tilevis.core.kinds import TensorRole
from tilevis.core.registry import Registry
from tilevis.core.trace import AxisInfo, TensorSnapshot, TensorStats
from tilevis.probe.adapters.base import ModelSpec

logger = logging.getLogger(__name__)

ReducerFn = Callable[[np.ndarray, Sequence[np.ndarray]], np.ndarray]

REDUCERS: Registry[str, ReducerFn] = Registry("reducers")


def compute_stride_indices(full_size: int, limit: int) -> np.ndarray:
    """Compute evenly spaced, deduplicated, strictly increasing indices within bounds.

    Args:
        full_size: Dimension length before reduction.
        limit: Maximum number of indices to select.

    Returns:
        1-D int64 numpy array of indices in [0, full_size).
    """
    if full_size <= limit or limit <= 0:
        return np.arange(full_size, dtype=np.int64)

    raw = np.linspace(0, full_size - 1, limit).round().astype(np.int64)
    unique = np.unique(raw)
    return unique.astype(np.int64)


@REDUCERS.register("none")
def _reduce_none(arr: np.ndarray, indices: Sequence[np.ndarray]) -> np.ndarray:
    """Identity reducer: leaves array untouched."""
    return arr


@REDUCERS.register("stride")
def _reduce_stride(arr: np.ndarray, indices: Sequence[np.ndarray]) -> np.ndarray:
    """Subsamples array along each dimension using specified axis indices."""
    if not indices or arr.ndim == 0:
        return arr
    return arr[np.ix_(*indices)]


@REDUCERS.register("block_mean")
def _reduce_block_mean(arr: np.ndarray, indices: Sequence[np.ndarray]) -> np.ndarray:
    """Stride-based fallback for block averaging."""
    return _reduce_stride(arr, indices)


@REDUCERS.register("max_abs")
def _reduce_max_abs(arr: np.ndarray, indices: Sequence[np.ndarray]) -> np.ndarray:
    """Stride-based fallback for max-abs pooling."""
    return _reduce_stride(arr, indices)


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """Coordinate alignment plan determining displayed indices per axis family."""

    model_dim: np.ndarray
    ff_dim: np.ndarray
    head_dim: np.ndarray
    head: np.ndarray
    max_features: int = 64
    max_tokens: int = 24
    max_heads: int = 12

    @classmethod
    def from_model_spec(
        cls,
        spec: ModelSpec,
        max_features: int = 64,
        max_tokens: int = 24,
        max_heads: int = 12,
    ) -> IndexPlan:
        """Construct IndexPlan from model architecture specification and settings."""
        emb_dim = spec.emb_dim
        ff_dim = 4 * emb_dim

        n_heads = spec.n_heads if spec.n_heads is not None and spec.n_heads > 0 else 1
        head_dim = max(1, emb_dim // n_heads)

        idx_model = compute_stride_indices(emb_dim, max_features)
        idx_ff = compute_stride_indices(ff_dim, max_features)
        idx_head_dim = compute_stride_indices(head_dim, max_features)
        idx_head = compute_stride_indices(n_heads, max_heads)

        return cls(
            model_dim=idx_model,
            ff_dim=idx_ff,
            head_dim=idx_head_dim,
            head=idx_head,
            max_features=max_features,
            max_tokens=max_tokens,
            max_heads=max_heads,
        )

    def get_token_indices(self, full_seq_len: int) -> np.ndarray:
        """Compute token indices keeping the most recent tokens (up to max_tokens)."""
        if full_seq_len <= self.max_tokens:
            return np.arange(full_seq_len, dtype=np.int64)
        return np.arange(full_seq_len - self.max_tokens, full_seq_len, dtype=np.int64)

    def get_axis_indices(self, axis_name: str, full_size: int) -> np.ndarray:
        """Get precalculated or dynamically generated indices for an axis name."""
        if axis_name == "token":
            return self.get_token_indices(full_size)
        if axis_name == "model_dim":
            if full_size == len(self.model_dim):
                return self.model_dim
            return compute_stride_indices(full_size, self.max_features)
        if axis_name == "ff_dim":
            if full_size == len(self.ff_dim):
                return self.ff_dim
            return compute_stride_indices(full_size, self.max_features)
        if axis_name == "head_dim":
            if full_size == len(self.head_dim):
                return self.head_dim
            return compute_stride_indices(full_size, self.max_features)
        if axis_name == "head":
            if full_size == len(self.head):
                return self.head
            return compute_stride_indices(full_size, self.max_heads)
        if axis_name == "vocab":
            return compute_stride_indices(full_size, self.max_features)
        return compute_stride_indices(full_size, self.max_features)


def compute_tensor_stats(arr: np.ndarray) -> TensorStats:
    """Compute summary statistics across the entire unreduced tensor."""
    if arr.size == 0:
        return TensorStats(
            min=0.0,
            max=0.0,
            mean=0.0,
            std=0.0,
            abs_max=0.0,
            abs_p99=0.0,
        )

    arr_flat = arr.ravel()
    min_val = float(np.min(arr_flat))
    max_val = float(np.max(arr_flat))
    mean_val = float(np.mean(arr_flat))
    std_val = float(np.std(arr_flat))

    abs_arr = np.abs(arr_flat)
    abs_max = float(np.max(abs_arr))
    abs_p99 = float(np.percentile(abs_arr, 99.0))

    return TensorStats(
        min=min_val,
        max=max_val,
        mean=mean_val,
        std=std_val,
        abs_max=abs_max,
        abs_p99=abs_p99,
    )


def create_tensor_snapshot(
    name: str,
    role: str | TensorRole,
    tensor: np.ndarray | torch.Tensor,
    axis_names: Sequence[str],
    plan: IndexPlan,
    labels_by_axis: Mapping[int, Sequence[str]] | None = None,
    custom_indices_by_axis: Mapping[int, np.ndarray] | None = None,
    reduction: str = "stride",
) -> tuple[TensorSnapshot, bool]:
    """Create an immutable TensorSnapshot with full statistics and reduced data.

    Args:
        name: Human-readable name.
        role: TensorRole or role string.
        tensor: Unreduced data tensor or array.
        axis_names: Sequence of axis names corresponding to tensor dimensions.
        plan: IndexPlan providing dimension reduction targets.
        labels_by_axis: Optional mapping from dimension index to label sequence.
        custom_indices_by_axis: Optional explicit indices for specific dimensions.
        reduction: Reducer name to apply.

    Returns:
        Tuple of (TensorSnapshot, had_non_finite_flag).
    """
    if isinstance(tensor, torch.Tensor):
        data_np = tensor.detach().cpu().to(torch.float32).numpy()
    else:
        data_np = np.asarray(tensor, dtype=np.float32)

    had_non_finite = bool(not np.all(np.isfinite(data_np)))
    if had_non_finite:
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

    full_shape = tuple(data_np.shape)
    stats = compute_tensor_stats(data_np)

    role_str = role.value if isinstance(role, TensorRole) else str(role)

    axes: list[AxisInfo] = []
    slice_indices: list[np.ndarray] = []

    for dim_idx, axis_name in enumerate(axis_names):
        full_size = full_shape[dim_idx]
        if custom_indices_by_axis and dim_idx in custom_indices_by_axis:
            indices = np.asarray(custom_indices_by_axis[dim_idx], dtype=np.int64)
        else:
            indices = plan.get_axis_indices(axis_name, full_size)

        # Ensure indices strictly increasing and in bounds
        if len(indices) > 0 and (indices[0] < 0 or indices[-1] >= full_size):
            indices = indices[(indices >= 0) & (indices < full_size)]
        indices = np.unique(indices).astype(np.int64)

        sliced_labels: tuple[str, ...] | None = None
        if labels_by_axis and dim_idx in labels_by_axis:
            full_labels = labels_by_axis[dim_idx]
            sliced_labels = tuple(
                str(full_labels[idx]) for idx in indices if 0 <= idx < len(full_labels)
            )

        axes.append(
            AxisInfo(
                name=axis_name,
                full_size=full_size,
                indices=indices,
                labels=sliced_labels,
            )
        )
        slice_indices.append(indices)

    # Apply reduction
    if reduction in REDUCERS and reduction != "none":
        reducer_fn = REDUCERS.get(reduction)
        reduced_data = reducer_fn(data_np, slice_indices)
    else:
        reduced_data = _reduce_stride(data_np, slice_indices)

    is_reduced = any(len(ax.indices) < ax.full_size for ax in axes)
    applied_reduction = reduction if is_reduced else "none"

    snapshot = TensorSnapshot(
        name=name,
        role=role_str,
        data=reduced_data,
        axes=tuple(axes),
        full_shape=full_shape,
        reduction=applied_reduction,
        stats=stats,
    )
    return snapshot, had_non_finite
