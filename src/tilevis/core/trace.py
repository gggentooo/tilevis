"""Trace, PassRecord, Step, and TensorSnapshot immutable dataclasses.

Single responsibility: define the in-memory data structures representing an execution trace.
Allowed dependencies: stdlib and numpy only.

Orientation convention:
Every weight matrix is stored as (in x out).
Activations are stored as (token x feature).
Therefore, the mathematical relationship:
    output_row = input_row @ weight
holds directly for visual layouts. Note that PyTorch's nn.Linear.weight is stored
in (out_features, in_features) format and must be transposed by weight extractors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tilevis.core.kinds import TensorRole

TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AxisInfo:
    """Metadata describing a single axis of a tensor.

    Attributes:
        name: Name of axis ('token', 'model_dim', 'ff_dim', 'head_dim', 'vocab', 'head').
        full_size: Size of dimension before reduction.
        indices: 1-D int64 array indicating which original indices are displayed.
        labels: Optional tuple of human-readable labels (e.g. token strings).
    """

    name: str
    full_size: int
    indices: np.ndarray
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        idx = np.asarray(self.indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"AxisInfo indices must be 1-dimensional, got ndim={idx.ndim}")
        idx.setflags(write=False)
        object.__setattr__(self, "indices", idx)
        if self.labels is not None and not isinstance(self.labels, tuple):
            object.__setattr__(self, "labels", tuple(self.labels))


@dataclass(frozen=True, slots=True)
class TensorStats:
    """Summary statistics computed on the full tensor prior to reduction."""

    min: float
    max: float
    mean: float
    std: float
    abs_max: float
    abs_p99: float


@dataclass(frozen=True, slots=True)
class TensorSnapshot:
    """Displayable snapshot of a tensor at a specific computation point.

    Attributes:
        name: Debug/human-readable name (e.g., 'trf_blocks.0.att.W_query.weight').
        role: Semantic role string (TensorRole).
        data: Reduced float32 array (1-D, 2-D, or 3-D for attention heads).
        axes: AxisInfo descriptor for each dimension of data.
        full_shape: Original unreduced shape of tensor without batch dimension.
        reduction: Name of reducer strategy applied ('none', 'stride', etc.).
        stats: Summary statistics computed on the full tensor.
    """

    name: str
    role: str
    data: np.ndarray
    axes: tuple[AxisInfo, ...]
    full_shape: tuple[int, ...]
    reduction: str
    stats: TensorStats

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        arr.setflags(write=False)
        object.__setattr__(self, "data", arr)
        if not isinstance(self.axes, tuple):
            object.__setattr__(self, "axes", tuple(self.axes))
        if not isinstance(self.full_shape, tuple):
            object.__setattr__(self, "full_shape", tuple(self.full_shape))


@dataclass(frozen=True, slots=True)
class Fidelity:
    """Numerical fidelity metric comparing recomputed/hooked values to ground truth.

    Attributes:
        status: Fidelity grade ('exact', 'approximate', 'unverified', 'failed').
        max_abs_err: Maximum absolute difference against reference tensor, if verified.
        note: Diagnostic information or warning details.
    """

    status: str
    max_abs_err: float | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Step:
    """A discrete operation step within a forward pass.

    Attributes:
        id: Unique step identifier within trace (e.g., 'p0.b0.att.scores').
        kind: Visual rendering kind (StepKind string).
        semantic: Semantic meaning (Semantic string).
        stage: Pipeline stage ('INPUT', 'BLOCK', 'OUTPUT').
        block_index: 0-indexed transformer block number, or None for global steps.
        detail_level: Minimum DetailLevel (1=Story, 2=Standard, 3=Deep) to show this step.
        module_path: PyTorch named module path (e.g., 'trf_blocks.0.att').
        tensors: Snapshots of tensors relevant to this step.
        meta: JSON-serializable arbitrary metadata dictionary.
        fidelity: Verification record.
    """

    id: str
    kind: str
    semantic: str
    stage: str
    block_index: int | None
    detail_level: int
    module_path: str | None
    tensors: tuple[TensorSnapshot, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)
    fidelity: Fidelity = field(default_factory=lambda: Fidelity("unverified"))

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, tuple):
            object.__setattr__(self, "tensors", tuple(self.tensors))

    def by_role(self, role: str | TensorRole) -> tuple[TensorSnapshot, ...]:
        """Return all tensor snapshots in this step matching the requested role."""
        target_role = role.value if isinstance(role, TensorRole) else str(role)
        return tuple(t for t in self.tensors if t.role == target_role)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Top-k next-word candidate ranking and chosen token prediction.

    Attributes:
        candidate_ids: Token IDs of top candidates sorted by probability descending.
        candidate_tokens: Decoded display strings of top candidates.
        candidate_probs: Probabilities in [0.0, 1.0] sorted descending.
        chosen_id: Final token ID chosen by sampling or greedy argmax.
        chosen_token: Display string of chosen token.
        strategy: Generation strategy ('greedy' or 'sample').
    """

    candidate_ids: tuple[int, ...]
    candidate_tokens: tuple[str, ...]
    candidate_probs: tuple[float, ...]
    chosen_id: int
    chosen_token: str
    strategy: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_ids, tuple):
            object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        if not isinstance(self.candidate_tokens, tuple):
            object.__setattr__(self, "candidate_tokens", tuple(self.candidate_tokens))
        if not isinstance(self.candidate_probs, tuple):
            object.__setattr__(
                self, "candidate_probs", tuple(float(p) for p in self.candidate_probs)
            )


@dataclass(frozen=True, slots=True)
class PassRecord:
    """Record of a single autoregressive generation forward pass.

    Attributes:
        index: 0-based index of this forward pass.
        input_ids: Sequence of input token IDs fed to the model for this pass.
        input_tokens: Human-readable display strings for input tokens.
        text_before: Full text prompt/prefix before this pass generated a token.
        steps: Ordered sequence of computation steps in this pass.
        prediction: Next-token prediction, or None if generation terminated (EOS).
    """

    index: int
    input_ids: tuple[int, ...]
    input_tokens: tuple[str, ...]
    text_before: str
    steps: tuple[Step, ...] = ()
    prediction: Prediction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, tuple):
            object.__setattr__(self, "input_ids", tuple(self.input_ids))
        if not isinstance(self.input_tokens, tuple):
            object.__setattr__(self, "input_tokens", tuple(self.input_tokens))
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))

    def steps_for_block(self, b: int | None) -> list[Step]:
        """Return all steps in this pass belonging to block index b."""
        return [s for s in self.steps if s.block_index == b]


@dataclass(frozen=True, slots=True)
class ModelSpecInfo:
    """Serializable summary of the model architecture specification."""

    name: str
    adapter: str
    n_layers: int
    emb_dim: int
    n_heads: int | None
    vocab_size: int
    context_length: int | None
    n_params: int
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Trace:
    """Top-level immutable container holding an entire recorded visualization session."""

    schema_version: int
    tilevis_version: str
    created_at: str
    model: ModelSpecInfo
    prompt: str
    settings: Mapping[str, Any]
    passes: tuple[PassRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.passes, tuple):
            object.__setattr__(self, "passes", tuple(self.passes))

    def step_by_id(self, step_id: str) -> Step:
        """Find a step by its unique id across all passes, or raise KeyError."""
        for p in self.passes:
            for s in p.steps:
                if s.id == step_id:
                    return s
        raise KeyError(f"Step with id {step_id!r} not found in trace.")
