"""Trace structural and numerical validation.

Single responsibility: enforce data integrity and invariants on Trace objects.
Allowed dependencies: stdlib and numpy only.
"""

from __future__ import annotations

import numpy as np

from tilevis.core.errors import TraceFormatError
from tilevis.core.kinds import StepKind
from tilevis.core.trace import Trace


def validate_trace(trace: Trace, strict: bool = False) -> list[str]:
    """Validate a Trace against all schema and numeric invariants.

    Args:
        trace: The Trace object to validate.
        strict: If True, raise TraceFormatError when any violation is found.

    Returns:
        List of human-readable issue descriptions (empty if valid).

    Raises:
        TraceFormatError: If strict is True and validation issues exist.
    """
    issues: list[str] = []

    # 1. Top-level metadata
    if trace.schema_version <= 0:
        issues.append(f"Invalid schema_version {trace.schema_version}: must be > 0")
    if not trace.tilevis_version.strip():
        issues.append("tilevis_version must be a non-empty string")
    if not trace.created_at.strip():
        issues.append("created_at must be a non-empty ISO-8601 string")

    # 2. Model specification
    model = trace.model
    if model.n_layers <= 0:
        issues.append(f"Model n_layers must be > 0, got {model.n_layers}")
    if model.emb_dim <= 0:
        issues.append(f"Model emb_dim must be > 0, got {model.emb_dim}")
    if model.vocab_size <= 0:
        issues.append(f"Model vocab_size must be > 0, got {model.vocab_size}")
    if model.n_heads is not None and model.n_heads <= 0:
        issues.append(f"Model n_heads must be > 0, got {model.n_heads}")

    # 3. Step ID uniqueness and passes validation
    seen_step_ids: set[str] = set()

    for p_idx, p in enumerate(trace.passes):
        if p.index != p_idx:
            issues.append(f"Pass index mismatch: expected {p_idx}, got {p.index}")

        # Validate input IDs and tokens length
        if len(p.input_ids) != len(p.input_tokens):
            issues.append(
                f"Pass {p_idx}: input_ids length ({len(p.input_ids)}) "
                f"does not match input_tokens length ({len(p.input_tokens)})"
            )

        # Validate steps in pass
        next_token_steps: list[int] = []

        for s_idx, s in enumerate(p.steps):
            # Check ID uniqueness
            if not s.id or not s.id.strip():
                issues.append(f"Pass {p_idx}, step {s_idx}: step id must be non-empty")
            elif s.id in seen_step_ids:
                issues.append(f"Duplicate step id {s.id!r} found in pass {p_idx}")
            else:
                seen_step_ids.add(s.id)

            # Check kind, semantic, stage strings
            if not s.kind or not str(s.kind).strip():
                issues.append(f"Step {s.id}: kind must be a non-empty string")
            if not s.semantic or not str(s.semantic).strip():
                issues.append(f"Step {s.id}: semantic must be a non-empty string")
            if not s.stage or not str(s.stage).strip():
                issues.append(f"Step {s.id}: stage must be a non-empty string")

            # Check block_index bounds
            if s.block_index is not None and (s.block_index < 0 or s.block_index >= model.n_layers):
                issues.append(
                    f"Step {s.id}: block_index {s.block_index} outside range [0, {model.n_layers})"
                )

            # Check detail_level
            if s.detail_level not in {1, 2, 3}:
                issues.append(f"Step {s.id}: detail_level {s.detail_level} must be 1, 2, or 3")

            # Check NEXT_TOKEN occurrences
            if s.kind == StepKind.NEXT_TOKEN or str(s.kind) == "NEXT_TOKEN":
                next_token_steps.append(s_idx)

            # Check tensors
            for _t_idx, t in enumerate(s.tensors):
                t_ctx = f"Step {s.id}, tensor {t.name!r} ({t.role})"

                # Check data finiteness
                if not np.all(np.isfinite(t.data)):
                    issues.append(f"{t_ctx}: data contains NaN or infinite values")

                # Dimensionality match
                if t.data.ndim != len(t.axes):
                    issues.append(
                        f"{t_ctx}: data ndim ({t.data.ndim}) != axes count ({len(t.axes)})"
                    )

                # Axes validation
                for a_idx, ax in enumerate(t.axes):
                    ax_len = len(ax.indices)
                    if a_idx < t.data.ndim and ax_len != t.data.shape[a_idx]:
                        issues.append(
                            f"{t_ctx}, axis {a_idx} ({ax.name}): indices length ({ax_len}) "
                            f"does not match tensor dimension ({t.data.shape[a_idx]})"
                        )

                    # Indices bounds and strictly increasing
                    if ax_len > 0:
                        if ax.indices[0] < 0:
                            issues.append(
                                f"{t_ctx}, axis {a_idx} ({ax.name}): min index {ax.indices[0]} < 0"
                            )
                        if ax.indices[-1] >= ax.full_size:
                            issues.append(
                                f"{t_ctx}, axis {a_idx} ({ax.name}): max index {ax.indices[-1]} "
                                f">= full_size {ax.full_size}"
                            )
                        if np.any(np.diff(ax.indices) <= 0):
                            issues.append(
                                f"{t_ctx}, axis {a_idx} ({ax.name}): not strictly increasing"
                            )

                    # Token axis labels check
                    if ax.name == "token":
                        if ax.labels is None:
                            issues.append(
                                f"{t_ctx}, axis {a_idx} ({ax.name}): token axis must have labels"
                            )
                        elif len(ax.labels) != ax_len:
                            issues.append(
                                f"{t_ctx}, axis {a_idx} ({ax.name}): labels len != indices len"
                            )

        # Each pass has at most one NEXT_TOKEN step, and it must be the last step
        if len(next_token_steps) > 1:
            issues.append(f"Pass {p_idx}: multiple NEXT_TOKEN steps at indices {next_token_steps}")
        elif len(next_token_steps) == 1 and next_token_steps[0] != len(p.steps) - 1:
            issues.append(
                f"Pass {p_idx}: NEXT_TOKEN step must be the pass's last step "
                f"(index {len(p.steps) - 1}), found at index {next_token_steps[0]}"
            )

        # 4. Prediction validation
        if p.prediction is not None:
            pred = p.prediction
            n_cands = len(pred.candidate_ids)
            if len(pred.candidate_tokens) != n_cands or len(pred.candidate_probs) != n_cands:
                issues.append(
                    f"Pass {p_idx}: Prediction candidate array lengths mismatch: "
                    f"ids={n_cands}, tokens={len(pred.candidate_tokens)}"
                )

            # Probabilities range and descending order
            probs = list(pred.candidate_probs)
            for prob_val in probs:
                if not (0.0 <= prob_val <= 1.0):
                    issues.append(
                        f"Pass {p_idx}: candidate probability {prob_val} not in [0.0, 1.0]"
                    )
            for i in range(len(probs) - 1):
                if probs[i] < probs[i + 1]:
                    issues.append(
                        f"Pass {p_idx}: candidate_probs not descending: {probs[i]} < {probs[i + 1]}"
                    )
                    break

    if strict and issues:
        raise TraceFormatError(
            f"Trace validation failed with {len(issues)} issue(s):\n" + "\n".join(issues)
        )

    return issues
