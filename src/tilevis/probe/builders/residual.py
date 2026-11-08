"""Residual addition step builder with numerical verification.

Single responsibility: construct RESIDUAL_ADD steps verifying shortcut + update == output.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step
from tilevis.probe.adapters.base import BlockSpec, ModelSpec
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan, create_tensor_snapshot

logger = logging.getLogger(__name__)


class ResidualStepBuilder:
    """Builds RESIDUAL_ADD steps with shortcut + update identity verification."""

    def build_residual(
        self,
        shortcut: torch.Tensor,
        update: torch.Tensor,
        out: torch.Tensor,
        semantic: str,
        block_index: int,
        plan: IndexPlan,
        ctx: StepBuildContext,
        residual_style: str = "pre_norm",
    ) -> Step:
        step_suffix = f"b{block_index}.{semantic.split('.')[-1]}"
        step_id = ctx.next_step_id(step_suffix)

        with torch.inference_mode():
            diff = float((out - (shortcut + update)).abs().max().item())

        if residual_style != "pre_norm":
            status = "failed"
            note = f"Non-standard residual style: {residual_style}"
        elif diff <= 1e-4:
            status = "exact"
            note = ""
        elif diff <= 1e-2:
            status = "approximate"
            note = ""
        else:
            status = "failed"
            note = f"Residual identity mismatch (err={diff:.4e})"

        fidelity = Fidelity(status=status, max_abs_err=diff, note=note)

        labels_map = {0: ctx.input_tokens}
        snap_sc, nf1 = create_tensor_snapshot(
            "shortcut",
            TensorRole.SHORTCUT.value,
            shortcut,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        snap_up, nf2 = create_tensor_snapshot(
            "update",
            TensorRole.UPDATE.value,
            update,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        snap_out, nf3 = create_tensor_snapshot(
            "output",
            TensorRole.OUTPUT.value,
            out,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )

        meta: dict[str, Any] = {"residual_style": residual_style}
        if nf1 or nf2 or nf3:
            meta["had_non_finite"] = True

        step_kind = StepKind.RESIDUAL_ADD.value if status != "failed" else StepKind.GENERIC.value

        return Step(
            id=step_id,
            kind=step_kind,
            semantic=semantic,
            stage=Stage.BLOCK.value,
            block_index=block_index,
            detail_level=int(DetailLevel.STANDARD),
            module_path=None,
            tensors=(snap_sc, snap_up, snap_out),
            meta=meta,
            fidelity=fidelity,
        )

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        semantic = kwargs.get("semantic", Semantic.BLOCK_RESID1.value)
        block: BlockSpec | None = kwargs.get("block")
        if block is None:
            return []

        b_idx = block.index

        if semantic == Semantic.BLOCK_RESID1.value:
            shortcut = capture.get_required(block.path, role="in")
            update = capture.get(block.attention, role="out") if block.attention else None
            if update is None and block.out_proj:
                update = capture.get_required(block.out_proj, role="out")
            if update is None:
                return []
            out = capture.get_required(block.norm2, role="in") if block.norm2 else None
            if out is None:
                # If no norm2 pre-hook, out = shortcut + update
                out = shortcut + update

            return [
                self.build_residual(
                    shortcut=shortcut,
                    update=update,
                    out=out,
                    semantic=semantic,
                    block_index=b_idx,
                    plan=plan,
                    ctx=ctx,
                    residual_style=spec.residual_style,
                )
            ]

        if semantic == Semantic.BLOCK_RESID2.value:
            shortcut = (
                capture.get_required(block.norm2, role="in")
                if block.norm2
                else capture.get_required(block.path, role="in")
            )
            update = capture.get_required(block.ff, role="out") if block.ff else None
            if update is None and block.ff_out:
                update = capture.get_required(block.ff_out, role="out")
            out = capture.get_required(block.path, role="out")

            if update is None:
                return []

            return [
                self.build_residual(
                    shortcut=shortcut,
                    update=update,
                    out=out,
                    semantic=semantic,
                    block_index=b_idx,
                    plan=plan,
                    ctx=ctx,
                    residual_style=spec.residual_style,
                )
            ]

        return []


resid_builder = ResidualStepBuilder()
STEP_BUILDERS.register(Semantic.BLOCK_RESID1.value)(resid_builder)
STEP_BUILDERS.register(Semantic.BLOCK_RESID2.value)(resid_builder)
