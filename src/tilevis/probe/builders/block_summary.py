"""Block summary step builder.

Single responsibility: construct BLOCK_SUMMARY steps (STORY level) representing input,
output, and overall change per transformer block.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step
from tilevis.probe.adapters.base import BlockSpec, ModelSpec
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan, create_tensor_snapshot

logger = logging.getLogger(__name__)


class BlockSummaryStepBuilder:
    """Builds BLOCK_SUMMARY steps showing input, output, and change at STORY detail level."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        block: BlockSpec | None = kwargs.get("block")
        if block is None:
            return []

        b_idx = block.index
        step_id = ctx.next_step_id(f"b{b_idx}.summary")

        inp = capture.get_required(block.path, role="in")
        out = capture.get_required(block.path, role="out")
        update = out - inp

        labels_map = {0: ctx.input_tokens}

        s_in, nf1 = create_tensor_snapshot(
            "block.input",
            TensorRole.INPUT.value,
            inp,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        s_out, nf2 = create_tensor_snapshot(
            "block.output",
            TensorRole.OUTPUT.value,
            out,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        s_up, nf3 = create_tensor_snapshot(
            "block.change",
            TensorRole.UPDATE.value,
            update,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )

        meta: dict[str, Any] = {"block_index": b_idx}
        if nf1 or nf2 or nf3:
            meta["had_non_finite"] = True

        return [
            Step(
                id=step_id,
                kind=StepKind.BLOCK_SUMMARY.value,
                semantic=Semantic.BLOCK_SUMMARY.value,
                stage=Stage.BLOCK.value,
                block_index=b_idx,
                detail_level=int(DetailLevel.STORY),
                module_path=block.path,
                tensors=(s_in, s_out, s_up),
                meta=meta,
                fidelity=Fidelity("exact"),
            )
        ]


summary_builder = BlockSummaryStepBuilder()
STEP_BUILDERS.register(Semantic.BLOCK_SUMMARY.value)(summary_builder)
