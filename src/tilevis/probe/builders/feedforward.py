"""FeedForward step builder (expand, activation, contract).

Single responsibility: construct LINEAR and ACTIVATION steps for transformer MLP/FFN sublayers.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step, TensorSnapshot
from tilevis.probe.adapters.base import BlockSpec, ModelSpec
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan, create_tensor_snapshot
from tilevis.probe.weights import extract_weights

logger = logging.getLogger(__name__)


def _extract_weight_matrix(module: nn.Module | None) -> np.ndarray | None:
    """Extract weight matrix transposed to (in_features, out_features)."""
    if module is None:
        return None
    res = extract_weights(module)
    if res is not None and res.weight is not None:
        return res.weight
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return w.t().detach().cpu().to(torch.float32).numpy()
    return None


class FeedForwardStepBuilder:
    """Builds the 3 sublayer steps of the FeedForward block."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        semantic = kwargs.get("semantic", Semantic.BLOCK_FF_EXPAND.value)
        block: BlockSpec | None = kwargs.get("block")
        if block is None:
            return []

        b_idx = block.index
        labels_map = {0: ctx.input_tokens}

        # 1. Expand
        if semantic == Semantic.BLOCK_FF_EXPAND.value:
            step_id = ctx.next_step_id(f"b{b_idx}.ff.expand")
            inp = capture.get(block.norm2, role="out") if block.norm2 else None
            if inp is None and block.ff_in:
                inp = capture.get(block.ff_in, role="in")
            if inp is None:
                inp = capture.get_required(block.path, role="in")

            out = capture.get(block.ff_in, role="out") if block.ff_in else None
            mod_in = model.get_submodule(block.ff_in) if block.ff_in else None
            if out is None and mod_in is not None:
                with torch.inference_mode():
                    device = next(mod_in.parameters()).device
                    out = mod_in(inp.to(device)).detach().cpu().to(torch.float32)

            if out is None:
                return []

            tensors: list[TensorSnapshot] = []
            s_in, nf1 = create_tensor_snapshot(
                "ff.expand.input",
                TensorRole.INPUT.value,
                inp,
                ["token", "model_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            tensors.append(s_in)

            w_in = _extract_weight_matrix(mod_in)
            if w_in is not None:
                s_w, nf2 = create_tensor_snapshot(
                    f"{block.ff_in}.weight",
                    TensorRole.WEIGHT.value,
                    w_in,
                    ["model_dim", "ff_dim"],
                    plan,
                )
                tensors.append(s_w)

            s_out, nf3 = create_tensor_snapshot(
                "ff.expand.output",
                TensorRole.OUTPUT.value,
                out,
                ["token", "ff_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            tensors.append(s_out)

            meta: dict[str, Any] = {}
            if nf1 or nf2 or nf3:
                meta["had_non_finite"] = True

            return [
                Step(
                    id=step_id,
                    kind=StepKind.LINEAR.value,
                    semantic=semantic,
                    stage=Stage.BLOCK.value,
                    block_index=b_idx,
                    detail_level=int(DetailLevel.STANDARD),
                    module_path=block.ff_in,
                    tensors=tuple(tensors),
                    meta=meta,
                    fidelity=Fidelity("exact"),
                )
            ]

        # 2. Activation
        if semantic == Semantic.BLOCK_FF_ACT.value:
            step_id = ctx.next_step_id(f"b{b_idx}.ff.act")
            inp = capture.get(block.ff_in, role="out") if block.ff_in else None
            out = capture.get(block.ff_act, role="out") if block.ff_act else None

            if out is None and inp is not None:
                out = F.gelu(inp)

            if inp is None or out is None:
                return []

            s_in, nf1 = create_tensor_snapshot(
                "ff.act.input",
                TensorRole.INPUT.value,
                inp,
                ["token", "ff_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            s_out, nf2 = create_tensor_snapshot(
                "ff.act.output",
                TensorRole.OUTPUT.value,
                out,
                ["token", "ff_dim"],
                plan,
                labels_by_axis=labels_map,
            )

            meta = {"activation": "GELU"}
            if nf1 or nf2:
                meta["had_non_finite"] = True

            return [
                Step(
                    id=step_id,
                    kind=StepKind.ACTIVATION.value,
                    semantic=semantic,
                    stage=Stage.BLOCK.value,
                    block_index=b_idx,
                    detail_level=int(DetailLevel.STANDARD),
                    module_path=block.ff_act,
                    tensors=(s_in, s_out),
                    meta=meta,
                    fidelity=Fidelity("exact"),
                )
            ]

        # 3. Contract
        if semantic == Semantic.BLOCK_FF_CONTRACT.value:
            step_id = ctx.next_step_id(f"b{b_idx}.ff.contract")
            inp = capture.get(block.ff_act, role="out") if block.ff_act else None
            if inp is None and block.ff_in:
                inp_expand = capture.get(block.ff_in, role="out")
                if inp_expand is not None:
                    inp = F.gelu(inp_expand)

            out = capture.get(block.ff_out, role="out") if block.ff_out else None
            if out is None and block.ff:
                out = capture.get(block.ff, role="out")

            mod_out = model.get_submodule(block.ff_out) if block.ff_out else None

            if inp is None or out is None:
                return []

            tensors = []
            s_in, nf1 = create_tensor_snapshot(
                "ff.contract.input",
                TensorRole.INPUT.value,
                inp,
                ["token", "ff_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            tensors.append(s_in)

            w_out = _extract_weight_matrix(mod_out)
            if w_out is not None:
                s_w, nf2 = create_tensor_snapshot(
                    f"{block.ff_out}.weight",
                    TensorRole.WEIGHT.value,
                    w_out,
                    ["ff_dim", "model_dim"],
                    plan,
                )
                tensors.append(s_w)

            s_out, nf3 = create_tensor_snapshot(
                "ff.contract.output",
                TensorRole.OUTPUT.value,
                out,
                ["token", "model_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            tensors.append(s_out)

            meta = {}
            if nf1 or nf3:
                meta["had_non_finite"] = True

            return [
                Step(
                    id=step_id,
                    kind=StepKind.LINEAR.value,
                    semantic=semantic,
                    stage=Stage.BLOCK.value,
                    block_index=b_idx,
                    detail_level=int(DetailLevel.STANDARD),
                    module_path=block.ff_out,
                    tensors=tuple(tensors),
                    meta=meta,
                    fidelity=Fidelity("exact"),
                )
            ]

        return []


ff_builder = FeedForwardStepBuilder()
STEP_BUILDERS.register(Semantic.BLOCK_FF_EXPAND.value)(ff_builder)
STEP_BUILDERS.register(Semantic.BLOCK_FF_ACT.value)(ff_builder)
STEP_BUILDERS.register(Semantic.BLOCK_FF_CONTRACT.value)(ff_builder)
