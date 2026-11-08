"""LayerNorm step builder for pre-layer and final layer normalization.

Single responsibility: construct LAYERNORM steps with scale/shift parameters and fidelity check.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step, TensorSnapshot
from tilevis.probe.adapters.base import BlockSpec, ModelSpec
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan, create_tensor_snapshot

logger = logging.getLogger(__name__)


def _extract_norm_weights(module: nn.Module) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract scale (weight) and shift (bias) parameter tensors from normalization module."""
    scale = getattr(module, "weight", None)
    shift = getattr(module, "bias", None)
    scale_t = scale.detach().cpu().to(torch.float32) if isinstance(scale, torch.Tensor) else None
    shift_t = shift.detach().cpu().to(torch.float32) if isinstance(shift, torch.Tensor) else None
    return scale_t, shift_t


class LayerNormStepBuilder:
    """Builds LAYERNORM steps for transformer blocks and final normalization."""

    def build_norm(
        self,
        norm_path: str,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        semantic: str,
        stage: str,
        block_index: int | None,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        eps: float = 1e-5,
    ) -> Step:
        step_suffix = (
            f"b{block_index}.{semantic.split('.')[-1]}" if block_index is not None else semantic
        )
        step_id = ctx.next_step_id(step_suffix)

        submod = model.get_submodule(norm_path) if norm_path else None
        scale_t, shift_t = _extract_norm_weights(submod) if submod is not None else (None, None)

        # Fidelity check
        with torch.inference_mode():
            mean = input_tensor.mean(dim=-1, keepdim=True)
            var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
            normed = (input_tensor - mean) / torch.sqrt(var + eps)
            if scale_t is not None:
                normed = normed * scale_t
            if shift_t is not None:
                normed = normed + shift_t
            err = (normed - output_tensor).abs().max().item()

        if err <= 1e-4:
            fidelity = Fidelity(status="exact", max_abs_err=err)
        elif err <= 1e-2:
            fidelity = Fidelity(status="approximate", max_abs_err=err)
        else:
            fidelity = Fidelity(status="failed", max_abs_err=err)

        labels_map = {0: ctx.input_tokens}
        tensors: list[TensorSnapshot] = []
        had_nf = False

        snap_in, nf_in = create_tensor_snapshot(
            name=f"{norm_path}.input",
            role=TensorRole.INPUT.value,
            tensor=input_tensor,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        tensors.append(snap_in)
        had_nf = had_nf or nf_in

        if scale_t is not None:
            snap_scale, nf_sc = create_tensor_snapshot(
                name=f"{norm_path}.scale",
                role=TensorRole.SCALE.value,
                tensor=scale_t,
                axis_names=["model_dim"],
                plan=plan,
            )
            tensors.append(snap_scale)
            had_nf = had_nf or nf_sc

        if shift_t is not None:
            snap_shift, nf_sh = create_tensor_snapshot(
                name=f"{norm_path}.shift",
                role=TensorRole.SHIFT.value,
                tensor=shift_t,
                axis_names=["model_dim"],
                plan=plan,
            )
            tensors.append(snap_shift)
            had_nf = had_nf or nf_sh

        snap_out, nf_out = create_tensor_snapshot(
            name=f"{norm_path}.output",
            role=TensorRole.OUTPUT.value,
            tensor=output_tensor,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        tensors.append(snap_out)
        had_nf = had_nf or nf_out

        meta: dict[str, Any] = {"eps": eps, "module": norm_path}
        if had_nf:
            meta["had_non_finite"] = True

        return Step(
            id=step_id,
            kind=StepKind.LAYERNORM.value,
            semantic=semantic,
            stage=stage,
            block_index=block_index,
            detail_level=int(DetailLevel.STANDARD),
            module_path=norm_path,
            tensors=tuple(tensors),
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
        # Dispatched by recorder per block or final_norm
        semantic = kwargs.get("semantic", Semantic.BLOCK_NORM1.value)
        block: BlockSpec | None = kwargs.get("block")

        if semantic == Semantic.BLOCK_NORM1.value and block and block.norm1:
            inp = capture.get_required(block.path, role="in")
            out = capture.get_required(block.norm1, role="out")
            return [
                self.build_norm(
                    block.norm1,
                    inp,
                    out,
                    semantic,
                    Stage.BLOCK.value,
                    block.index,
                    model,
                    plan,
                    ctx,
                )
            ]

        if semantic == Semantic.BLOCK_NORM2.value and block and block.norm2:
            inp = capture.get_required(block.norm2, role="in")
            out = capture.get_required(block.norm2, role="out")
            return [
                self.build_norm(
                    block.norm2,
                    inp,
                    out,
                    semantic,
                    Stage.BLOCK.value,
                    block.index,
                    model,
                    plan,
                    ctx,
                )
            ]

        if semantic == Semantic.FINAL_NORM.value and spec.final_norm:
            inp = capture.get_required(spec.final_norm, role="in")
            out = capture.get_required(spec.final_norm, role="out")
            return [
                self.build_norm(
                    spec.final_norm,
                    inp,
                    out,
                    semantic,
                    Stage.OUTPUT.value,
                    None,
                    model,
                    plan,
                    ctx,
                )
            ]

        return []


norm_builder = LayerNormStepBuilder()
STEP_BUILDERS.register(Semantic.BLOCK_NORM1.value)(norm_builder)
STEP_BUILDERS.register(Semantic.BLOCK_NORM2.value)(norm_builder)
STEP_BUILDERS.register(Semantic.FINAL_NORM.value)(norm_builder)
