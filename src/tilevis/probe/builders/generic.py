"""Generic step builder fallback for arbitrary neural network architectures.

Single responsibility: construct baseline GENERIC/LINEAR/LAYERNORM/ACTIVATION steps
when standard transformer adapter structures are not present.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step, TensorSnapshot
from tilevis.probe.adapters.base import ModelSpec
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan, create_tensor_snapshot

logger = logging.getLogger(__name__)


def _classify_axes_and_labels(
    tensor: torch.Tensor, tokens: tuple[str, ...]
) -> tuple[list[str], dict[int, tuple[str, ...]] | None]:
    """Determine axis names and token label mappings matching tensor dimensions."""
    if tensor.ndim == 1:
        return ["model_dim"], None
    if tensor.ndim == 2:
        return ["token", "model_dim"], {0: tokens}
    if tensor.ndim == 3:
        return ["head", "token", "token"], {1: tokens, 2: tokens}
    return [f"dim_{i}" for i in range(tensor.ndim)], None


class GenericStepBuilder:
    """Builds fallback steps for models using GenericAdapter."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        steps: list[Step] = []

        for (mod_path, call_idx, role), tensor in sorted(
            capture._storage.items(), key=lambda x: (x[0][1], x[0][0])
        ):
            if role != "out":
                continue

            submod = model.get_submodule(mod_path) if mod_path else None
            mod_type = type(submod).__name__ if submod is not None else "Module"

            step_id = ctx.next_step_id(f"gen.{mod_path}.{call_idx}")

            # Classify kind
            if "Linear" in mod_type:
                kind = StepKind.LINEAR.value
            elif "Norm" in mod_type:
                kind = StepKind.LAYERNORM.value
            elif any(act in mod_type.lower() for act in ["gelu", "relu", "silu", "tanh"]):
                kind = StepKind.ACTIVATION.value
            else:
                kind = StepKind.GENERIC.value

            tensors: list[TensorSnapshot] = []

            # Input if available
            in_t = capture.get(mod_path, call_idx=call_idx, role="in")
            if in_t is not None:
                in_axes, in_labels = _classify_axes_and_labels(in_t, ctx.input_tokens)
                s_in, _ = create_tensor_snapshot(
                    f"{mod_path}.input",
                    TensorRole.INPUT.value,
                    in_t,
                    in_axes,
                    plan,
                    labels_by_axis=in_labels,
                )
                tensors.append(s_in)

            # Output
            out_axes, out_labels = _classify_axes_and_labels(tensor, ctx.input_tokens)
            s_out, _ = create_tensor_snapshot(
                f"{mod_path}.output",
                TensorRole.OUTPUT.value,
                tensor,
                out_axes,
                plan,
                labels_by_axis=out_labels,
            )
            tensors.append(s_out)

            steps.append(
                Step(
                    id=step_id,
                    kind=kind,
                    semantic=Semantic.GENERIC.value,
                    stage=Stage.BLOCK.value,
                    block_index=None,
                    detail_level=int(DetailLevel.STANDARD),
                    module_path=mod_path,
                    tensors=tuple(tensors),
                    meta={"module_type": mod_type, "call_index": call_idx},
                    fidelity=Fidelity("exact"),
                )
            )

        return steps


generic_builder = GenericStepBuilder()
STEP_BUILDERS.register(Semantic.GENERIC.value)(generic_builder)
