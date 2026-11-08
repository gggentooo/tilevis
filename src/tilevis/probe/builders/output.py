"""Output stage step builders: final norm, LM head logits, and next token prediction.

Single responsibility: construct LOGITS and NEXT_TOKEN steps with top-k probability distributions
sourced from ground-truth forward pass logits.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step, TensorSnapshot
from tilevis.probe.adapters.base import ModelSpec
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


class OutputStepBuilder:
    """Builds LM_HEAD (LOGITS) and NEXT_TOKEN steps using ground truth prediction logits."""

    def build_lm_head(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
    ) -> Step:
        step_id = ctx.next_step_id("lm_head")

        # Final norm output provides input to lm_head
        fn_out = capture.get(spec.final_norm, role="out") if spec.final_norm else None
        if fn_out is None and spec.blocks:
            fn_out = capture.get(spec.blocks[-1].path, role="out")
        if fn_out is None:
            fn_out = capture.get_required(spec.lm_head, role="in")

        # Use the LAST token's vector only
        last_vec = fn_out[-1:].clone()  # Shape (1, emb_dim)
        last_token_label = ctx.input_tokens[-1] if ctx.input_tokens else ""

        pred = ctx.prediction
        if pred is not None:
            cand_ids = np.array(pred.candidate_ids, dtype=np.int64)
            cand_tokens = pred.candidate_tokens
        else:
            cand_ids = np.arange(min(10, spec.vocab_size), dtype=np.int64)
            cand_tokens = tuple(ctx.tokenizer.token_display(int(i)) for i in cand_ids)

        mod_head = model.get_submodule(spec.lm_head) if spec.lm_head else None
        w_head = _extract_weight_matrix(mod_head)

        tensors: list[TensorSnapshot] = []
        labels_map_in = {0: (last_token_label,)}

        s_in, nf1 = create_tensor_snapshot(
            "last_token_vector",
            TensorRole.INPUT.value,
            last_vec,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map_in,
        )
        tensors.append(s_in)

        # Weight columns corresponding to top-k candidates
        if w_head is not None:
            w_top_k = w_head[:, cand_ids]  # (emb_dim, k)
            s_w, _nf2 = create_tensor_snapshot(
                f"{spec.lm_head}.candidate_weights",
                TensorRole.WEIGHT.value,
                w_top_k,
                ["model_dim", "vocab"],
                plan,
                labels_by_axis={1: cand_tokens},
                custom_indices_by_axis={1: cand_ids},
            )
            tensors.append(s_w)

        # Hooked or recomputed candidate logits
        hooked_logits = capture.get(spec.lm_head, role="out")
        if hooked_logits is not None:
            last_logits_cand = hooked_logits[-1:, cand_ids]
        elif mod_head is not None:
            with torch.inference_mode():
                device = next(mod_head.parameters()).device
                out_t = mod_head(last_vec.to(device)).detach().cpu().to(torch.float32)
                last_logits_cand = out_t[:, cand_ids]
        else:
            last_logits_cand = torch.zeros((1, len(cand_ids)), dtype=torch.float32)

        s_out, nf3 = create_tensor_snapshot(
            "candidate_logits",
            TensorRole.LOGITS.value,
            last_logits_cand,
            ["token", "vocab"],
            plan,
            labels_by_axis={0: (last_token_label,), 1: cand_tokens},
            custom_indices_by_axis={1: cand_ids},
        )
        tensors.append(s_out)

        meta: dict[str, Any] = {
            "top_k": len(cand_ids),
            "candidate_tokens": list(cand_tokens),
        }
        if nf1 or nf3:
            meta["had_non_finite"] = True

        return Step(
            id=step_id,
            kind=StepKind.LOGITS.value,
            semantic=Semantic.LM_HEAD.value,
            stage=Stage.OUTPUT.value,
            block_index=None,
            detail_level=int(DetailLevel.STANDARD),
            module_path=spec.lm_head,
            tensors=tuple(tensors),
            meta=meta,
            fidelity=Fidelity("exact"),
        )

    def build_next_token(
        self,
        spec: ModelSpec,
        plan: IndexPlan,
        ctx: StepBuildContext,
    ) -> Step:
        step_id = ctx.next_step_id("next_token")
        pred = ctx.prediction

        if pred is not None:
            cand_ids = np.array(pred.candidate_ids, dtype=np.int64)
            cand_tokens = pred.candidate_tokens
            cand_probs = np.array(pred.candidate_probs, dtype=np.float32)
            chosen_id = pred.chosen_id
            chosen_token = pred.chosen_token
            strategy = pred.strategy
        else:
            cand_ids = np.array([0], dtype=np.int64)
            cand_tokens = ("<none>",)
            cand_probs = np.array([1.0], dtype=np.float32)
            chosen_id = 0
            chosen_token = "<none>"
            strategy = "greedy"

        s_probs, nf = create_tensor_snapshot(
            "candidate_probs",
            TensorRole.PROBS.value,
            cand_probs,
            ["vocab"],
            plan,
            labels_by_axis={0: cand_tokens},
            custom_indices_by_axis={0: cand_ids},
        )

        meta: dict[str, Any] = {
            "chosen_id": int(chosen_id),
            "chosen_token": chosen_token,
            "candidate_ids": [int(i) for i in cand_ids],
            "candidate_tokens": list(cand_tokens),
            "candidate_probs": [float(p) for p in cand_probs],
            "strategy": strategy,
        }
        if nf:
            meta["had_non_finite"] = True

        return Step(
            id=step_id,
            kind=StepKind.NEXT_TOKEN.value,
            semantic=Semantic.NEXT_TOKEN.value,
            stage=Stage.OUTPUT.value,
            block_index=None,
            detail_level=int(DetailLevel.STORY),
            module_path=None,
            tensors=(s_probs,),
            meta=meta,
            fidelity=Fidelity("exact"),
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
        semantic = kwargs.get("semantic", Semantic.LM_HEAD.value)
        if semantic == Semantic.LM_HEAD.value:
            return [self.build_lm_head(spec, capture, model, plan, ctx)]
        if semantic == Semantic.NEXT_TOKEN.value:
            return [self.build_next_token(spec, plan, ctx)]
        return []


out_builder = OutputStepBuilder()
STEP_BUILDERS.register(Semantic.LM_HEAD.value)(out_builder)
STEP_BUILDERS.register(Semantic.NEXT_TOKEN.value)(out_builder)
