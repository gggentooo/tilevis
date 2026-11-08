"""Embedding, tokenization, and prompt step builders.

Single responsibility: generate PROMPT, TOKENIZE, and EMBEDDING steps.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, Step
from tilevis.probe.adapters.base import ModelSpec
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan, create_tensor_snapshot

logger = logging.getLogger(__name__)


class PromptStepBuilder:
    """Builds the initial PROMPT step."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        step_id = ctx.next_step_id("prompt")
        meta = {
            "text": ctx.text_before,
            "n_tokens": len(ctx.input_tokens),
            "pass_index": ctx.pass_index,
        }
        return [
            Step(
                id=step_id,
                kind=StepKind.PROMPT.value,
                semantic=Semantic.PROMPT.value,
                stage=Stage.INPUT.value,
                block_index=None,
                detail_level=int(DetailLevel.STORY),
                module_path=None,
                tensors=(),
                meta=meta,
                fidelity=Fidelity("exact"),
            )
        ]


class TokenizeStepBuilder:
    """Builds the TOKENIZE step representing text decomposition into token IDs."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        step_id = ctx.next_step_id("tokenize")
        token_ids_arr = np.array(ctx.input_ids, dtype=np.float32)

        snap, had_non_finite = create_tensor_snapshot(
            name="token_ids",
            role=TensorRole.INPUT.value,
            tensor=token_ids_arr,
            axis_names=["token"],
            plan=plan,
            labels_by_axis={0: ctx.input_tokens},
        )

        meta: dict[str, Any] = {
            "tokens": list(ctx.input_tokens),
            "token_ids": list(ctx.input_ids),
        }
        if had_non_finite:
            meta["had_non_finite"] = True

        return [
            Step(
                id=step_id,
                kind=StepKind.TOKENIZE.value,
                semantic=Semantic.TOKENIZE.value,
                stage=Stage.INPUT.value,
                block_index=None,
                detail_level=int(DetailLevel.STORY),
                module_path=None,
                tensors=(snap,),
                meta=meta,
                fidelity=Fidelity("exact"),
            )
        ]


class EmbeddingStepBuilder:
    """Builds the EMBEDDING step combining token vectors and positional encodings."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        step_id = ctx.next_step_id("embed")
        seq_len = len(ctx.input_ids)

        tok_vecs = capture.get(spec.token_embedding)
        if tok_vecs is None:
            # Recompute directly using model submodule
            submod = model.get_submodule(spec.token_embedding)
            with torch.inference_mode():
                device = next(submod.parameters()).device
                ids_t = torch.tensor([ctx.input_ids], device=device)
                tok_vecs = submod(ids_t)[0].detach().cpu().to(torch.float32)

        if spec.position_embedding and spec.position_embedding in model._modules:
            pos_vecs = capture.get(spec.position_embedding)
            if pos_vecs is None:
                submod = model.get_submodule(spec.position_embedding)
                with torch.inference_mode():
                    device = next(submod.parameters()).device
                    pos_ids = torch.arange(seq_len, device=device)
                    pos_vecs = submod(pos_ids).detach().cpu().to(torch.float32)
        else:
            pos_vecs = torch.zeros((seq_len, spec.emb_dim), dtype=torch.float32)

        # Normalize to 2-D (seq_len, emb_dim)
        if tok_vecs.ndim == 3 and tok_vecs.shape[0] == 1:
            tok_vecs = tok_vecs[0]
        elif tok_vecs.ndim == 1:
            tok_vecs = tok_vecs.unsqueeze(0)

        if pos_vecs.ndim == 3 and pos_vecs.shape[0] == 1:
            pos_vecs = pos_vecs[0]
        elif pos_vecs.ndim == 1:
            pos_vecs = pos_vecs.unsqueeze(0)

        embedded_input = tok_vecs + pos_vecs

        labels_map = {0: ctx.input_tokens}
        snap_tok, nf1 = create_tensor_snapshot(
            name=f"{spec.token_embedding}.output",
            role=TensorRole.TOKEN_VECTORS.value,
            tensor=tok_vecs,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        snap_pos, nf2 = create_tensor_snapshot(
            name=f"{spec.position_embedding or 'pos_emb'}.output",
            role=TensorRole.POSITION_VECTORS.value,
            tensor=pos_vecs,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        snap_out, nf3 = create_tensor_snapshot(
            name="embedded_input",
            role=TensorRole.OUTPUT.value,
            tensor=embedded_input,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )

        meta: dict[str, Any] = {
            "emb_dim": spec.emb_dim,
            "has_pos_emb": spec.position_embedding is not None,
        }
        if nf1 or nf2 or nf3:
            meta["had_non_finite"] = True

        return [
            Step(
                id=step_id,
                kind=StepKind.EMBEDDING.value,
                semantic=Semantic.EMBED.value,
                stage=Stage.INPUT.value,
                block_index=None,
                detail_level=int(DetailLevel.STORY),
                module_path=spec.token_embedding,
                tensors=(snap_tok, snap_pos, snap_out),
                meta=meta,
                fidelity=Fidelity("exact"),
            )
        ]


# Register builders
STEP_BUILDERS.register(Semantic.PROMPT.value)(PromptStepBuilder())
STEP_BUILDERS.register(Semantic.TOKENIZE.value)(TokenizeStepBuilder())
STEP_BUILDERS.register(Semantic.EMBED.value)(EmbeddingStepBuilder())
