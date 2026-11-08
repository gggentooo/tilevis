"""Attention step builder with head recomputation, causal masking, and fidelity grading.

Single responsibility: recompute multi-head attention scores, verify numerical fidelity,
detect n_heads divisors, and construct QKV_PROJECTION, ATTENTION_PATTERN, ATTENTION_MIX,
and LINEAR steps.
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


class AttentionStepBuilder:
    """Builds the 4 attention sub-steps with dynamic n_heads inference and fidelity grading."""

    def __init__(self) -> None:
        self.detected_n_heads: int | None = None

    def _find_best_n_heads(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_true: torch.Tensor,
        emb_dim: int,
        seq_len: int,
        provisional_heads: int | None,
    ) -> tuple[int, Fidelity, torch.Tensor, torch.Tensor]:
        """Test candidate divisors of emb_dim and select the matching head count."""
        scale = max(1.0, float(ctx_true.abs().max().item()))

        if seq_len == 1:
            h = (
                self.detected_n_heads
                or provisional_heads
                or (emb_dim // 64 if emb_dim >= 64 and emb_dim % 64 == 0 else 1)
            )
            logger.warning(
                "Single-token prompt: attention heads cannot be verified numerically; using h=%d",
                h,
            )
            head_dim = max(1, emb_dim // h)
            scores = (q @ k.t()) / (head_dim**0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_heads = attn_weights.unsqueeze(0).expand(h, -1, -1)
            fidelity = Fidelity(
                status="unverified",
                max_abs_err=0.0,
                note="Single token prompt: n_heads unverified",
            )
            return h, fidelity, attn_heads, attn_weights

        # Determine divisors to try
        divisors = [d for d in range(1, emb_dim + 1) if emb_dim % d == 0]
        pref_64 = emb_dim // 64
        candidate_order: list[int] = []
        if self.detected_n_heads and self.detected_n_heads in divisors:
            candidate_order.append(self.detected_n_heads)
        elif provisional_heads and provisional_heads in divisors:
            candidate_order.append(provisional_heads)

        if pref_64 in divisors and pref_64 not in candidate_order:
            candidate_order.append(pref_64)

        for d in divisors:
            if d not in candidate_order:
                candidate_order.append(d)

        best_h = candidate_order[0]
        best_err = float("inf")
        best_attn_heads = torch.empty(0)
        best_mean_attn = torch.empty(0)

        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device), diagonal=1).bool()

        for cand in candidate_order:
            cand_head_dim = emb_dim // cand
            try:
                q_h = q.view(seq_len, cand, cand_head_dim).transpose(0, 1)
                k_h = k.view(seq_len, cand, cand_head_dim).transpose(0, 1)
                v_h = v.view(seq_len, cand, cand_head_dim).transpose(0, 1)

                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (cand_head_dim**0.5)
                scores = scores.masked_fill(causal_mask, float("-inf"))
                weights = torch.softmax(scores, dim=-1)

                ctx_h = torch.matmul(weights, v_h)
                ctx_cand = ctx_h.transpose(0, 1).contiguous().view(seq_len, emb_dim)

                err = float((ctx_cand - ctx_true).abs().max().item())
                if err < best_err:
                    best_err = err
                    best_h = cand
                    best_attn_heads = weights
                    best_mean_attn = weights.mean(dim=0)

                if err <= 1e-4 * scale:
                    best_h = cand
                    best_err = err
                    best_attn_heads = weights
                    best_mean_attn = weights.mean(dim=0)
                    break
            except Exception:
                continue

        if best_err <= 1e-4 * scale:
            fidelity = Fidelity(status="exact", max_abs_err=best_err)
            self.detected_n_heads = best_h
        elif best_err <= 1e-2 * scale:
            fidelity = Fidelity(status="approximate", max_abs_err=best_err)
            self.detected_n_heads = best_h
        else:
            fidelity = Fidelity(
                status="failed",
                max_abs_err=best_err,
                note=f"Attention recompute mismatch (err={best_err:.4e})",
            )

        return best_h, fidelity, best_attn_heads, best_mean_attn

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
        if block is None or not block.attention:
            return []

        b_idx = block.index
        norm1_out = capture.get_required(block.norm1) if block.norm1 else None
        if norm1_out is None:
            norm1_out = capture.get_required(block.path, role="in")

        # Capture Q, K, V
        q = capture.get(block.q) if block.q else None
        k = capture.get(block.k) if block.k else None
        v = capture.get(block.v) if block.v else None

        # Context true: input of out_proj (or att output if out_proj missing)
        ctx_true = capture.get(block.out_proj, role="in") if block.out_proj else None
        if ctx_true is None and block.out_proj:
            ctx_true = capture.get(block.out_proj, role="out")

        att_out = capture.get(block.attention, role="out")
        if att_out is None and block.out_proj:
            att_out = capture.get(block.out_proj, role="out")

        # If essential components missing, fallback to generic
        if q is None or k is None or v is None or ctx_true is None:
            return self._build_generic_fallback(
                block, norm1_out, att_out, plan, ctx, note="Missing Q/K/V/ctx_true captures"
            )

        seq_len = q.shape[0]
        emb_dim = spec.emb_dim

        # Recompute and grade fidelity
        h, fidelity, attn_heads, mean_attn = self._find_best_n_heads(
            q=q,
            k=k,
            v=v,
            ctx_true=ctx_true,
            emb_dim=emb_dim,
            seq_len=seq_len,
            provisional_heads=spec.n_heads,
        )

        if fidelity.status == "failed":
            return self._build_generic_fallback(
                block, norm1_out, att_out, plan, ctx, note=fidelity.note
            )

        labels_map = {0: ctx.input_tokens}
        steps: list[Step] = []

        # 1. QKV_PROJECTION step
        qkv_id = ctx.next_step_id(f"b{b_idx}.att.qkv")
        qkv_tensors: list[TensorSnapshot] = []
        snap_in, _nf_in = create_tensor_snapshot(
            name=f"{block.norm1 or block.path}.output",
            role=TensorRole.INPUT.value,
            tensor=norm1_out,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        qkv_tensors.append(snap_in)

        # Extract weights W_q, W_k, W_v
        mod_q = model.get_submodule(block.q) if block.q else None
        mod_k = model.get_submodule(block.k) if block.k else None
        mod_v = model.get_submodule(block.v) if block.v else None

        w_q = _extract_weight_matrix(mod_q)
        w_k = _extract_weight_matrix(mod_k)
        w_v = _extract_weight_matrix(mod_v)

        if w_q is not None:
            s_wq, _ = create_tensor_snapshot(
                f"{block.q}.weight",
                TensorRole.WEIGHT.value,
                w_q,
                ["model_dim", "model_dim"],
                plan,
            )
            qkv_tensors.append(s_wq)
        if w_k is not None:
            s_wk, _ = create_tensor_snapshot(
                f"{block.k}.weight",
                TensorRole.WEIGHT.value,
                w_k,
                ["model_dim", "model_dim"],
                plan,
            )
            qkv_tensors.append(s_wk)
        if w_v is not None:
            s_wv, _ = create_tensor_snapshot(
                f"{block.v}.weight",
                TensorRole.WEIGHT.value,
                w_v,
                ["model_dim", "model_dim"],
                plan,
            )
            qkv_tensors.append(s_wv)

        s_q, _ = create_tensor_snapshot(
            f"{block.q}.output",
            TensorRole.QUERIES.value,
            q,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        s_k, _ = create_tensor_snapshot(
            f"{block.k}.output",
            TensorRole.KEYS.value,
            k,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        s_v, _ = create_tensor_snapshot(
            f"{block.v}.output",
            TensorRole.VALUES.value,
            v,
            ["token", "model_dim"],
            plan,
            labels_by_axis=labels_map,
        )
        qkv_tensors.extend([s_q, s_k, s_v])

        steps.append(
            Step(
                id=qkv_id,
                kind=StepKind.QKV_PROJECTION.value,
                semantic=Semantic.BLOCK_ATT_QKV.value,
                stage=Stage.BLOCK.value,
                block_index=b_idx,
                detail_level=int(DetailLevel.STANDARD),
                module_path=block.attention,
                tensors=tuple(qkv_tensors),
                meta={"n_heads": h, "emb_dim": emb_dim},
                fidelity=Fidelity("exact"),
            )
        )

        # 2. ATTENTION_PATTERN step
        scores_id = ctx.next_step_id(f"b{b_idx}.att.scores")
        snap_heads, _ = create_tensor_snapshot(
            name="attention_weights",
            role=TensorRole.ATTN_WEIGHTS.value,
            tensor=attn_heads,
            axis_names=["head", "token", "token"],
            plan=plan,
            labels_by_axis={1: ctx.input_tokens, 2: ctx.input_tokens},
        )
        snap_mean, _ = create_tensor_snapshot(
            name="mean_attention_weights",
            role=TensorRole.EXTRA.value,
            tensor=mean_attn,
            axis_names=["token", "token"],
            plan=plan,
            labels_by_axis={0: ctx.input_tokens, 1: ctx.input_tokens},
        )
        steps.append(
            Step(
                id=scores_id,
                kind=StepKind.ATTENTION_PATTERN.value,
                semantic=Semantic.BLOCK_ATT_SCORES.value,
                stage=Stage.BLOCK.value,
                block_index=b_idx,
                detail_level=int(DetailLevel.STANDARD),
                module_path=block.attention,
                tensors=(snap_heads, snap_mean),
                meta={"n_heads": h, "head_dim": emb_dim // h, "causal": True},
                fidelity=fidelity,
            )
        )

        # 3. ATTENTION_MIX step
        mix_id = ctx.next_step_id(f"b{b_idx}.att.mix")
        snap_mix_ctx, _ = create_tensor_snapshot(
            name="context_vectors",
            role=TensorRole.OUTPUT.value,
            tensor=ctx_true,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        steps.append(
            Step(
                id=mix_id,
                kind=StepKind.ATTENTION_MIX.value,
                semantic=Semantic.BLOCK_ATT_MIX.value,
                stage=Stage.BLOCK.value,
                block_index=b_idx,
                detail_level=int(DetailLevel.STANDARD),
                module_path=block.attention,
                tensors=(snap_mean, s_v, snap_mix_ctx),
                meta={"n_heads": h},
                fidelity=fidelity,
            )
        )

        # 4. LINEAR (out_proj) step
        out_id = ctx.next_step_id(f"b{b_idx}.att.out")
        out_tensors: list[TensorSnapshot] = [snap_mix_ctx]
        mod_out = model.get_submodule(block.out_proj) if block.out_proj else None
        w_out = _extract_weight_matrix(mod_out)
        if w_out is not None:
            snap_w_out, _ = create_tensor_snapshot(
                f"{block.out_proj}.weight",
                TensorRole.WEIGHT.value,
                w_out,
                ["model_dim", "model_dim"],
                plan,
            )
            out_tensors.append(snap_w_out)

        snap_att_out, _ = create_tensor_snapshot(
            name=f"{block.attention}.output",
            role=TensorRole.OUTPUT.value,
            tensor=att_out if att_out is not None else ctx_true,
            axis_names=["token", "model_dim"],
            plan=plan,
            labels_by_axis=labels_map,
        )
        out_tensors.append(snap_att_out)

        steps.append(
            Step(
                id=out_id,
                kind=StepKind.LINEAR.value,
                semantic=Semantic.BLOCK_ATT_OUT.value,
                stage=Stage.BLOCK.value,
                block_index=b_idx,
                detail_level=int(DetailLevel.STANDARD),
                module_path=block.out_proj or block.attention,
                tensors=tuple(out_tensors),
                meta={},
                fidelity=Fidelity("exact"),
            )
        )

        return steps

    def _build_generic_fallback(
        self,
        block: BlockSpec,
        norm1_out: torch.Tensor | None,
        att_out: torch.Tensor | None,
        plan: IndexPlan,
        ctx: StepBuildContext,
        note: str,
    ) -> list[Step]:
        """Emit a single black-box GENERIC step when attention recomputation fails."""
        step_id = ctx.next_step_id(f"b{block.index}.att.scores")
        tensors: list[TensorSnapshot] = []
        labels_map = {0: ctx.input_tokens}
        if norm1_out is not None:
            s_in, _ = create_tensor_snapshot(
                "attention.input",
                TensorRole.INPUT.value,
                norm1_out,
                ["token", "model_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            tensors.append(s_in)
        if att_out is not None:
            s_out, _ = create_tensor_snapshot(
                "attention.output",
                TensorRole.OUTPUT.value,
                att_out,
                ["token", "model_dim"],
                plan,
                labels_by_axis=labels_map,
            )
            tensors.append(s_out)

        return [
            Step(
                id=step_id,
                kind=StepKind.GENERIC.value,
                semantic=Semantic.BLOCK_ATT_SCORES.value,
                stage=Stage.BLOCK.value,
                block_index=block.index,
                detail_level=int(DetailLevel.STANDARD),
                module_path=block.attention,
                tensors=tuple(tensors),
                meta={"note": note},
                fidelity=Fidelity(status="failed", note=note),
            )
        ]


att_builder = AttentionStepBuilder()
STEP_BUILDERS.register(Semantic.BLOCK_ATT_QKV.value)(att_builder)
STEP_BUILDERS.register(Semantic.BLOCK_ATT_SCORES.value)(att_builder)
STEP_BUILDERS.register(Semantic.BLOCK_ATT_MIX.value)(att_builder)
STEP_BUILDERS.register(Semantic.BLOCK_ATT_OUT.value)(att_builder)
