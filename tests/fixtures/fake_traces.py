"""Torch-free synthetic Trace generator for unit, render, and integration tests.

Produces valid, reproducible Trace objects matching any architecture shape.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import (
    TRACE_SCHEMA_VERSION,
    AxisInfo,
    Fidelity,
    ModelSpecInfo,
    PassRecord,
    Prediction,
    Step,
    TensorSnapshot,
    TensorStats,
    Trace,
)


def _compute_stats(arr: np.ndarray) -> TensorStats:
    """Compute TensorStats on an array."""
    data = arr.astype(np.float32)
    abs_data = np.abs(data)
    return TensorStats(
        min=float(np.min(data)),
        max=float(np.max(data)),
        mean=float(np.mean(data)),
        std=float(np.std(data)),
        abs_max=float(np.max(abs_data)),
        abs_p99=float(np.percentile(abs_data, 99)),
    )


def make_snapshot(
    name: str,
    role: str | TensorRole,
    data: np.ndarray,
    axis_names: tuple[str, ...],
    full_shape: tuple[int, ...] | None = None,
    reduction: str = "none",
    token_labels: tuple[str, ...] | None = None,
) -> TensorSnapshot:
    """Helper to create a valid TensorSnapshot with AxisInfo descriptors."""
    role_str = role.value if isinstance(role, TensorRole) else str(role)
    data = np.asarray(data, dtype=np.float32)
    if full_shape is None:
        full_shape = data.shape

    axes: list[AxisInfo] = []
    for dim_idx, ax_name in enumerate(axis_names):
        dim_size = data.shape[dim_idx]
        full_size = full_shape[dim_idx]
        indices = np.arange(dim_size, dtype=np.int64)
        labels = token_labels if ax_name == "token" else None
        axes.append(
            AxisInfo(
                name=ax_name,
                full_size=full_size,
                indices=indices,
                labels=labels,
            )
        )

    return TensorSnapshot(
        name=name,
        role=role_str,
        data=data,
        axes=tuple(axes),
        full_shape=full_shape,
        reduction=reduction,
        stats=_compute_stats(data),
    )


def make_fake_trace(
    n_layers: int = 2,
    emb_dim: int = 16,
    n_tokens: int = 5,
    n_passes: int = 2,
    n_heads: int = 2,
    ff_mult: int = 4,
    include_kinds: Sequence[StepKind | str] | None = None,
    seed: int = 0,
) -> Trace:
    """Generate a valid, fully populated Trace object without PyTorch.

    Args:
        n_layers: Number of transformer blocks.
        emb_dim: Embedding dimension.
        n_tokens: Starting context token count.
        n_passes: Number of autoregressive token passes.
        n_heads: Number of attention heads.
        ff_mult: Feed-forward multiplier (ff_dim = ff_mult * emb_dim).
        include_kinds: Optional subset of StepKinds to include (defaults to ALL).
        seed: Random seed for deterministic reproducibility.

    Returns:
        A completely valid Trace instance meeting all schema invariants.
    """
    rng = np.random.default_rng(seed)
    ff_dim = ff_mult * emb_dim
    vocab_size = 50257
    head_dim = max(1, emb_dim // n_heads)

    filter_kinds = set(include_kinds) if include_kinds is not None else None

    def should_include(kind: StepKind) -> bool:
        if filter_kinds is None:
            return True
        return kind in filter_kinds or str(kind) in filter_kinds

    prompt_text = "Every effort moves you forward"
    base_tokens = ["Every", "·effort", "·moves", "·you", "·forward"]
    gen_tokens = ["·toward", "·great", "·success", "·today", "·always"]

    passes: list[PassRecord] = []
    current_tokens = list(base_tokens[:n_tokens])
    current_ids = list(range(100, 100 + len(current_tokens)))

    for p_idx in range(n_passes):
        t_count = len(current_tokens)
        token_labels = tuple(current_tokens)
        steps: list[Step] = []

        # 1. Prompt step
        if should_include(StepKind.PROMPT):
            steps.append(
                Step(
                    id=f"p{p_idx}.prompt",
                    kind=StepKind.PROMPT,
                    semantic=Semantic.PROMPT,
                    stage=Stage.INPUT,
                    block_index=None,
                    detail_level=DetailLevel.STORY,
                    module_path=None,
                    meta={"prompt": prompt_text, "pass": p_idx},
                )
            )

        # 2. Tokenize step
        if should_include(StepKind.TOKENIZE):
            steps.append(
                Step(
                    id=f"p{p_idx}.tokenize",
                    kind=StepKind.TOKENIZE,
                    semantic=Semantic.TOKENIZE,
                    stage=Stage.INPUT,
                    block_index=None,
                    detail_level=DetailLevel.STORY,
                    module_path=None,
                    meta={"token_ids": list(current_ids), "tokens": list(current_tokens)},
                )
            )

        # 3. Embedding step
        if should_include(StepKind.EMBEDDING):
            tok_vecs = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
            pos_vecs = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
            combined = tok_vecs + pos_vecs
            steps.append(
                Step(
                    id=f"p{p_idx}.embed",
                    kind=StepKind.EMBEDDING,
                    semantic=Semantic.EMBED,
                    stage=Stage.INPUT,
                    block_index=None,
                    detail_level=DetailLevel.STORY,
                    module_path="tok_emb",
                    tensors=(
                        make_snapshot(
                            "token_vectors",
                            TensorRole.TOKEN_VECTORS,
                            tok_vecs,
                            ("token", "model_dim"),
                            token_labels=token_labels,
                        ),
                        make_snapshot(
                            "position_vectors",
                            TensorRole.POSITION_VECTORS,
                            pos_vecs,
                            ("token", "model_dim"),
                            token_labels=token_labels,
                        ),
                        make_snapshot(
                            "embedded_input",
                            TensorRole.OUTPUT,
                            combined,
                            ("token", "model_dim"),
                            token_labels=token_labels,
                        ),
                    ),
                    fidelity=Fidelity("exact", 0.0),
                )
            )

        # 4. Transformer Blocks
        for b in range(n_layers):
            block_in = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
            block_out = block_in + rng.standard_normal((t_count, emb_dim), dtype=np.float32) * 0.1

            # Block Summary (Story mode)
            if should_include(StepKind.BLOCK_SUMMARY):
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.summary",
                        kind=StepKind.BLOCK_SUMMARY,
                        semantic=Semantic.BLOCK_SUMMARY,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STORY,
                        module_path=f"trf_blocks.{b}",
                        tensors=(
                            make_snapshot(
                                f"b{b}.input",
                                TensorRole.INPUT,
                                block_in,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.output",
                                TensorRole.OUTPUT,
                                block_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.change",
                                TensorRole.UPDATE,
                                block_out - block_in,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # Norm1
            if should_include(StepKind.LAYERNORM):
                norm1_out = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                scale = np.ones((emb_dim,), dtype=np.float32)
                shift = np.zeros((emb_dim,), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.norm1",
                        kind=StepKind.LAYERNORM,
                        semantic=Semantic.BLOCK_NORM1,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.norm1",
                        tensors=(
                            make_snapshot(
                                f"b{b}.norm1.in",
                                TensorRole.INPUT,
                                block_in,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.norm1.scale", TensorRole.SCALE, scale, ("model_dim",)
                            ),
                            make_snapshot(
                                f"b{b}.norm1.shift", TensorRole.SHIFT, shift, ("model_dim",)
                            ),
                            make_snapshot(
                                f"b{b}.norm1.out",
                                TensorRole.OUTPUT,
                                norm1_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # QKV projection
            if should_include(StepKind.QKV_PROJECTION):
                q = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                k = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                v = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.att.qkv",
                        kind=StepKind.QKV_PROJECTION,
                        semantic=Semantic.BLOCK_ATT_QKV,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.att",
                        tensors=(
                            make_snapshot(
                                f"b{b}.q",
                                TensorRole.QUERIES,
                                q,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.k",
                                TensorRole.KEYS,
                                k,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.v",
                                TensorRole.VALUES,
                                v,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # Attention Pattern
            if should_include(StepKind.ATTENTION_PATTERN):
                raw_attn = rng.uniform(0.0, 1.0, size=(n_heads, t_count, t_count)).astype(
                    np.float32
                )
                # Lower triangular causal masking
                for h in range(n_heads):
                    for r in range(t_count):
                        raw_attn[h, r, r + 1 :] = 0.0
                        row_sum = np.sum(raw_attn[h, r, : r + 1])
                        if row_sum > 0:
                            raw_attn[h, r, : r + 1] /= row_sum
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.att.scores",
                        kind=StepKind.ATTENTION_PATTERN,
                        semantic=Semantic.BLOCK_ATT_SCORES,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.att",
                        tensors=(
                            make_snapshot(
                                f"b{b}.attn_weights",
                                TensorRole.ATTN_WEIGHTS,
                                raw_attn,
                                ("head", "token", "token"),
                                token_labels=token_labels,
                            ),
                        ),
                        meta={"n_heads": n_heads, "head_dim": head_dim},
                        fidelity=Fidelity("exact", 1e-6),
                    )
                )

            # Attention Mix
            if should_include(StepKind.ATTENTION_MIX):
                ctx_vec = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.att.mix",
                        kind=StepKind.ATTENTION_MIX,
                        semantic=Semantic.BLOCK_ATT_MIX,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.att",
                        tensors=(
                            make_snapshot(
                                f"b{b}.context",
                                TensorRole.OUTPUT,
                                ctx_vec,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # Attention out projection (LINEAR)
            if should_include(StepKind.LINEAR):
                w_out = rng.standard_normal((emb_dim, emb_dim), dtype=np.float32)
                att_out = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.att.out",
                        kind=StepKind.LINEAR,
                        semantic=Semantic.BLOCK_ATT_OUT,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.att.out_proj",
                        tensors=(
                            make_snapshot(
                                f"b{b}.w_out", TensorRole.WEIGHT, w_out, ("model_dim", "model_dim")
                            ),
                            make_snapshot(
                                f"b{b}.att_out",
                                TensorRole.OUTPUT,
                                att_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # Residual 1
            if should_include(StepKind.RESIDUAL_ADD):
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.resid1",
                        kind=StepKind.RESIDUAL_ADD,
                        semantic=Semantic.BLOCK_RESID1,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}",
                        tensors=(
                            make_snapshot(
                                f"b{b}.shortcut1",
                                TensorRole.SHORTCUT,
                                block_in,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.update1",
                                TensorRole.UPDATE,
                                block_out - block_in,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.resid1_out",
                                TensorRole.OUTPUT,
                                block_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # FeedForward steps
            if should_include(StepKind.LINEAR):
                w_ff1 = rng.standard_normal((emb_dim, ff_dim), dtype=np.float32)
                ff_mid = rng.standard_normal((t_count, ff_dim), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.ff.expand",
                        kind=StepKind.LINEAR,
                        semantic=Semantic.BLOCK_FF_EXPAND,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.ff.layers.0",
                        tensors=(
                            make_snapshot(
                                f"b{b}.w_ff1", TensorRole.WEIGHT, w_ff1, ("model_dim", "ff_dim")
                            ),
                            make_snapshot(
                                f"b{b}.ff_mid",
                                TensorRole.OUTPUT,
                                ff_mid,
                                ("token", "ff_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # Activation
            if should_include(StepKind.ACTIVATION):
                act_out = rng.standard_normal((t_count, ff_dim), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.ff.act",
                        kind=StepKind.ACTIVATION,
                        semantic=Semantic.BLOCK_FF_ACT,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.ff.layers.1",
                        tensors=(
                            make_snapshot(
                                f"b{b}.act_out",
                                TensorRole.OUTPUT,
                                act_out,
                                ("token", "ff_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            # FF contract & resid2
            if should_include(StepKind.LINEAR):
                w_ff2 = rng.standard_normal((ff_dim, emb_dim), dtype=np.float32)
                ff_out = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.ff.contract",
                        kind=StepKind.LINEAR,
                        semantic=Semantic.BLOCK_FF_CONTRACT,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}.ff.layers.2",
                        tensors=(
                            make_snapshot(
                                f"b{b}.w_ff2", TensorRole.WEIGHT, w_ff2, ("ff_dim", "model_dim")
                            ),
                            make_snapshot(
                                f"b{b}.ff_out",
                                TensorRole.OUTPUT,
                                ff_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

            if should_include(StepKind.RESIDUAL_ADD):
                steps.append(
                    Step(
                        id=f"p{p_idx}.b{b}.resid2",
                        kind=StepKind.RESIDUAL_ADD,
                        semantic=Semantic.BLOCK_RESID2,
                        stage=Stage.BLOCK,
                        block_index=b,
                        detail_level=DetailLevel.STANDARD,
                        module_path=f"trf_blocks.{b}",
                        tensors=(
                            make_snapshot(
                                f"b{b}.shortcut2",
                                TensorRole.SHORTCUT,
                                block_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.update2",
                                TensorRole.UPDATE,
                                block_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                            make_snapshot(
                                f"b{b}.resid2_out",
                                TensorRole.OUTPUT,
                                block_out,
                                ("token", "model_dim"),
                                token_labels=token_labels,
                            ),
                        ),
                        fidelity=Fidelity("exact", 0.0),
                    )
                )

        # 5. Final LayerNorm
        if should_include(StepKind.LAYERNORM):
            final_norm_out = rng.standard_normal((t_count, emb_dim), dtype=np.float32)
            steps.append(
                Step(
                    id=f"p{p_idx}.final_norm",
                    kind=StepKind.LAYERNORM,
                    semantic=Semantic.FINAL_NORM,
                    stage=Stage.OUTPUT,
                    block_index=None,
                    detail_level=DetailLevel.STANDARD,
                    module_path="final_norm",
                    tensors=(
                        make_snapshot(
                            "final_norm_out",
                            TensorRole.OUTPUT,
                            final_norm_out,
                            ("token", "model_dim"),
                            token_labels=token_labels,
                        ),
                    ),
                    fidelity=Fidelity("exact", 0.0),
                )
            )

        # 6. LM Head (LOGITS)
        if should_include(StepKind.LOGITS):
            last_logits = rng.standard_normal((10,), dtype=np.float32)
            steps.append(
                Step(
                    id=f"p{p_idx}.lm_head",
                    kind=StepKind.LOGITS,
                    semantic=Semantic.LM_HEAD,
                    stage=Stage.OUTPUT,
                    block_index=None,
                    detail_level=DetailLevel.STANDARD,
                    module_path="out_head",
                    tensors=(make_snapshot("logits", TensorRole.LOGITS, last_logits, ("vocab",)),),
                    fidelity=Fidelity("exact", 0.0),
                )
            )

        # 7. Next Token Prediction
        candidate_ids = tuple(range(1000, 1010))
        next_word = gen_tokens[p_idx % len(gen_tokens)]
        candidate_tokens = (
            next_word,
            "·other",
            "·path",
            "·road",
            "·way",
            "·goal",
            "·light",
            "·star",
            "·hope",
            "·day",
        )
        # Ensure descending probabilities
        raw_probs = np.sort(rng.uniform(0.01, 0.99, size=10))[::-1]
        raw_probs /= np.sum(raw_probs)
        candidate_probs = tuple(float(p) for p in raw_probs)

        prediction = Prediction(
            candidate_ids=candidate_ids,
            candidate_tokens=candidate_tokens,
            candidate_probs=candidate_probs,
            chosen_id=candidate_ids[0],
            chosen_token=next_word,
            strategy="greedy",
        )

        if should_include(StepKind.NEXT_TOKEN):
            steps.append(
                Step(
                    id=f"p{p_idx}.next_token",
                    kind=StepKind.NEXT_TOKEN,
                    semantic=Semantic.NEXT_TOKEN,
                    stage=Stage.OUTPUT,
                    block_index=None,
                    detail_level=DetailLevel.STORY,
                    module_path=None,
                    tensors=(
                        make_snapshot(
                            "probs",
                            TensorRole.PROBS,
                            np.array(candidate_probs, dtype=np.float32),
                            ("vocab",),
                        ),
                    ),
                    meta={"chosen_token": next_word, "chosen_id": candidate_ids[0]},
                    fidelity=Fidelity("exact", 0.0),
                )
            )

        pass_rec = PassRecord(
            index=p_idx,
            input_ids=tuple(current_ids),
            input_tokens=tuple(current_tokens),
            text_before=" ".join(current_tokens),
            steps=tuple(steps),
            prediction=prediction,
        )
        passes.append(pass_rec)

        # Append predicted token for the next pass
        current_tokens.append(next_word)
        current_ids.append(candidate_ids[0])

    model_info = ModelSpecInfo(
        name=f"FakeGPT-{n_layers}L-{emb_dim}D",
        adapter="gpt_scratch",
        n_layers=n_layers,
        emb_dim=emb_dim,
        n_heads=n_heads,
        vocab_size=vocab_size,
        context_length=1024,
        n_params=n_layers * emb_dim * emb_dim * 12,
        extras={"ff_mult": ff_mult},
    )

    return Trace(
        schema_version=TRACE_SCHEMA_VERSION,
        tilevis_version="0.2.0",
        created_at="2026-11-08T12:00:00Z",
        model=model_info,
        prompt=prompt_text,
        settings={"n_tokens": n_tokens, "seed": seed},
        passes=tuple(passes),
    )
