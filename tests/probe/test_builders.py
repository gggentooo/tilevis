"""Unit tests for probe step builders.

Single responsibility: test step construction, tensor snapshot assignment, attention recomputation,
fidelity grading, and residual verification across all builders.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import build_matrix_model

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Prediction
from tilevis.probe.adapters import select_adapter
from tilevis.probe.adapters.base import ModelSpec
from tilevis.probe.builders.attention import AttentionStepBuilder
from tilevis.probe.builders.base import StepBuildContext
from tilevis.probe.builders.block_summary import BlockSummaryStepBuilder
from tilevis.probe.builders.embedding import (
    EmbeddingStepBuilder,
    PromptStepBuilder,
    TokenizeStepBuilder,
)
from tilevis.probe.builders.layernorm import LayerNormStepBuilder
from tilevis.probe.builders.output import OutputStepBuilder
from tilevis.probe.builders.residual import ResidualStepBuilder
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan
from tilevis.probe.text.tokenizer import TiktokenTokenizer


@pytest.fixture
def tiny_model_and_spec() -> tuple[nn.Module, ModelSpec, IndexPlan, TiktokenTokenizer]:
    """Create a small_bias model with ModelSpec and IndexPlan."""
    entry = COMPAT_MATRIX["small_bias"]
    model = build_matrix_model(entry, seed=42)
    _adapter, spec, _ = select_adapter(model)
    plan = IndexPlan.from_model_spec(spec, max_features=32, max_tokens=16)
    tokenizer = TiktokenTokenizer("gpt2")
    return model, spec, plan, tokenizer


def test_prompt_and_tokenize_builders(tiny_model_and_spec) -> None:
    """Test PROMPT and TOKENIZE step builders."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995),
        input_tokens=("Hello", "·world"),
        text_before="Hello world",
        tokenizer=tokenizer,
    )
    buffer = CaptureBuffer()

    p_builder = PromptStepBuilder()
    prompt_steps = p_builder.build(spec, buffer, model, plan, ctx)
    assert len(prompt_steps) == 1
    assert prompt_steps[0].kind == StepKind.PROMPT.value
    assert prompt_steps[0].semantic == Semantic.PROMPT.value
    assert prompt_steps[0].meta["n_tokens"] == 2

    t_builder = TokenizeStepBuilder()
    tok_steps = t_builder.build(spec, buffer, model, plan, ctx)
    assert len(tok_steps) == 1
    assert tok_steps[0].kind == StepKind.TOKENIZE.value
    assert len(tok_steps[0].tensors) == 1
    assert tok_steps[0].tensors[0].role == TensorRole.INPUT.value


def test_embedding_builder(tiny_model_and_spec) -> None:
    """Test EMBEDDING builder with token vectors, position vectors, and sum."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995),
        input_tokens=("Hello", "·world"),
        text_before="Hello world",
        tokenizer=tokenizer,
    )
    buffer = CaptureBuffer()

    builder = EmbeddingStepBuilder()
    steps = builder.build(spec, buffer, model, plan, ctx)
    assert len(steps) == 1
    step = steps[0]
    assert step.kind == StepKind.EMBEDDING.value
    assert step.semantic == Semantic.EMBED.value
    assert len(step.tensors) == 3

    roles = [t.role for t in step.tensors]
    assert TensorRole.TOKEN_VECTORS.value in roles
    assert TensorRole.POSITION_VECTORS.value in roles
    assert TensorRole.OUTPUT.value in roles


def test_layernorm_builder(tiny_model_and_spec) -> None:
    """Test LayerNorm builder and numerical fidelity verification."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995),
        input_tokens=("Hello", "·world"),
        text_before="Hello world",
        tokenizer=tokenizer,
    )
    buffer = CaptureBuffer()

    # Synthetic input and output conforming to layernorm
    block = spec.blocks[0]
    norm_mod = model.get_submodule(block.norm1)
    inp = torch.randn(2, spec.emb_dim)
    with torch.inference_mode():
        out = norm_mod(inp)

    buffer.store(block.path, 0, "in", inp)
    buffer.store(block.norm1, 0, "out", out)

    builder = LayerNormStepBuilder()
    steps = builder.build(
        spec, buffer, model, plan, ctx, block=block, semantic=Semantic.BLOCK_NORM1.value
    )
    assert len(steps) == 1
    step = steps[0]
    assert step.kind == StepKind.LAYERNORM.value
    assert step.fidelity.status == "exact"
    assert step.fidelity.max_abs_err is not None
    assert step.fidelity.max_abs_err < 1e-4


def test_attention_builder_recomputation_and_fidelity(tiny_model_and_spec) -> None:
    """Test AttentionStepBuilder multi-head recomputation, fidelity, and step emission."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995, 1234),
        input_tokens=("Hello", "·world", "·test"),
        text_before="Hello world test",
        tokenizer=tokenizer,
    )
    buffer = CaptureBuffer()
    block = spec.blocks[0]
    att_mod = model.get_submodule(block.attention)

    # Run real forward through attention submodules
    seq_len = 3
    x = torch.randn(seq_len, spec.emb_dim)
    buffer.store(block.norm1, 0, "out", x)

    mod_q = model.get_submodule(block.q)
    mod_k = model.get_submodule(block.k)
    mod_v = model.get_submodule(block.v)
    mod_out = model.get_submodule(block.out_proj)

    with torch.inference_mode():
        q = mod_q(x)
        k = mod_k(x)
        v = mod_v(x)
        # Real attention context calculation from previous_chapters
        att_out = att_mod(x.unsqueeze(0))[0]

    buffer.store(block.q, 0, "out", q)
    buffer.store(block.k, 0, "out", k)
    buffer.store(block.v, 0, "out", v)
    buffer.store(block.attention, 0, "out", att_out)

    # To get ctx_true (input to out_proj)
    # in MultiHeadAttention: context = context.view(1, seq_len, emb_dim) -> out_proj(context)
    # We can solve ctx_true by passing x to attention
    # Let's inspect out_proj input
    with torch.inference_mode():
        # Pre-hook on out_proj captures ctx_true
        def hook_fn(m, args):
            buffer.store(block.out_proj, 0, "in", args[0])

        handle = mod_out.register_forward_pre_hook(hook_fn)
        _ = att_mod(x.unsqueeze(0))
        handle.remove()

    builder = AttentionStepBuilder()
    steps = builder.build(spec, buffer, model, plan, ctx, block=block)

    # Must emit 4 steps
    assert len(steps) == 4
    kinds = [s.kind for s in steps]
    assert kinds == [
        StepKind.QKV_PROJECTION.value,
        StepKind.ATTENTION_PATTERN.value,
        StepKind.ATTENTION_MIX.value,
        StepKind.LINEAR.value,
    ]

    # Attention pattern fidelity must be exact
    att_step = steps[1]
    assert att_step.fidelity.status == "exact"
    assert att_step.fidelity.max_abs_err is not None
    assert att_step.fidelity.max_abs_err < 1e-4

    # Detected n_heads must match spec.n_heads (4)
    assert builder.detected_n_heads == spec.n_heads


def test_residual_builder_identity(tiny_model_and_spec) -> None:
    """Test ResidualStepBuilder verifies shortcut + update == out."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995),
        input_tokens=("Hello", "·world"),
        text_before="Hello world",
        tokenizer=tokenizer,
    )
    buffer = CaptureBuffer()
    block = spec.blocks[0]

    sc = torch.randn(2, spec.emb_dim)
    up = torch.randn(2, spec.emb_dim)
    out = sc + up

    buffer.store(block.path, 0, "in", sc)
    buffer.store(block.attention, 0, "out", up)
    buffer.store(block.norm2, 0, "in", out)

    builder = ResidualStepBuilder()
    steps = builder.build(
        spec, buffer, model, plan, ctx, block=block, semantic=Semantic.BLOCK_RESID1.value
    )
    assert len(steps) == 1
    assert steps[0].kind == StepKind.RESIDUAL_ADD.value
    assert steps[0].fidelity.status == "exact"
    assert steps[0].fidelity.max_abs_err is not None
    assert steps[0].fidelity.max_abs_err < 1e-5


def test_block_summary_builder(tiny_model_and_spec) -> None:
    """Test BlockSummaryStepBuilder input, output, and update change."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995),
        input_tokens=("Hello", "·world"),
        text_before="Hello world",
        tokenizer=tokenizer,
    )
    buffer = CaptureBuffer()
    block = spec.blocks[0]

    inp = torch.randn(2, spec.emb_dim)
    out = inp + torch.randn(2, spec.emb_dim)
    buffer.store(block.path, 0, "in", inp)
    buffer.store(block.path, 0, "out", out)

    builder = BlockSummaryStepBuilder()
    steps = builder.build(spec, buffer, model, plan, ctx, block=block)
    assert len(steps) == 1
    assert steps[0].kind == StepKind.BLOCK_SUMMARY.value
    assert steps[0].detail_level == int(DetailLevel.STORY)
    assert len(steps[0].tensors) == 3


def test_output_step_builder(tiny_model_and_spec) -> None:
    """Test LM_HEAD and NEXT_TOKEN output builders."""
    model, spec, plan, tokenizer = tiny_model_and_spec
    pred = Prediction(
        candidate_ids=(100, 200, 300),
        candidate_tokens=("·apple", "·banana", "·cherry"),
        candidate_probs=(0.6, 0.3, 0.1),
        chosen_id=100,
        chosen_token="·apple",
        strategy="greedy",
    )
    ctx = StepBuildContext(
        pass_index=0,
        input_ids=(15496, 995),
        input_tokens=("Hello", "·world"),
        text_before="Hello world",
        tokenizer=tokenizer,
        prediction=pred,
    )
    buffer = CaptureBuffer()

    fn_out = torch.randn(2, spec.emb_dim)
    buffer.store(spec.final_norm, 0, "out", fn_out)

    builder = OutputStepBuilder()
    lm_steps = builder.build(spec, buffer, model, plan, ctx, semantic=Semantic.LM_HEAD.value)
    assert len(lm_steps) == 1
    assert lm_steps[0].kind == StepKind.LOGITS.value
    assert lm_steps[0].stage == Stage.OUTPUT.value

    # Next token step
    tok_steps = builder.build(spec, buffer, model, plan, ctx, semantic=Semantic.NEXT_TOKEN.value)
    assert len(tok_steps) == 1
    assert tok_steps[0].kind == StepKind.NEXT_TOKEN.value
    assert tok_steps[0].meta["chosen_id"] == 100
    assert tok_steps[0].detail_level == int(DetailLevel.STORY)
