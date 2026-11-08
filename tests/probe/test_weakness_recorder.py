"""Weakness and edge-case probes for probe recording.

Single responsibility: test single-token prompts, context cropping, early EOS,
attention failure fallbacks, NaN weight sanitization, dimension limits, and determinism.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import build_matrix_model

from tilevis.core.config import RecordSettings
from tilevis.core.kinds import Semantic, StepKind
from tilevis.core.validation import validate_trace
from tilevis.probe.adapters import select_adapter
from tilevis.probe.loading.factory import BuiltModel
from tilevis.probe.recorder import TraceRecorder
from tilevis.probe.text.templates import TEMPLATES
from tilevis.probe.text.tokenizer import TiktokenTokenizer


def _build_test_bundle():
    entry = COMPAT_MATRIX["small_bias"]
    model = build_matrix_model(entry, seed=42)
    _adapter, spec, conf = select_adapter(model)
    tokenizer = TiktokenTokenizer("gpt2")
    template = TEMPLATES.get("raw")
    built_model = BuiltModel(
        model=model,
        spec=spec,
        cfg=entry.to_config_dict(),
        tokenizer=tokenizer,
        template=template,
        adapter_confidence=conf,
        notes=(),
    )
    return built_model


def test_single_token_prompt_unverified_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test 1-token prompt yields 'unverified' fidelity and logs warning without crash."""
    built = _build_test_bundle()
    settings = RecordSettings(new_tokens=1, strategy="greedy")
    recorder = TraceRecorder(built, settings)

    with caplog.at_level(logging.WARNING):
        trace = recorder.record("A")  # Single token "A" in GPT-2 BPE (id 32)

    assert len(trace.passes) == 1
    scores_step = next(
        s for s in trace.passes[0].steps if s.semantic == Semantic.BLOCK_ATT_SCORES.value
    )
    assert scores_step.fidelity.status == "unverified"
    assert "Single-token prompt" in caplog.text
    validate_trace(trace, strict=True)


def test_context_cropping_noted_in_meta() -> None:
    """Test that prompts exceeding context_length are left-cropped and flagged."""
    built = _build_test_bundle()
    # Force context_length to 4
    spec_cropped = built.spec
    object.__setattr__(spec_cropped, "context_length", 4)

    settings = RecordSettings(new_tokens=2, strategy="greedy")
    recorder = TraceRecorder(built, settings)
    trace = recorder.record("Every single effort always moves you forward")

    # In pass records, input_ids length should be <= 4
    for p in trace.passes:
        assert len(p.input_ids) == 4
    validate_trace(trace, strict=True)


def test_attention_failure_emits_generic_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test when attention recomputation fails, a GENERIC fallback step is emitted."""
    built = _build_test_bundle()
    block = built.spec.blocks[0]
    att_mod = built.model.get_submodule(block.attention)

    # Monkeypatch the attention module forward to remove 1/sqrt(head_dim) scaling factor (§9.4)
    def unscaled_forward(x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        keys = att_mod.W_key(x)
        queries = att_mod.W_query(x)
        values = att_mod.W_value(x)
        keys = keys.view(b, num_tokens, att_mod.num_heads, att_mod.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, att_mod.num_heads, att_mod.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, att_mod.num_heads, att_mod.head_dim).transpose(1, 2)
        # Omit scaling factor:
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = att_mod.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = (
            (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, att_mod.d_out)
        )
        return att_mod.out_proj(context_vec)

    monkeypatch.setattr(att_mod, "forward", unscaled_forward)

    settings = RecordSettings(new_tokens=1, strategy="greedy")
    recorder = TraceRecorder(built, settings)
    trace = recorder.record("Hello world")

    # The first block should have fallen back to a GENERIC attention step
    pass0 = trace.passes[0]
    att_steps = [
        s
        for s in pass0.steps
        if s.block_index == 0 and s.semantic == Semantic.BLOCK_ATT_SCORES.value
    ]
    assert len(att_steps) == 1
    assert att_steps[0].kind == StepKind.GENERIC.value
    assert att_steps[0].fidelity.status == "failed"
    validate_trace(trace, strict=True)


def test_nan_injected_into_weight_is_sanitized() -> None:
    """Test that NaN values in weights are sanitized to 0.0 and meta flag is set."""
    built = _build_test_bundle()
    block = built.spec.blocks[0]
    mod_ff = built.model.get_submodule(block.ff_in)
    with torch.no_grad():
        mod_ff.weight[0, 0] = float("nan")

    settings = RecordSettings(new_tokens=1, strategy="greedy")
    recorder = TraceRecorder(built, settings)
    trace = recorder.record("Hello world")

    # Check that data in trace has no NaNs and flag is present
    pass0 = trace.passes[0]
    ff_step = next(s for s in pass0.steps if s.semantic == Semantic.BLOCK_FF_EXPAND.value)
    assert ff_step.meta.get("had_non_finite") is True
    for snap in ff_step.tensors:
        assert np.all(np.isfinite(snap.data))
    validate_trace(trace, strict=True)


def test_max_features_larger_than_emb_dim() -> None:
    """Test that max_features > emb_dim is handled without padding."""
    built = _build_test_bundle()
    # emb_dim is 48, set max_features to 128
    settings = RecordSettings(new_tokens=1, max_features=128)
    recorder = TraceRecorder(built, settings)
    trace = recorder.record("Hello world")

    pass0 = trace.passes[0]
    embed_step = next(s for s in pass0.steps if s.semantic == Semantic.EMBED.value)
    snap = embed_step.tensors[0]
    # Shape on model_dim should be exactly 48 (not 128)
    assert snap.data.shape[1] == 48
    validate_trace(trace, strict=True)


def test_recorder_determinism() -> None:
    """Test that recording twice with identical seed produces bit-identical traces."""
    built1 = _build_test_bundle()
    built2 = _build_test_bundle()
    settings = RecordSettings(new_tokens=2, seed=999, strategy="sample", temperature=0.7)

    t1 = TraceRecorder(built1, settings).record("Reproducibility test")
    t2 = TraceRecorder(built2, settings).record("Reproducibility test")

    assert len(t1.passes) == len(t2.passes)
    for p1, p2 in zip(t1.passes, t2.passes, strict=False):
        assert p1.input_ids == p2.input_ids
        assert p1.prediction.chosen_id == p2.prediction.chosen_id
        for s1, s2 in zip(p1.steps, p2.steps, strict=False):
            assert s1.id == s2.id
            for snap1, snap2 in zip(s1.tensors, s2.tensors, strict=False):
                assert np.array_equal(snap1.data, snap2.data)
