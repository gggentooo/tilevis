"""Compatibility test suite for probe recording across all compat_matrix variants.

Single responsibility: test non-interference, fidelity, coverage, generation matching,
and auto-detection across all architecture variants.
"""

from __future__ import annotations

import pytest
import torch
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import (
    apply_lora_wrapper,
    build_matrix_model,
)

from tilevis.core.config import RecordSettings
from tilevis.core.kinds import Semantic, StepKind
from tilevis.core.validation import validate_trace
from tilevis.probe.adapters import select_adapter
from tilevis.probe.loading.factory import BuiltModel
from tilevis.probe.recorder import TraceRecorder
from tilevis.probe.text.templates import TEMPLATES
from tilevis.probe.text.tokenizer import TiktokenTokenizer


@pytest.mark.parametrize(
    "matrix_name",
    [
        "tiny_ch07",
        "small_bias",
        "heads_8_of_64",
        "head_dim_1",
        "deep_narrow",
        "lora_small",
    ],
)
def test_compat_matrix_recording(matrix_name: str) -> None:
    """Test full recording, coverage, non-interference, and fidelity on all matrix entries."""
    entry = COMPAT_MATRIX[matrix_name]
    model = build_matrix_model(entry, seed=42)
    model.eval()

    if entry.variant == "lora":
        rank = entry.extras.get("lora_rank", 4) if entry.extras else 4
        alpha = entry.extras.get("lora_alpha", 8.0) if entry.extras else 8.0
        apply_lora_wrapper(model, rank=rank, alpha=alpha)

    # 1. Baseline prediction logits before recording
    test_ids = torch.tensor([[50, 60, 70]])
    with torch.no_grad():
        baseline_logits = model(test_ids).clone()

    orig_training = model.training
    orig_req_grad = {n: p.requires_grad for n, p in model.named_parameters()}

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

    settings = RecordSettings(new_tokens=2, strategy="greedy")
    recorder = TraceRecorder(built_model, settings)
    trace = recorder.record("A simple test prompt")

    # (a) Non-interference
    with torch.no_grad():
        post_logits = model(test_ids)
    assert torch.equal(baseline_logits, post_logits), f"Logits mismatch on {matrix_name}"
    assert model.training == orig_training
    for n, p in model.named_parameters():
        assert p.requires_grad == orig_req_grad[n]

    for mod in model.modules():
        assert len(mod._forward_hooks) == 0
        assert len(mod._forward_pre_hooks) == 0

    # (b) Coverage
    assert len(trace.passes) == 2
    for p in trace.passes:
        assert p.steps[-1].kind == StepKind.NEXT_TOKEN.value
        for b in range(entry.n_layers):
            assert f"p{p.index}.b{b}.summary" in [s.id for s in p.steps]

    # (c) Fidelity checks on steps
    for p in trace.passes:
        for s in p.steps:
            if s.semantic in (
                Semantic.BLOCK_ATT_SCORES.value,
                Semantic.BLOCK_RESID1.value,
                Semantic.BLOCK_RESID2.value,
            ):
                assert s.fidelity.status == "exact", f"Fidelity failed on {s.id}: {s.fidelity}"

    # (d) Auto-detection check: n_heads correctly confirmed
    assert trace.model.n_heads == entry.n_heads

    # (e) Validate trace
    validate_trace(trace, strict=True)


def test_compat_matrix_generic_adapter_recording() -> None:
    """Test recording when generic adapter is explicitly selected."""
    entry = COMPAT_MATRIX["tiny_ch07"]
    model = build_matrix_model(entry, seed=42)
    _adapter, spec, conf = select_adapter(model, adapter_name="generic")

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

    settings = RecordSettings(new_tokens=2, strategy="greedy")
    recorder = TraceRecorder(built_model, settings)
    trace = recorder.record("Hello world")

    assert len(trace.passes) == 2
    for p in trace.passes:
        assert p.steps[-1].kind == StepKind.NEXT_TOKEN.value
    validate_trace(trace, strict=True)


@pytest.mark.slow
def test_compat_matrix_gpt2_small_shape_recording() -> None:
    """Test performance and memory limits on gpt2_small_shape (slow)."""
    entry = COMPAT_MATRIX["gpt2_small_shape"]
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

    settings = RecordSettings(new_tokens=2, strategy="greedy")
    recorder = TraceRecorder(built_model, settings)
    trace = recorder.record("Hello world")

    assert len(trace.passes) == 2
    validate_trace(trace, strict=True)
