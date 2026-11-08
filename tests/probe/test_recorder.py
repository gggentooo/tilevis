"""Unit tests for TraceRecorder and non-interference guarantees.

Single responsibility: test end-to-end trace recording, non-interference of hooks,
and roundtrip validation of recorded traces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import build_matrix_model

from tilevis.core.config import RecordSettings
from tilevis.core.kinds import Semantic, StepKind
from tilevis.core.serialization import load_trace, save_trace
from tilevis.core.validation import validate_trace
from tilevis.probe.adapters import select_adapter
from tilevis.probe.loading.factory import BuiltModel
from tilevis.probe.recorder import TraceRecorder
from tilevis.probe.text.templates import TEMPLATES
from tilevis.probe.text.tokenizer import TiktokenTokenizer


def test_trace_recorder_non_interference_and_coverage() -> None:
    """Test TraceRecorder non-interference, template coverage, and hook cleanup."""
    entry = COMPAT_MATRIX["small_bias"]
    model = build_matrix_model(entry, seed=42)
    model.eval()

    # Pre-record baseline logits
    test_input = torch.tensor([[100, 200, 300]])
    with torch.no_grad():
        baseline_logits = model(test_input).clone()

    orig_training = model.training
    orig_req_grad = {name: p.requires_grad for name, p in model.named_parameters()}

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
    trace = recorder.record("Every effort moves")

    # 1. Non-interference checks
    with torch.no_grad():
        post_logits = model(test_input)
    assert torch.equal(baseline_logits, post_logits), "Model logits altered after recording!"
    assert model.training == orig_training, "Model training mode changed!"
    for name, p in model.named_parameters():
        assert p.requires_grad == orig_req_grad[name], f"requires_grad changed for {name}!"

    for mod in model.modules():
        assert len(mod._forward_hooks) == 0, "Dangling forward hook left on module!"
        assert len(mod._forward_pre_hooks) == 0, "Dangling forward pre-hook left on module!"

    # 2. Coverage and template order checks
    assert len(trace.passes) == 2
    for p in trace.passes:
        # Must have prompt, tokenize, embed, then blocks, then final_norm, lm_head, next_token
        semantics = [s.semantic for s in p.steps]
        assert semantics[0] == Semantic.PROMPT.value
        assert semantics[1] == Semantic.TOKENIZE.value
        assert semantics[2] == Semantic.EMBED.value

        # Last step must be next_token
        assert semantics[-1] == Semantic.NEXT_TOKEN.value
        assert p.steps[-1].kind == StepKind.NEXT_TOKEN.value

        # Block coverage: for each block b, must have block.summary, block.norm1, etc.
        for b in range(spec.n_layers if hasattr(spec, "n_layers") else len(spec.blocks)):
            assert f"p{p.index}.b{b}.summary" in [s.id for s in p.steps]

    # 3. Validation and serialization roundtrip
    validate_trace(trace, strict=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "recorder_test.trace.npz"
        save_trace(trace, tmp_path)
        loaded = load_trace(tmp_path)
        assert len(loaded.passes) == len(trace.passes)
        assert loaded.model.name == trace.model.name
