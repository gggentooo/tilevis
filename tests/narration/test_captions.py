"""Unit tests for tilevis.narration.captions."""

from __future__ import annotations

import logging

import pytest
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind
from tilevis.core.trace import Fidelity, PassRecord, Step
from tilevis.narration.captions import CaptionProvider


def test_every_semantic_and_stepkind_resolves_without_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every enum member of Semantic and StepKind resolves cleanly with zero warnings."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=2, ff_mult=4)
    pass_rec = trace.passes[0]
    provider = CaptionProvider("en")

    with caplog.at_level(logging.WARNING, logger="tilevis.narration"):
        # Verify all Semantics
        for sem in Semantic:
            step = Step(
                id=f"test.{sem.value}",
                kind=StepKind.GENERIC.value,
                semantic=sem.value,
                stage=Stage.BLOCK.value,
                block_index=0,
                detail_level=int(DetailLevel.STANDARD),
                module_path="trf_blocks.0",
                meta={"n_heads": 2, "emb_dim": 16, "ff_dim": 64},
            )
            cap = provider.caption(step, trace, pass_rec)
            assert cap.title.strip()
            assert cap.body.strip()
            assert cap.nerd.strip()
            assert "?" not in cap.title
            assert "?" not in cap.body

        # Verify all StepKinds with unknown semantic so StepKind fallback path is exercised
        for kind in StepKind:
            step = Step(
                id=f"test.kind.{kind.value}",
                kind=kind.value,
                semantic="unregistered.semantic",
                stage=Stage.BLOCK.value,
                block_index=0,
                detail_level=int(DetailLevel.STANDARD),
                module_path=None,
                meta={"n_heads": 2, "emb_dim": 16, "ff_dim": 64},
            )
            cap = provider.caption(step, trace, pass_rec)
            assert cap.title.strip()
            assert cap.body.strip()
            assert "?" not in cap.title
            assert "?" not in cap.body

    assert caplog.records == [], (
        f"Unexpected caption warnings: {[r.message for r in caplog.records]}"
    )


def test_unknown_semantic_and_kind_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Forward-compatible unknown Semantic and StepKind render cleanly using generic entry."""
    trace = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=3, n_passes=1)
    step = Step(
        id="p0.future",
        kind="FUTURE_OP",
        semantic="future.custom.op",
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=2,
        module_path=None,
    )
    provider = CaptionProvider("en")
    with caplog.at_level(logging.WARNING, logger="tilevis.narration"):
        cap = provider.caption(step, trace, trace.passes[0])

    assert cap.title == "Neural Layer Step"
    assert "number tiles" in cap.body
    assert any("fallback" in r.message for r in caplog.records)


def test_missing_placeholders_render_question_mark(caplog: pytest.LogCaptureFixture) -> None:
    """Missing values (None prediction, missing block_index, None n_heads) render '?' safely."""
    base_trace = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=2, n_passes=1)
    empty_pass = PassRecord(index=0, input_ids=(), input_tokens=(), text_before="", steps=())
    from dataclasses import replace

    trace_no_heads = replace(
        base_trace,
        model=replace(base_trace.model, n_heads=None),
        passes=(empty_pass,),
    )
    provider = CaptionProvider("en")

    # next_token with prediction=None triggers {top_word} and {top_prob:.0%} missing
    step_next = Step(
        id="p0.next",
        kind=StepKind.NEXT_TOKEN.value,
        semantic=Semantic.NEXT_TOKEN.value,
        stage=Stage.OUTPUT.value,
        block_index=None,
        detail_level=1,
        module_path=None,
    )
    # block.summary with block_index=None triggers {block_num} missing
    step_summary = Step(
        id="p0.b_missing",
        kind=StepKind.BLOCK_SUMMARY.value,
        semantic=Semantic.BLOCK_SUMMARY.value,
        stage=Stage.BLOCK.value,
        block_index=None,
        detail_level=1,
        module_path=None,
    )
    # attention scores with n_heads=None triggers {n_heads} missing
    step_attn = Step(
        id="p0.att",
        kind=StepKind.ATTENTION_PATTERN.value,
        semantic=Semantic.BLOCK_ATT_SCORES.value,
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=2,
        module_path="att",
        fidelity=Fidelity("unverified", note="Single token"),
    )

    with caplog.at_level(logging.WARNING, logger="tilevis.narration"):
        c_next = provider.caption(step_next, trace_no_heads, empty_pass)
        c_sum = provider.caption(step_summary, trace_no_heads, empty_pass)
        c_att = provider.caption(step_attn, trace_no_heads, empty_pass)

    assert "?" in c_next.body
    assert "?" in c_sum.title
    assert "?" in c_att.body
    assert "fidelity=unverified" in c_att.nerd
    assert any("Missing placeholder" in r.message for r in caplog.records)


def test_unknown_locale_falls_back_to_english(caplog: pytest.LogCaptureFixture) -> None:
    """Requesting a missing locale logs warning and serves English captions."""
    trace = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=2, n_passes=1)
    with caplog.at_level(logging.WARNING, logger="tilevis.narration"):
        provider = CaptionProvider("xx_nonexistent")
    cap = provider.caption(trace.passes[0].steps[0], trace, trace.passes[0])
    assert cap.title
    assert any("falling back to 'en'" in r.message for r in caplog.records)
