"""Dedicated behavior checks and edge-shape weakness probes for Phase 6 scenes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pygame
import pytest
from tests.fixtures.fake_traces import make_fake_trace, make_snapshot

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind, TensorRole
from tilevis.core.trace import Fidelity, PassRecord, Prediction, Step
from tilevis.render.colormap import COLORMAPS
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.normalization import get_normalizer
from tilevis.render.scenes.attention import AttentionPatternScene
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.embedding import EmbeddingScene
from tilevis.render.scenes.layernorm import LayerNormScene
from tilevis.render.scenes.logits import LogitsScene
from tilevis.render.scenes.matmul import MatmulScene
from tilevis.render.scenes.prediction import PredictionScene
from tilevis.render.scenes.title import EndScene, TitleScene
from tilevis.render.scenes.tokenize import TokenizeScene
from tilevis.render.text import FontManager
from tilevis.render.themes.builtin import get_theme


def _ctx(
    trace=None,
    pass_rec: PassRecord | None = None,
    *,
    theme_name: str = "dark",
    w: int = 1280,
    h: int = 720,
    confetti: bool | None = None,
) -> SceneContext:
    pygame.font.init()
    tr = trace or make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=4, seed=9)
    pr = pass_rec or tr.passes[0]
    theme = get_theme(theme_name)
    if confetti is not None:
        theme = replace(theme, confetti=confetti)
    scale = UiScale.from_viewport(w, h)
    return SceneContext(
        theme=theme,
        colormaps=COLORMAPS,
        normalizer=get_normalizer("shared_step"),
        fonts=FontManager(theme.font_family),
        scale=scale,
        trace=tr,
        pass_record=pr,
        detail_level=int(DetailLevel.STANDARD),
        nerd_mode=False,
        animation_speed=1.0,
    )


def test_attention_head_cycling_causal_mask_and_self_loop() -> None:
    """Verify ] / [ / A head commands, causal mask colors/tooltips, and T=1 self-loop."""
    trace = make_fake_trace(n_layers=1, emb_dim=16, n_tokens=4, n_passes=1, n_heads=4, seed=11)
    pr = trace.passes[0]
    attn_step = trace.step_by_id("p0.b0.att.scores")
    ctx = _ctx(trace, pr)
    surf = pygame.Surface((ctx.scale.viewport_w, ctx.scale.viewport_h))

    scene = AttentionPatternScene()
    scene.enter(attn_step, ctx, animate=False)
    scene.draw(surf, ctx.scale.stage)

    assert scene.head_count == 4
    assert scene.head_index == 0
    assert scene.average_heads is False

    # Cycle forward through all 4 heads and wrap back to 0
    assert scene.handle_command(Command.CYCLE_HEAD) is True
    assert scene.head_index == 1
    assert scene.handle_command(Command.CYCLE_HEAD) is True
    assert scene.head_index == 2
    assert scene.handle_command(Command.CYCLE_HEAD) is True
    assert scene.head_index == 3
    assert scene.handle_command(Command.CYCLE_HEAD) is True
    assert scene.head_index == 0

    # Backward wraps cleanly 0 -> 3
    assert scene.handle_command(Command.CYCLE_HEAD_PREV) is True
    assert scene.head_index == 3
    assert scene.handle_command(Command.CYCLE_HEAD_PREV) is True
    assert scene.head_index == 2

    # Average toggle switches mode and cycling head exits average cleanly
    assert scene.handle_command(Command.AVG_HEADS) is True
    assert scene.average_heads is True
    scene.draw(surf, ctx.scale.stage)
    assert scene.handle_command(Command.CYCLE_HEAD) is True
    assert scene.average_heads is False
    assert scene.head_index == 3

    # Unhandled commands return False
    assert scene.handle_command(Command.NEXT) is False
    assert scene.handle_command(Command.TOGGLE_NERD) is False

    # Verify causal upper-triangle cell renders exact masked_cell color & tooltip
    scene.draw(surf, ctx.scale.stage)
    assert scene._grid is not None
    upper_pos = scene._grid.cell_center(0, 3)  # col 3 > row 0 -> future masked
    assert surf.get_at(upper_pos)[:3] == ctx.theme.masked_cell
    tt_masked = scene.hit_test(upper_pos)
    assert tt_masked is not None and "Masked" in tt_masked.title

    lower_pos = scene._grid.cell_center(3, 0)  # row 3 >= col 0 -> active attention weight
    tt_weight = scene.hit_test(lower_pos)
    assert tt_weight is not None and "Attention Weight" in tt_weight.title

    # Edge case T=1 single-token self-attention loop + n_heads=1
    tr1 = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=1, n_passes=1, n_heads=1, seed=3)
    s1 = tr1.step_by_id("p0.b0.att.scores")
    ctx1 = _ctx(tr1, tr1.passes[0])
    sc1 = AttentionPatternScene()
    sc1.enter(s1, ctx1, animate=False)
    sc1.draw(surf, ctx1.scale.stage)
    assert sc1.head_count == 1
    assert sc1.handle_command(Command.CYCLE_HEAD) is True
    assert sc1.head_index == 0
    assert len(sc1._arc_nodes) == 1


@pytest.mark.parametrize(
    ("semantic", "expected_tag"),
    [
        (Semantic.BLOCK_ATT_OUT.value, "att.out"),
        (Semantic.BLOCK_FF_EXPAND.value, "ff.expand"),
        (Semantic.BLOCK_FF_CONTRACT.value, "ff.contract"),
    ],
)
def test_matmul_orientation_labels_and_bias_variants(semantic: str, expected_tag: str) -> None:
    """Verify (T, in) x (in, out) -> (T, out) orientation, bias handling, and semantic tags."""
    in_dim, out_dim, tok = 12, 24, 5
    rng = np.random.default_rng(5)
    t_lbls = ("a", "b", "c", "d", "e")
    snaps_base = (
        make_snapshot(
            "in",
            TensorRole.INPUT,
            rng.normal(size=(tok, in_dim)),
            ("token", "model_dim"),
            token_labels=t_lbls,
        ),
        make_snapshot(
            "w", TensorRole.WEIGHT, rng.normal(size=(in_dim, out_dim)), ("model_dim", "ff_dim")
        ),
        make_snapshot(
            "out",
            TensorRole.OUTPUT,
            rng.normal(size=(tok, out_dim)),
            ("token", "ff_dim"),
            token_labels=t_lbls,
        ),
    )
    bias_snap = make_snapshot("b", TensorRole.BIAS, rng.normal(size=(out_dim,)), ("ff_dim",))

    trace = make_fake_trace(n_layers=1, emb_dim=in_dim, n_tokens=tok, n_passes=1)
    ctx = _ctx(trace)
    surf = pygame.Surface((ctx.scale.viewport_w, ctx.scale.viewport_h))

    for with_bias in (False, True):
        snaps = (*snaps_base, bias_snap) if with_bias else snaps_base
        step = Step(
            id=f"p0.{semantic}.{with_bias}",
            kind=StepKind.LINEAR.value,
            semantic=semantic,
            stage=Stage.BLOCK.value,
            block_index=0,
            detail_level=int(DetailLevel.STANDARD),
            module_path="trf_blocks.0.ff.layers.0",
            tensors=snaps,
            meta={"bias": with_bias},
            fidelity=Fidelity("exact"),
        )
        scene = MatmulScene()
        scene.enter(step, ctx, animate=True)
        scene.update(0.6)
        scene.draw(surf, ctx.scale.stage)

        assert scene.displayed_tokens == tok
        assert scene.displayed_in_dim == in_dim
        assert scene.displayed_out_dim == out_dim
        assert expected_tag in scene._banner_text
        assert scene._bias_snap is (bias_snap if with_bias else None)
        assert scene._in_card is not None
        assert scene._w_card is not None
        assert scene._out_card is not None
        assert scene._w_card.grid.orig_rows == in_dim
        assert scene._w_card.grid.orig_cols == out_dim

        scene.finish()
        scene.draw(surf, ctx.scale.stage)

    # Also handles minimal fake trace step without INPUT snapshot gracefully
    min_step = replace(trace.step_by_id("p0.b0.ff.expand"), semantic=semantic)
    scene_min = MatmulScene()
    scene_min.enter(min_step, ctx, animate=False)
    scene_min.draw(surf, ctx.scale.stage)
    assert scene_min._placeholder_rect is not None
    assert scene_min.hit_test(scene_min._placeholder_rect.center) is not None


def test_prediction_scene_monotonic_bars_sampled_non_argmax_winner_and_confetti() -> None:
    """Verify bar monotonicity, non-argmax sampled winner highlight, and theme confetti toggle."""
    trace = make_fake_trace(n_layers=1, emb_dim=16, n_tokens=4, n_passes=1, seed=21)
    probs = (0.42, 0.25, 0.16, 0.10, 0.05, 0.02)
    cand_ids = (10, 20, 30, 40, 50, 60)
    cand_toks = ("·world", "·life", "·day", "·star", "·dream", "·hope")
    # Deliberately pick rank #3 ("·star", p=0.10) instead of argmax ("·world", p=0.42)
    sampled_pred = Prediction(
        candidate_ids=cand_ids,
        candidate_tokens=cand_toks,
        candidate_probs=probs,
        chosen_id=40,
        chosen_token="·star",
        strategy="sample",
    )
    mod_pass = replace(trace.passes[0], prediction=sampled_pred)
    next_step = trace.step_by_id("p0.next_token")

    ctx_confetti = _ctx(trace, mod_pass, confetti=True)
    surf = pygame.Surface((ctx_confetti.scale.viewport_w, ctx_confetti.scale.viewport_h))
    scene = PredictionScene()
    scene.enter(next_step, ctx_confetti, animate=True)
    scene.update(
        0.98
    )  # inside active confetti burst window 0.55 <= t < 0.96? Wait: 0.98/1.35 = 0.726
    scene.draw(surf, ctx_confetti.scale.stage)

    assert scene.confetti_drawn_last_frame > 0
    assert len(scene.bars) >= 4

    # Strictly non-increasing bar widths for descending candidate_probs
    for i in range(len(scene.bars) - 1):
        assert scene.bars[i].prob >= scene.bars[i + 1].prob
        assert scene.bars[i].bar_rect.width >= scene.bars[i + 1].bar_rect.width

    # Non-argmax rank-3 candidate is highlighted winner, NOT rank-0
    winners = [b for b in scene.bars if b.is_winner]
    assert len(winners) == 1
    assert winners[0].candidate_id == 40
    assert winners[0].token == "·star"
    assert scene.bars[0].is_winner is False

    # Confetti disappears cleanly at t=1.0
    scene.finish()
    scene.draw(surf, ctx_confetti.scale.stage)
    assert scene.confetti_drawn_last_frame == 0

    # Theme confetti=False suppresses confetti even mid-animation
    ctx_no_conf = _ctx(trace, mod_pass, confetti=False)
    scene2 = PredictionScene()
    scene2.enter(next_step, ctx_no_conf, animate=True)
    scene2.update(0.98)
    scene2.draw(surf, ctx_no_conf.scale.stage)
    assert scene2.confetti_drawn_last_frame == 0

    # EOS step with prediction=None renders cleanly
    eos_pass = replace(trace.passes[0], prediction=None)
    eos_step = Step(
        id="p0.eos",
        kind=StepKind.NEXT_TOKEN.value,
        semantic=Semantic.NEXT_TOKEN.value,
        stage=Stage.OUTPUT.value,
        block_index=None,
        detail_level=int(DetailLevel.STORY),
        module_path=None,
        meta={},
        fidelity=Fidelity("exact"),
    )
    scene_eos = PredictionScene()
    scene_eos.enter(eos_step, _ctx(trace, eos_pass), animate=False)
    scene_eos.draw(surf, ctx_confetti.scale.stage)
    assert scene_eos.bars == []
    assert scene_eos.confetti_drawn_last_frame == 0


def test_scene_edge_shapes_missing_pos_layernorm_1d_logits_and_24_tokens() -> None:
    """Verify edge cases: pos_emb=None, LN strips, 1-D vs 2-D logits, T=24, EOS EndScene."""
    trace = make_fake_trace(n_layers=1, emb_dim=16, n_tokens=4, n_passes=1, seed=17)
    surf_big = pygame.Surface((1280, 720))

    # 1. EmbeddingScene with pos_emb=None
    embed_step = trace.step_by_id("p0.embed")
    tok_only = tuple(t for t in embed_step.tensors if t.role != TensorRole.POSITION_VECTORS.value)
    no_pos_step = replace(embed_step, tensors=tok_only, meta={"has_pos_emb": False})
    emb_sc = EmbeddingScene()
    ctx = _ctx(trace)
    emb_sc.enter(no_pos_step, ctx, animate=False)
    emb_sc.draw(surf_big, ctx.scale.stage)
    assert emb_sc._missing_pos_rect is not None
    assert emb_sc.hit_test(emb_sc._missing_pos_rect.center) is not None

    # 2. LayerNormScene with both SCALE & SHIFT present AND with minimal OUTPUT-only
    rng = np.random.default_rng(2)
    ln_full = Step(
        id="p0.ln.full",
        kind=StepKind.LAYERNORM.value,
        semantic=Semantic.BLOCK_NORM1.value,
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=int(DetailLevel.STANDARD),
        module_path="trf_blocks.0.norm1",
        tensors=(
            make_snapshot("in", TensorRole.INPUT, rng.normal(size=(4, 16)), ("token", "model_dim")),
            make_snapshot("sc", TensorRole.SCALE, np.ones((16,), dtype=np.float32), ("model_dim",)),
            make_snapshot(
                "sh", TensorRole.SHIFT, np.zeros((16,), dtype=np.float32), ("model_dim",)
            ),
            make_snapshot(
                "out", TensorRole.OUTPUT, rng.normal(size=(4, 16)), ("token", "model_dim")
            ),
        ),
        fidelity=Fidelity("exact"),
    )
    ln_sc = LayerNormScene()
    ln_sc.enter(ln_full, ctx, animate=False)
    ln_sc.draw(surf_big, ctx.scale.stage)
    assert len(ln_sc._cards) == 4  # Before + Scale + Shift + After
    assert ln_sc._spread_before_rect is not None
    assert ln_sc._spread_after_rect is not None

    ln_min = trace.step_by_id("p0.final_norm")
    ln_sc2 = LayerNormScene()
    ln_sc2.enter(ln_min, ctx, animate=False)
    ln_sc2.draw(surf_big, ctx.scale.stage)
    assert len(ln_sc2._cards) >= 1

    # 3. LogitsScene with both 1-D (k,) and 2-D (1, k) logits + WEIGHT
    lm_1d = trace.step_by_id("p0.lm_head")
    log_sc1 = LogitsScene()
    log_sc1.enter(lm_1d, ctx, animate=False)
    log_sc1.draw(surf_big, ctx.scale.stage)
    assert len(log_sc1._logit_rows) >= 5

    lm_2d = replace(
        lm_1d,
        tensors=(
            make_snapshot("in", TensorRole.INPUT, rng.normal(size=(1, 16)), ("token", "model_dim")),
            make_snapshot("w", TensorRole.WEIGHT, rng.normal(size=(16, 8)), ("model_dim", "vocab")),
            make_snapshot("logits", TensorRole.LOGITS, rng.normal(size=(1, 8)), ("token", "vocab")),
        ),
    )
    log_sc2 = LogitsScene()
    log_sc2.enter(lm_2d, ctx, animate=False)
    log_sc2.draw(surf_big, ctx.scale.stage)
    assert log_sc2._scoreboard_rect.width > 0

    # 4. TokenizeScene with T=24 Korean, newline, and long tokens at small viewport
    long_toks = tuple(
        ["안녕", "·세상", "↵", "supercalifragilistic", "·테스트"] * 4 + ["t21", "t22", "t23", "t24"]
    )
    long_ids = tuple(range(24))
    pass_24 = replace(
        trace.passes[0], input_tokens=long_toks, input_ids=long_ids, text_before="안녕 세상"
    )
    tok_step = Step(
        id="p0.tok24",
        kind=StepKind.TOKENIZE.value,
        semantic=Semantic.TOKENIZE.value,
        stage=Stage.INPUT.value,
        block_index=None,
        detail_level=int(DetailLevel.STANDARD),
        module_path=None,
        meta={"tokens": list(long_toks), "token_ids": list(long_ids)},
        fidelity=Fidelity("exact"),
    )
    surf_small = pygame.Surface((800, 600))
    ctx_small = _ctx(trace, pass_24, w=800, h=600)
    tok_sc = TokenizeScene()
    tok_sc.enter(tok_step, ctx_small, animate=False)
    tok_sc.draw(surf_small, ctx_small.scale.stage)
    assert len(tok_sc._chips) >= 1

    # 5. TitleScene (n_heads=None) & EndScene (EOS 0 generated tokens)
    t_sc = TitleScene()
    t_step = Step(
        id="synthetic.title",
        kind="TITLE",
        semantic="title",
        stage=Stage.INPUT.value,
        block_index=None,
        detail_level=int(DetailLevel.STORY),
        module_path=None,
        meta={
            "model_name": "Custom",
            "prompt": "Hello",
            "n_layers": 2,
            "emb_dim": 8,
            "n_heads": None,
        },
        fidelity=Fidelity("exact"),
    )
    t_sc.enter(t_step, ctx, animate=False)
    t_sc.draw(surf_big, ctx.scale.stage)
    assert any(b.value == "Dynamic" for b in t_sc._badges)

    eos_trace = replace(trace, passes=(replace(trace.passes[0], prediction=None),))
    e_sc = EndScene()
    e_step = Step(
        id="synthetic.end",
        kind="END",
        semantic="end",
        stage=Stage.OUTPUT.value,
        block_index=None,
        detail_level=int(DetailLevel.STORY),
        module_path=None,
        meta={"model_name": "Custom", "n_passes": 1},
        fidelity=Fidelity("exact"),
    )
    e_sc.enter(e_step, _ctx(eos_trace, eos_trace.passes[0]), animate=False)
    e_sc.draw(surf_big, ctx.scale.stage)
    assert not any(c.is_generated for c in e_sc._chips)
