"""Parameterized scene contract tests across every registered StepKind and synthetic card."""

from __future__ import annotations

import pygame
import pytest
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind
from tilevis.core.trace import Fidelity, PassRecord, Step, Trace
from tilevis.render.colormap import COLORMAPS
from tilevis.render.layout import UiScale
from tilevis.render.normalization import get_normalizer
from tilevis.render.scenes.base import SceneContext
from tilevis.render.scenes.generic import GenericScene
from tilevis.render.scenes.registry import SCENES, create_scene
from tilevis.render.text import FontManager
from tilevis.render.themes.builtin import get_theme


def _build_step_for_kind(kind: str) -> tuple[Trace, PassRecord, Step]:
    """Build valid Trace, PassRecord, and Step for any registered StepKind or synthetic card."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=4, seed=7)
    pass_rec = trace.passes[0]

    if kind == "TITLE":
        step = Step(
            id="synthetic.title",
            kind="TITLE",
            semantic="title",
            stage=Stage.INPUT.value,
            block_index=None,
            detail_level=int(DetailLevel.STORY),
            module_path=None,
            meta={
                "model_name": trace.model.name,
                "prompt": trace.prompt,
                "n_layers": trace.model.n_layers,
                "emb_dim": trace.model.emb_dim,
                "n_heads": trace.model.n_heads,
                "n_passes": len(trace.passes),
            },
            fidelity=Fidelity("exact"),
        )
        return trace, pass_rec, step

    if kind == "END":
        step = Step(
            id="synthetic.end",
            kind="END",
            semantic="end",
            stage=Stage.OUTPUT.value,
            block_index=None,
            detail_level=int(DetailLevel.STORY),
            module_path=None,
            meta={
                "model_name": trace.model.name,
                "n_passes": len(trace.passes),
            },
            fidelity=Fidelity("exact"),
        )
        return trace, pass_rec, step

    for s in pass_rec.steps:
        if s.kind == kind:
            return trace, pass_rec, s

    # Fallback (e.g. GENERIC): wrap a block output snapshot cleanly
    ref_step = trace.step_by_id("p0.b0.ff.act")
    step = Step(
        id="p0.generic",
        kind=StepKind.GENERIC.value,
        semantic=Semantic.GENERIC.value,
        stage=Stage.BLOCK.value,
        block_index=0,
        detail_level=int(DetailLevel.STANDARD),
        module_path="trf_blocks.0.generic",
        tensors=ref_step.tensors,
        fidelity=Fidelity("unverified", note="Generic contract fixture"),
    )
    return trace, pass_rec, step


def _make_context(
    trace: Trace,
    pass_rec: PassRecord,
    *,
    theme_name: str = "dark",
    animation_speed: float = 1.0,
) -> SceneContext:
    pygame.font.init()
    theme = get_theme(theme_name)
    scale = UiScale.from_viewport(1280, 720)
    return SceneContext(
        theme=theme,
        colormaps=COLORMAPS,
        normalizer=get_normalizer("shared_step"),
        fonts=FontManager(theme.font_family),
        scale=scale,
        trace=trace,
        pass_record=pass_rec,
        detail_level=int(DetailLevel.STANDARD),
        nerd_mode=False,
        animation_speed=animation_speed,
    )


def test_every_stepkind_is_registered_with_dedicated_scene() -> None:
    """Every StepKind (plus TITLE and END) has a registered Scene class."""
    registered = set(SCENES.keys())
    expected = {k.value for k in StepKind} | {"TITLE", "END"}
    assert expected <= registered

    for k in StepKind:
        scene = create_scene(k.value)
        if k != StepKind.GENERIC:
            assert not isinstance(scene, GenericScene), f"{k.value} fell back to GenericScene"

    assert isinstance(create_scene("future_unrecognized_op"), GenericScene)


@pytest.mark.parametrize("kind", sorted(SCENES.keys()))
@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_scene_lifecycle_finish_equivalence_and_hit_test(kind: str, theme_name: str) -> None:
    """Verify full Scene Protocol contract for every registered kind under dark and light themes."""
    pygame.init()
    trace, pass_rec, step = _build_step_for_kind(kind)
    ctx = _make_context(trace, pass_rec, theme_name=theme_name, animation_speed=1.0)
    stage = ctx.scale.stage
    w, h = ctx.scale.viewport_w, ctx.scale.viewport_h

    # 1. Instant-speed check: speed=0.0 finishes immediately on enter()
    ctx_instant = _make_context(trace, pass_rec, theme_name=theme_name, animation_speed=0.0)
    s_instant = create_scene(kind)
    s_instant.enter(step, ctx_instant, animate=True)
    assert not s_instant.is_animating

    # 2. Animated entry starts animating and terminates within <= 5s at 16ms increments
    s_anim = create_scene(kind)
    s_anim.enter(step, ctx, animate=True)
    assert s_anim.is_animating

    surf_anim = pygame.Surface((w, h))
    surf_anim.fill(ctx.theme.background)
    s_anim.draw(surf_anim, stage)  # frame at t=0

    elapsed = 0.0
    frames = 0
    while s_anim.is_animating and elapsed < 5.0:
        s_anim.update(0.016)
        elapsed += 0.016
        frames += 1
        if frames % 10 == 0 or not s_anim.is_animating:
            surf_anim.fill(ctx.theme.background)
            s_anim.draw(surf_anim, stage)

    assert not s_anim.is_animating, f"{kind} did not finish within 5s simulated time"
    bytes_stepped = pygame.image.tobytes(surf_anim, "RGB")

    # 3. finish() produces bit-identical rendered surface to stepping update() to completion
    s_finish = create_scene(kind)
    s_finish.enter(step, ctx, animate=True)
    assert s_finish.is_animating
    s_finish.finish()
    assert not s_finish.is_animating

    surf_finish = pygame.Surface((w, h))
    surf_finish.fill(ctx.theme.background)
    s_finish.draw(surf_finish, stage)
    assert pygame.image.tobytes(surf_finish, "RGB") == bytes_stepped

    # 4. enter(..., animate=False) (used by PREV) also starts directly in identical end state
    s_prev = create_scene(kind)
    s_prev.enter(step, ctx, animate=False)
    assert not s_prev.is_animating

    surf_prev = pygame.Surface((w, h))
    surf_prev.fill(ctx.theme.background)
    s_prev.draw(surf_prev, stage)
    assert pygame.image.tobytes(surf_prev, "RGB") == bytes_stepped

    # 5. Hit testing returns None on empty background and non-empty Tooltips on widgets/tiles
    assert s_finish.hit_test((0, 0)) is None

    cards = getattr(s_finish, "_cards", [])
    panels = getattr(s_finish, "_panels", [])
    regions = getattr(s_finish, "_hit_regions", [])
    grid_single = getattr(s_finish, "_grid", None)

    hit_targets: list[tuple[int, int]] = [card.grid.cell_center(0, 0) for card in cards]
    hit_targets.extend(panel.grid.cell_center(0, 0) for panel in panels)
    if grid_single is not None:
        hit_targets.append(grid_single.cell_center(0, 0))
    if regions:
        hit_targets.append(regions[0].rect.center)

    assert hit_targets, f"Expected at least one interactive card or region on {kind}"
    for pt in hit_targets:
        tt = s_finish.hit_test(pt)
        assert tt is not None, f"hit_test({pt}) returned None on {kind}"
        assert tt.title
        assert tt.lines
