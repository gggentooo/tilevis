"""Unit and Hypothesis property tests for playlist filtering and StepPlayer."""

from __future__ import annotations

from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel, Semantic, StepKind
from tilevis.render.player import StepPlayer
from tilevis.render.playlist import build_playlist


def test_playlist_filtering_detail_levels_and_block_modes() -> None:
    """Playlist respects STORY/STANDARD/DEEP detail filtering and detail_blocks selectors."""
    trace = make_fake_trace(n_layers=3, emb_dim=16, n_tokens=4, n_passes=2)

    # STORY collapses all blocks to BLOCK_SUMMARY
    story_pl = build_playlist(trace, PresenterSettings(detail_level=1))
    story_block_kinds = [r.kind for r in story_pl if r.block_index is not None]
    assert story_block_kinds
    assert all(k == StepKind.BLOCK_SUMMARY.value for k in story_block_kinds)

    # STANDARD with "first" expands block 0 in pass 0 and collapses blocks 1, 2
    std_pl = build_playlist(trace, PresenterSettings(detail_level=2, detail_blocks="first"))
    p0_b0 = [r for r in std_pl if r.pass_index == 0 and r.block_index == 0]
    p0_b1 = [r for r in std_pl if r.pass_index == 0 and r.block_index == 1]
    assert all(r.kind != StepKind.BLOCK_SUMMARY.value for r in p0_b0)
    assert len(p0_b0) > 5
    assert [r.kind for r in p0_b1] == [StepKind.BLOCK_SUMMARY.value]

    # Pass 1 uses later_pass_level=1 (STORY) -> collapsed even for block 0
    p1_b0 = [r for r in std_pl if r.pass_index == 1 and r.block_index == 0]
    assert [r.kind for r in p1_b0] == [StepKind.BLOCK_SUMMARY.value]

    # "first_last" expands 0 and 2
    fl_pl = build_playlist(trace, PresenterSettings(detail_level=2, detail_blocks="first_last"))
    assert all(
        r.kind != StepKind.BLOCK_SUMMARY.value
        for r in fl_pl
        if r.pass_index == 0 and r.block_index in {0, 2}
    )
    assert [r.kind for r in fl_pl if r.pass_index == 0 and r.block_index == 1] == [
        StepKind.BLOCK_SUMMARY.value
    ]

    # Explicit list [1]
    list_pl = build_playlist(trace, PresenterSettings(detail_level=2, detail_blocks=[1]))
    assert [r.kind for r in list_pl if r.pass_index == 0 and r.block_index == 0] == [
        StepKind.BLOCK_SUMMARY.value
    ]
    assert len([r for r in list_pl if r.pass_index == 0 and r.block_index == 1]) > 5

    # Synthetic cards toggle
    bare_pl = build_playlist(
        trace, PresenterSettings(include_title=False, include_end=False, detail_level=1)
    )
    assert not any(r.is_synthetic for r in bare_pl)


def test_player_reanchor_preserves_pass_and_block() -> None:
    """Switching detail_level keeps cursor anchored on the same pass and block."""
    trace = make_fake_trace(n_layers=3, emb_dim=16, n_tokens=4, n_passes=2)
    player = StepPlayer(trace, PresenterSettings(detail_level=2, detail_blocks="first"))

    assert player.goto_block(0, 0)
    # Advance into internal layer step inside block 0 (e.g., attention scores)
    for _ in range(3):
        player.next()
    before = player.current
    assert before is not None and before.pass_index == 0 and before.block_index == 0

    # Switch to STORY -> should land on pass 0, block 0 BLOCK_SUMMARY
    reanchored = player.update_settings(PresenterSettings(detail_level=1))
    assert reanchored is not None
    assert reanchored.pass_index == 0
    assert reanchored.block_index == 0
    assert reanchored.semantic == Semantic.BLOCK_SUMMARY.value

    # Switch back to STANDARD -> returns to pass 0, block 0
    reanchored2 = player.update_settings(PresenterSettings(detail_level=2, detail_blocks="first"))
    assert reanchored2 is not None
    assert reanchored2.pass_index == 0
    assert reanchored2.block_index == 0


def test_single_step_playlist_and_eos_bounds() -> None:
    """Bounded navigation never wraps or crashes on 1-step playlists and prediction=None."""
    trace = make_fake_trace(n_layers=1, emb_dim=8, n_tokens=2, n_passes=1)
    eos_pass = replace(trace.passes[0], steps=(trace.passes[0].steps[0],), prediction=None)
    minimal_trace = replace(trace, passes=(eos_pass,))
    settings_min = PresenterSettings(include_title=False, include_end=False)
    player = StepPlayer(minimal_trace, settings_min)

    assert player.total_steps == 1
    assert not player.prev()
    assert not player.next()
    assert not player.next_stage()
    assert not player.prev_stage()
    assert not player.next_pass()
    assert not player.first()
    assert not player.last()
    assert player.cursor == 0


@given(
    actions=st.lists(
        st.one_of(
            st.tuples(st.just("next")),
            st.tuples(st.just("prev")),
            st.tuples(st.just("next_stage")),
            st.tuples(st.just("prev_stage")),
            st.tuples(st.just("first")),
            st.tuples(st.just("last")),
            st.tuples(st.just("next_pass")),
            st.tuples(st.just("goto_block"), st.integers(0, 1), st.integers(0, 3)),
            st.tuples(st.just("detail"), st.sampled_from([1, 2, 3])),
        ),
        min_size=1,
        max_size=60,
    )
)
@settings(max_examples=50, deadline=None)
def test_property_step_player_navigation_invariants(actions: list[tuple[object, ...]]) -> None:
    """Random navigation sequences always maintain valid cursor, never wrap, and re-anchor."""
    trace = make_fake_trace(n_layers=4, emb_dim=16, n_tokens=3, n_passes=2)
    player = StepPlayer(trace, PresenterSettings(detail_level=2, detail_blocks="first_last"))

    for act in actions:
        op = act[0]
        prev_cur = player.cursor
        prev_ref = player.current

        if op == "next":
            changed = player.next()
            assert changed == (prev_cur + 1 < player.total_steps)
            if not changed:
                assert player.cursor == player.total_steps - 1
        elif op == "prev":
            changed = player.prev()
            assert changed == (prev_cur > 0)
            if not changed:
                assert player.cursor == 0
        elif op == "next_stage":
            player.next_stage()
        elif op == "prev_stage":
            player.prev_stage()
        elif op == "first":
            player.first()
            assert player.cursor == 0
        elif op == "last":
            player.last()
            assert player.cursor == player.total_steps - 1
        elif op == "next_pass":
            player.next_pass()
        elif op == "goto_block":
            _, p_i, b_i = act
            player.goto_block(int(p_i), int(b_i))  # type: ignore[arg-type]
        elif op == "detail":
            level = int(act[1])  # type: ignore[arg-type]
            new_cfg = PresenterSettings(detail_level=level, detail_blocks="first_last")
            new_ref = player.update_settings(new_cfg)
            assert new_ref is not None
            if (
                prev_ref is not None
                and prev_ref.pass_index is not None
                and prev_ref.block_index is not None
            ):
                # Re-anchoring preserves pass_index and block_index when block still exists
                assert new_ref.pass_index == prev_ref.pass_index
                assert new_ref.block_index == prev_ref.block_index

        assert 0 <= player.cursor < player.total_steps
        assert player.current is not None
        assert player.current.detail_level <= max(
            DetailLevel.STORY,
            player.settings.detail_level
            if player.current.pass_index in (None, 0)
            else player.settings.later_pass_level,
        )
