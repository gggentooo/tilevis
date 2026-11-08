"""Unit tests for Timeline, Tween, and easing functions."""

from __future__ import annotations

from tilevis.render.animation import Timeline, Tween, ease_in_out_quad, ease_out_cubic, linear


def test_easing_endpoints_and_monotonicity() -> None:
    """All built-in easings map [0, 1] monotonically from 0.0 to 1.0."""
    for fn in (linear, ease_out_cubic, ease_in_out_quad):
        assert fn(-0.5) == 0.0
        assert fn(0.0) == 0.0
        assert fn(1.0) == 1.0
        assert fn(2.0) == 1.0
        samples = [fn(i / 20.0) for i in range(21)]
        assert all(samples[i] <= samples[i + 1] for i in range(20))


def test_timeline_progress_speed_and_finish() -> None:
    """Timeline tracks dt monotonically, respects speed multiplier, and jumps on finish()."""
    tl = Timeline(duration=1.0, speed=2.0)
    assert not tl.is_finished
    assert tl.progress == 0.0

    # Negative dt clamped safely
    assert tl.update(-0.5) == 0.0
    assert not tl.is_finished

    p1 = tl.update(0.15)  # 0.15 * 2.0 = 0.30
    assert abs(p1 - 0.30) < 1e-6
    assert not tl.is_finished

    p2 = tl.update(0.15)
    assert abs(p2 - 0.60) < 1e-6

    tl.finish()
    assert tl.is_finished
    assert tl.progress == 1.0

    tl.reset(animate=True)
    assert not tl.is_finished
    tl.update(1.0)
    assert tl.is_finished
    assert tl.progress == 1.0


def test_tween_delay_and_interpolation() -> None:
    """Tween respects delay interval and lands precisely on end value."""
    tw = Tween(start=10.0, end=50.0, duration=0.5, easing="linear", delay=0.2)
    assert tw.value_at(0.0) == 10.0
    assert tw.value_at(0.2) == 10.0
    assert abs(tw.value_at(0.45) - 30.0) < 1e-6
    assert tw.value_at(0.7) == 50.0
    assert tw.value_at(5.0) == 50.0
