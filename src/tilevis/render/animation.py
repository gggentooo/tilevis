"""Tween, Timeline, and easing functions for step animations.

Single responsibility: drive continuous time-based interpolation with finish/skip capabilities.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


def _clamp01(t: float) -> float:
    """Clamp normalized time parameter cleanly into [0.0, 1.0]."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return float(t)


def linear(t: float) -> float:
    """Linear easing function over [0, 1]."""
    return _clamp01(t)


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out easing function decelerating toward 1.0."""
    u = _clamp01(t)
    return 1.0 - (1.0 - u) ** 3


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out easing function."""
    u = _clamp01(t)
    if u < 0.5:
        return 2.0 * u * u
    return 1.0 - ((-2.0 * u + 2.0) ** 2) / 2.0


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_quad": ease_in_out_quad,
}


@dataclass(frozen=True, slots=True)
class Tween:
    """Scalar value interpolation across a time interval [delay, delay + duration]."""

    start: float
    end: float
    duration: float
    easing: str | Callable[[float], float] = "ease_out_cubic"
    delay: float = 0.0

    def value_at(self, elapsed: float) -> float:
        """Evaluate interpolated value at given elapsed timeline seconds."""
        if elapsed <= self.delay:
            return self.start
        if self.duration <= 0.0 or elapsed >= self.delay + self.duration:
            return self.end
        raw_t = _clamp01((elapsed - self.delay) / self.duration)
        fn = EASINGS.get(self.easing, linear) if isinstance(self.easing, str) else self.easing
        eased = _clamp01(fn(raw_t))
        return self.start + (self.end - self.start) * eased


@dataclass(slots=True)
class Timeline:
    """Mutable animation clock supporting speed scaling and immediate completion."""

    duration: float
    speed: float = 1.0
    elapsed: float = 0.0
    _finished: bool = field(init=False)

    def __post_init__(self) -> None:
        self.duration = max(0.0, float(self.duration))
        self.speed = max(0.0, float(self.speed))
        self._finished = self.duration <= 0.0 or self.elapsed >= self.duration

    @property
    def is_finished(self) -> bool:
        """Return True once animation has reached completion."""
        return self._finished

    @property
    def progress(self) -> float:
        """Return normalized linear progress in [0.0, 1.0]."""
        if self._finished or self.duration <= 0.0:
            return 1.0
        return _clamp01(self.elapsed / self.duration)

    def eased(self, name: str = "ease_out_cubic") -> float:
        """Return normalized progress passed through named easing function."""
        fn = EASINGS.get(name, linear)
        return fn(self.progress)

    def update(self, dt: float, speed_override: float | None = None) -> float:
        """Advance timeline by max(0, dt) * speed and return current linear progress."""
        if self._finished:
            return 1.0
        eff_speed = self.speed if speed_override is None else max(0.0, float(speed_override))
        step = max(0.0, float(dt)) * eff_speed
        self.elapsed = min(self.duration, self.elapsed + step)
        if self.elapsed >= self.duration:
            self._finished = True
        return self.progress

    def finish(self) -> None:
        """Jump immediately to the final animation state (progress == 1.0)."""
        self.elapsed = self.duration
        self._finished = True

    def reset(self, *, animate: bool = True) -> None:
        """Reset timeline to start (or immediately finished if animate=False)."""
        if animate and self.duration > 0.0:
            self.elapsed = 0.0
            self._finished = False
        else:
            self.finish()
