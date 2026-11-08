"""Theme dataclass definition.

Single responsibility: define theme properties (background, text, accent, fonts, tile gaps).
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

ColorRGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Theme:
    """Visual styling tokens and layout metrics for renderer themes."""

    name: str
    background: ColorRGB
    panel: ColorRGB
    panel_alt: ColorRGB
    border: ColorRGB
    text_primary: ColorRGB
    text_secondary: ColorRGB
    accent: ColorRGB
    accent_muted: ColorRGB
    masked_cell: ColorRGB
    highlight: ColorRGB
    success: ColorRGB
    warning: ColorRGB
    error: ColorRGB
    font_family: str = "NotoSans"
    title_font_px: int = 42
    heading_font_px: int = 32
    body_font_px: int = 26
    small_font_px: int = 22
    tile_gap_px: int = 1
    corner_radius: int = 10
    diverging_cmap: str = "diverging_blue_orange"
    sequential_cmap: str = "sequential_purple"
    confetti: bool = True
    animation_speed_factor: float = 1.0
