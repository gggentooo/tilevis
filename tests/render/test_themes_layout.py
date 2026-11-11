"""Tests for Theme contrast ratios, UiScale layouts, and bundled FontManager."""

from __future__ import annotations

import pytest

from tilevis.render.layout import UiScale
from tilevis.render.text import FontManager
from tilevis.render.themes.base import ColorRGB
from tilevis.render.themes.builtin import THEMES


def _rel_luminance(rgb: ColorRGB) -> float:
    def _lin(c255: int) -> float:
        c = c255 / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast_ratio(c1: ColorRGB, c2: ColorRGB) -> float:
    l1 = _rel_luminance(c1)
    l2 = _rel_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def test_builtin_themes_wcag_contrast_ratio() -> None:
    """Both dark and light themes meet >= 4.5:1 WCAG contrast on background and panels."""
    assert "dark" in THEMES
    assert "light" in THEMES

    for name in ("dark", "light"):
        theme = THEMES.get(name)
        for fg_name, fg in (
            ("text_primary", theme.text_primary),
            ("text_secondary", theme.text_secondary),
        ):
            for bg_name, bg in (
                ("background", theme.background),
                ("panel", theme.panel),
                ("panel_alt", theme.panel_alt),
            ):
                ratio = _contrast_ratio(fg, bg)
                assert ratio >= 4.5, (
                    f"{name}: {fg_name} vs {bg_name} contrast ratio {ratio:.2f} < 4.5"
                )


@pytest.mark.parametrize(
    ("vw", "vh"),
    [
        (800, 600),
        (1280, 720),
        (1366, 768),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (3440, 1440),
        (320, 240),
    ],
)
def test_uiscale_regions_non_overlapping_across_viewports(vw: int, vh: int) -> None:
    """Regions across standard and extreme viewports never overlap or produce negative rects."""
    scale = UiScale.from_viewport(vw, vh)
    regions = [scale.header, scale.stage, scale.caption, scale.footer]

    assert scale.canvas_rect.width > 0 and scale.canvas_rect.height > 0
    assert scale.canvas_rect.right <= vw
    assert scale.canvas_rect.bottom <= vh

    for r in regions:
        assert r.width >= 1 and r.height >= 1
        assert scale.canvas_rect.contains(r)

    assert sum(r.height for r in regions) == scale.canvas_rect.height

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            assert not regions[i].colliderect(regions[j])


def test_uiscale_1080p_exact_snapping() -> None:
    """UiScale on 1080p viewports produces exact s == 1.0, Rect(0, 0, 1920, 1080), and integers."""
    import pygame

    scale = UiScale.from_viewport(1920, 1080)
    assert scale.s == 1.0
    assert scale.canvas_rect == pygame.Rect(0, 0, 1920, 1080)
    assert scale.px(42) == 42
    assert scale.px(100) == 100
    assert scale.font_px(24) == 24


def test_font_manager_bundled_loading_caching_and_wrapping() -> None:
    """FontManager loads Noto Sans OFL fonts from package assets and caches Font instances."""
    fm = FontManager()
    f1 = fm.get(26, bold=False)
    f2 = fm.get(26, bold=False)
    f3 = fm.get(26, bold=True)

    assert f1 is f2
    assert f1 is not f3

    wrapped = fm.wrap_text(
        "The model processes colored tiles across multiple attention spotlight heads cleanly.",
        max_width=240,
        size_px=24,
    )
    f24 = fm.get(24, bold=False)
    assert len(wrapped) >= 2
    for line in wrapped:
        assert f24.size(line)[0] <= 240

    # Oversized single word after a shorter word is also safely clamped
    long_word_lines = fm.wrap_text(
        "ok supercalifragilisticexpialidociousword", max_width=180, size_px=24
    )
    for line in long_word_lines:
        assert f24.size(line)[0] <= 180

    # Unsupported arrow/newline glyphs fall back cleanly instead of drawing .notdef boxes
    rendered = fm.render_text("↵ → ←", 24, (255, 255, 255))
    assert rendered.get_width() > 0
    assert fm._prepare_text("a↵b") == "a\\nb"
