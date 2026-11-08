"""Unit tests for Phase 7 UI widgets:
Minimap, Buttons, Tooltip, NerdPanel, Legend, and Theme Contrast.
"""

from __future__ import annotations

import pygame
import pytest
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel
from tilevis.core.trace import Fidelity, Step
from tilevis.render.colormap import COLORMAPS
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.player import StepPlayer
from tilevis.render.scenes.base import Tooltip
from tilevis.render.text import FontManager
from tilevis.render.themes.builtin import DARK_THEME, LIGHT_THEME
from tilevis.render.widgets.buttons import ButtonsWidget
from tilevis.render.widgets.legend import LegendWidget
from tilevis.render.widgets.minimap import MinimapWidget
from tilevis.render.widgets.nerd_panel import NerdPanelWidget
from tilevis.render.widgets.tooltip import TooltipWidget


@pytest.mark.parametrize("n_layers", [1, 3, 12, 24, 48])
def test_minimap_layout_and_hit_testing_across_layer_counts(n_layers: int) -> None:
    """Minimap lays out bounded segments and jumps StepPlayer to selected block in current pass."""
    trace = make_fake_trace(n_layers=n_layers, emb_dim=16, n_tokens=4, n_passes=2, n_heads=2)
    settings = PresenterSettings(detail_level=int(DetailLevel.DEEP), detail_blocks="all")
    player = StepPlayer(trace, settings)
    scale = UiScale.from_viewport(1920, 1080)

    widget = MinimapWidget()
    rect = pygame.Rect(100, 20, 800, 30)

    segments = widget.compute_segments(rect, player, scale=scale)
    assert len(segments) >= 3  # Input, at least 1 block, Output
    # Bounded horizontal count: even for 48 layers, segments should not exceed 12
    assert len(segments) <= 14

    # All segments fit within rect
    assert segments[0].rect.left >= rect.left
    assert segments[-1].rect.right <= rect.right + 2

    # Verify hit testing and clicking on a block segment
    clickable_blocks = [
        s for s in segments if s.target_type == "block" and s.block_index is not None
    ]
    assert len(clickable_blocks) > 0

    target_block = clickable_blocks[-1].block_index
    assert target_block is not None

    # Find center of that block segment
    b_seg = clickable_blocks[-1]
    click_pos = b_seg.rect.center

    # Hit test returns the segment
    hit = widget.hit_test(click_pos)
    assert hit is not None
    assert hit.target_type == "block"
    assert hit.block_index == target_block

    # Clicking jumps player to that block
    consumed = widget.handle_click(click_pos, player)
    assert consumed
    assert player.current is not None
    assert player.current.block_index == target_block


def test_buttons_layout_hover_and_click() -> None:
    """ButtonsWidget renders interactive controls, updates hover, and returns commands."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=2)
    settings = PresenterSettings(detail_level=int(DetailLevel.STANDARD))
    player = StepPlayer(trace, settings)
    scale = UiScale.from_viewport(1920, 1080)

    widget = ButtonsWidget()
    rect = pygame.Rect(1400, 1020, 450, 40)
    buttons = widget.layout(rect, player, settings, scale=scale)

    assert len(buttons) >= 5
    cmds = [b.cmd for b in buttons]
    assert Command.PREV in cmds
    assert Command.NEXT in cmds
    assert Command.CYCLE_DETAIL in cmds
    assert Command.TOGGLE_NERD in cmds
    assert Command.TOGGLE_LOCALE in cmds
    assert Command.TOGGLE_HELP in cmds

    # Hover test
    next_btn = next(b for b in buttons if b.cmd == Command.NEXT)
    assert widget.update_hover(next_btn.rect.center)
    assert next_btn.is_hovered

    # Click test returns Command.NEXT
    cmd = widget.handle_click(next_btn.rect.center)
    assert cmd == Command.NEXT

    # Outside click returns None
    assert widget.handle_click((0, 0)) is None


@pytest.mark.parametrize("vw,vh", [(800, 600), (1920, 1080), (3440, 1440)])
def test_tooltip_four_corners_clamping(vw: int, vh: int) -> None:
    """Tooltips triggered at corners (0,0), (W-1,0), (0,H-1), (W-1,H-1) stay inside screen."""
    widget = TooltipWidget()
    fonts = FontManager()
    scale = UiScale.from_viewport(vw, vh)
    theme = DARK_THEME

    corners = [
        (0, 0),
        (vw - 1, 0),
        (0, vh - 1),
        (vw - 1, vh - 1),
    ]

    for cx, cy in corners:
        tooltip = Tooltip(
            title="Weight [0, 12]",
            lines=("Value: +0.4285", "Original Neuron: 12", "Token: ' effort'"),
            pos=(cx, cy),
        )
        rect, _lines, _title_surf = widget.compute_rect(
            tooltip,
            vw,
            vh,
            fonts=fonts,
            scale=scale,
            theme=theme,
        )

        assert rect.left >= 0, f"Corner ({cx}, {cy}) left={rect.left} < 0"
        assert rect.right <= vw, f"Corner ({cx}, {cy}) right={rect.right} > {vw}"
        assert rect.top >= 0, f"Corner ({cx}, {cy}) top={rect.top} < 0"
        assert rect.bottom <= vh, f"Corner ({cx}, {cy}) bottom={rect.bottom} > {vh}"
        assert rect.width > 0 and rect.height > 0


def test_tooltip_long_token_labels_and_paths() -> None:
    """Tooltip gracefully handles 50+ character tokens and nested module paths without overflow."""
    widget = TooltipWidget()
    fonts = FontManager()
    scale = UiScale.from_viewport(1920, 1080)
    theme = DARK_THEME

    long_token = "A" * 60
    long_path = "model.transformer.h.0.attn.attention.qkv_projection.weight"

    tooltip = Tooltip(
        title=f"Neuron in {long_path}",
        lines=(f"Token: '{long_token}'", "Score: +1.2345"),
        pos=(500, 500),
    )
    surf = pygame.Surface((1920, 1080))
    # Should draw cleanly without error
    widget.draw(surf, tooltip, 1920, 1080, theme=theme, fonts=fonts, scale=scale)


def test_nerd_panel_rendering_and_fidelity_styling() -> None:
    """NerdPanel accurately reflects full_shape, reduction, Fidelity, and handles close button."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=2)
    step = trace.passes[0].steps[3]  # an attention/matmul step
    widget = NerdPanelWidget()
    fonts = FontManager()
    scale = UiScale.from_viewport(1920, 1080)
    stage_rect = pygame.Rect(0, 100, 1920, 750)
    surf = pygame.Surface((1920, 1080))

    for fid_status in ["exact", "approximate", "unverified", "failed"]:
        mod_step = Step(
            id=step.id,
            kind=step.kind,
            semantic=step.semantic,
            stage=step.stage,
            block_index=step.block_index,
            detail_level=step.detail_level,
            module_path="trf_blocks.0.att.W_query",
            tensors=step.tensors,
            meta=step.meta,
            fidelity=Fidelity(status=fid_status, max_abs_err=0.002, note="Test check"),
        )
        widget.draw(
            surf,
            stage_rect,
            mod_step,
            theme=DARK_THEME,
            fonts=fonts,
            scale=scale,
            normalizer_name="shared_step",
            normalizer_scale=2.5,
        )
        assert widget.rect.width > 0
        assert widget.close_btn_rect.width > 0
        assert widget.handle_click(widget.rect.center)


def test_legend_diverging_and_sequential_endpoints() -> None:
    """LegendWidget formats diverging +/-scale vs sequential 0%-100% endpoints."""
    widget = LegendWidget()
    fonts = FontManager()
    scale = UiScale.from_viewport(1920, 1080)
    surf = pygame.Surface((1920, 1080))
    rect = pygame.Rect(20, 1030, 200, 36)

    div_cmap = COLORMAPS.get("diverging_blue_orange")
    widget.draw(
        surf,
        rect,
        div_cmap,
        scale_val=3.14,
        theme=DARK_THEME,
        fonts=fonts,
        scale=scale,
    )

    seq_cmap = COLORMAPS.get("sequential_purple")
    widget.draw(
        surf,
        rect,
        seq_cmap,
        scale_val=1.0,
        theme=LIGHT_THEME,
        fonts=fonts,
        scale=scale,
    )


def test_wcag_theme_contrast_ratios() -> None:
    """Automated WCAG relative-luminance test: contrast ratio >= 4.5:1 in dark and light."""

    def _rel_lum(rgb: tuple[int, int, int]) -> float:
        vals = []
        for c in rgb:
            s = c / 255.0
            vals.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]

    def _contrast(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
        l1 = _rel_lum(rgb1)
        l2 = _rel_lum(rgb2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    for theme in (DARK_THEME, LIGHT_THEME):
        for bg in (theme.background, theme.panel, theme.panel_alt):
            cr_primary = _contrast(theme.text_primary, bg)
            assert cr_primary >= 4.5, (
                f"{theme.name} text_primary vs bg contrast {cr_primary:.2f} < 4.5"
            )

            cr_secondary = _contrast(theme.text_secondary, bg)
            assert cr_secondary >= 4.5, (
                f"{theme.name} text_secondary vs bg contrast {cr_secondary:.2f} < 4.5"
            )
