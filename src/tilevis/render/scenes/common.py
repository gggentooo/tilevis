"""Shared helpers, geometry primitives, and base class for step scenes.

Single responsibility: provide reusable animation lifecycle, TileGrid helpers, and geometric
drawing primitives for dedicated scenes without circular imports.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from tilevis.core.kinds import TensorRole
from tilevis.core.trace import AxisInfo, Step, TensorSnapshot
from tilevis.render.animation import Timeline
from tilevis.render.colormap import Colormap, ColorRGB
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext, Tooltip
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme
from tilevis.render.tiles import TileGrid

UNSIGNED_ROLES: frozenset[str] = frozenset(
    {
        TensorRole.ATTN_WEIGHTS.value,
        TensorRole.PROBS.value,
    }
)

# Colorblind-safe Okabe-Ito-inspired chip accent colors (distinct hues across themes)
_DARK_CHIP_ACCENTS: tuple[ColorRGB, ...] = (
    (66, 133, 214),
    (217, 119, 36),
    (46, 160, 130),
    (184, 108, 178),
    (196, 152, 42),
    (74, 168, 212),
)

_LIGHT_CHIP_ACCENTS: tuple[ColorRGB, ...] = (
    (28, 96, 178),
    (186, 86, 12),
    (18, 128, 100),
    (148, 62, 144),
    (152, 108, 14),
    (20, 124, 176),
)


@dataclass(slots=True)
class HitRegion:
    """Interactive non-grid rectangular zone returning a Tooltip on hover."""

    rect: pygame.Rect
    tooltip_title: str
    tooltip_lines: tuple[str, ...]


@dataclass(slots=True)
class GridCard:
    """Cached card frame containing a titled TileGrid and metadata."""

    title: str
    subtitle: str
    rect: pygame.Rect
    grid: TileGrid
    tooltip_title: str
    role: str = ""
    highlighted: bool = False


class BaseScene:
    """Reusable base implementing Timeline animation, caching, and tooltip hit-testing."""

    DEFAULT_DURATION_SEC: float = 1.0

    def __init__(self) -> None:
        self._step: Step | None = None
        self._ctx: SceneContext | None = None
        self._timeline: Timeline = Timeline(duration=0.0)
        self._cached_stage_size: tuple[int, int] | None = None
        self._cards: list[GridCard] = []
        self._hit_regions: list[HitRegion] = []
        self.downsample_notes: list[str] = []

    def enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        """Prepare scene state and animation timeline cleanly for step."""
        self._step = step
        self._ctx = ctx
        self._cached_stage_size = None
        self._cards = []
        self._hit_regions = []
        self.downsample_notes = []

        speed = max(0.0, float(ctx.animation_speed) * ctx.theme.animation_speed_factor)
        duration = self.DEFAULT_DURATION_SEC if (animate and speed > 0.0) else 0.0
        self._timeline = Timeline(duration=duration, speed=speed)
        self._on_enter(step, ctx, animate=animate)
        if not animate:
            self._timeline.finish()

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        """Subclass hook called during enter() after timeline setup."""

    def update(self, dt: float) -> None:
        """Advance scene timeline monotonically."""
        self._timeline.update(dt)

    def finish(self) -> None:
        """Jump immediately to completed state (progress == 1.0)."""
        self._timeline.finish()

    @property
    def is_animating(self) -> bool:
        """Return True while animation timeline is still advancing."""
        return not self._timeline.is_finished

    def handle_command(self, cmd: Command) -> bool:
        """Consume no extra scene commands by default."""
        _ = cmd
        return False

    def hit_test(self, pos: tuple[int, int]) -> Tooltip | None:
        """Hit-test interactive regions first, then cached TileGrid cards."""
        for region in self._hit_regions:
            if region.rect.collidepoint(pos):
                return Tooltip(title=region.tooltip_title, lines=region.tooltip_lines, pos=pos)
        for card in self._cards:
            tt = grid_tooltip(card.tooltip_title, card.grid, pos)
            if tt is not None:
                return tt
        return None


def find_snapshot(
    step: Step,
    role: TensorRole | str,
    *,
    name_contains: str | None = None,
    index: int = 0,
) -> TensorSnapshot | None:
    """Locate a TensorSnapshot matching role and optional substring/index."""
    matches = step.by_role(role)
    if name_contains is not None:
        low = name_contains.lower()
        filtered = tuple(t for t in matches if low in t.name.lower())
        if filtered:
            return filtered[0] if index >= len(filtered) else filtered[index]
    if 0 <= index < len(matches):
        return matches[index]
    return None


def resolve_colormap(ctx: SceneContext, *, signed: bool) -> Colormap:
    """Resolve signed diverging or unsigned sequential colormap from Theme."""
    cmap_name = ctx.theme.diverging_cmap if signed else ctx.theme.sequential_cmap
    if cmap_name in ctx.colormaps:
        return ctx.colormaps.get(cmap_name)
    fallback = "diverging_blue_orange" if signed else "sequential_purple"
    return ctx.colormaps.get(fallback)


def slice_2d(
    snap: TensorSnapshot, head_idx: int = 0, *, average_heads: bool = False
) -> tuple[np.ndarray, tuple[AxisInfo, ...]]:
    """Return 1-D or 2-D array + matching AxisInfo tuple from snapshot."""
    if snap.data.ndim != 3:
        return snap.data, snap.axes
    h_total = max(1, int(snap.data.shape[0]))
    if average_heads:
        arr2d = np.mean(snap.data, axis=0, dtype=np.float32)
    else:
        arr2d = snap.data[head_idx % h_total]
    axes2 = (snap.axes[1], snap.axes[2]) if len(snap.axes) >= 3 else snap.axes[-2:]
    return arr2d, axes2


def make_grid(
    snap: TensorSnapshot,
    step: Step,
    ctx: SceneContext,
    max_size: tuple[int, int],
    *,
    cell_px: int = 20,
    min_cell_px: int = 6,
    signed: bool | None = None,
    gap_color: ColorRGB | None = None,
    data_override: np.ndarray | None = None,
    axes_override: tuple[AxisInfo, ...] | None = None,
) -> TileGrid:
    """Build a display-fitted TileGrid honoring normalizer and theme settings."""
    use_signed = (snap.role not in UNSIGNED_ROLES) if signed is None else signed
    cmap = resolve_colormap(ctx, signed=use_signed)
    norm_scale = ctx.normalizer.compute_scale(snap, step, ctx.pass_record)
    arr = data_override if data_override is not None else snap.data
    axes = axes_override if axes_override is not None else snap.axes
    return TileGrid(
        data=arr,
        axes=axes,
        colormap=cmap,
        scale=norm_scale,
        cell_px=max(1, int(cell_px)),
        gap_px=ctx.theme.tile_gap_px,
        signed=use_signed,
        min_cell_px=max(1, min(min_cell_px, max_size[0], max_size[1])),
        max_display_size=(max(1, max_size[0]), max(1, max_size[1])),
        gap_color=gap_color if gap_color is not None else ctx.theme.panel,
    )


def grid_tooltip(title: str, grid: TileGrid, pos: tuple[int, int]) -> Tooltip | None:
    """Build formatted hover Tooltip for a TileGrid cell at pixel pos."""
    hit = grid.cell_at(pos)
    if hit is None:
        return None
    row_txt = f'"{hit.row_label}"' if hit.row_label is not None else f"row {hit.orig_row}"
    col_txt = f'"{hit.col_label}"' if hit.col_label is not None else f"dim {hit.orig_col}"
    return Tooltip(
        title=title,
        lines=(f"{row_txt} . {col_txt}", f"value: {hit.value:+.4f}"),
        pos=pos,
    )


def chip_accent(index: int, theme: Theme) -> ColorRGB:
    """Return consistent colorblind-safe chip color adapted for dark vs light theme."""
    is_dark = sum(theme.background) < 384
    palette = _DARK_CHIP_ACCENTS if is_dark else _LIGHT_CHIP_ACCENTS
    return palette[index % len(palette)]


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    theme: Theme,
    scale: UiScale,
    *,
    highlighted: bool = False,
    alt: bool = False,
    border_color: ColorRGB | None = None,
) -> None:
    """Draw rounded panel card background and border inside rect."""
    if rect.width <= 0 or rect.height <= 0:
        return
    radius = max(2, scale.px(theme.corner_radius))
    bg = theme.panel_alt if alt else theme.panel
    bd = (
        border_color
        if border_color is not None
        else (theme.accent if highlighted else theme.border)
    )
    width = max(1, scale.px(2)) if (highlighted or border_color is not None) else 1
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    pygame.draw.rect(surface, bd, rect, width=width, border_radius=radius)


def draw_pill(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    fonts: FontManager,
    font_px: int,
    fg: ColorRGB,
    bg: ColorRGB,
    border: ColorRGB | None = None,
    *,
    bold: bool = True,
    family: str = "NotoSans",
) -> None:
    """Draw a rounded pill badge with centered single-line label."""
    if rect.width <= 2 or rect.height <= 2:
        return
    radius = max(3, min(rect.height // 2, rect.width // 4))
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=1, border_radius=radius)
    txt_surf = fonts.render_text(
        text,
        max(9, font_px),
        fg,
        bold=bold,
        family=family,
        max_width=max(8, rect.width - 8),
    )
    tx = rect.x + max(0, (rect.width - txt_surf.get_width()) // 2)
    ty = rect.y + max(0, (rect.height - txt_surf.get_height()) // 2)
    surface.blit(txt_surf, (tx, ty))


def draw_grid_card(
    surface: pygame.Surface,
    card: GridCard,
    fonts: FontManager,
    theme: Theme,
    scale: UiScale,
    *,
    reveal_cols: int | None = None,
    reveal_rows: int | None = None,
    highlight_rows: tuple[int, ...] = (),
    highlight_cols: tuple[int, ...] = (),
    title_px: int | None = None,
    sub_px: int | None = None,
) -> None:
    """Render GridCard panel, labels, cached TileGrid, and fast O(1) sweep mask."""
    draw_panel(surface, card.rect, theme, scale, highlighted=card.highlighted)
    t_px = title_px or max(11, min(scale.font_px(22, minimum=11), max(11, card.rect.height // 10)))
    s_px = sub_px or max(9, min(scale.font_px(16, minimum=9), t_px - 2))

    pad = max(4, min(scale.px(12, minimum=4), card.rect.width // 16))
    max_tw = max(12, card.rect.width - 2 * pad)
    t_surf = fonts.render_text(
        card.title,
        t_px,
        theme.accent if card.highlighted else theme.text_primary,
        bold=True,
        family=theme.font_family,
        max_width=max_tw,
    )
    surface.blit(t_surf, (card.rect.x + pad, card.rect.y + pad // 2))

    if card.subtitle:
        sub_y = card.rect.y + pad // 2 + t_surf.get_height()
        if sub_y + 8 < card.rect.bottom:
            s_surf = fonts.render_text(
                card.subtitle,
                s_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=max_tw,
            )
            surface.blit(s_surf, (card.rect.x + pad, sub_y))

    grid = card.grid
    grid.draw(
        surface,
        highlight_rows=highlight_rows,
        highlight_cols=highlight_cols,
        highlight_color=theme.highlight,
    )

    # O(1) rectangular sweep mask avoids O(rows * cols) per-cell Python loops on 64x64 grids
    pitch = grid.cell_px + grid.effective_gap_px
    if reveal_cols is not None and reveal_cols < grid.disp_cols:
        vis_c = max(0, reveal_cols)
        hidden_x = grid.rect.x + vis_c * pitch
        hidden_w = grid.rect.right - hidden_x
        if hidden_w > 0 and grid.rect.height > 0:
            pygame.draw.rect(
                surface,
                theme.masked_cell,
                pygame.Rect(hidden_x, grid.rect.y, hidden_w, grid.rect.height),
            )
    if reveal_rows is not None and reveal_rows < grid.disp_rows:
        vis_r = max(0, reveal_rows)
        hidden_y = grid.rect.y + vis_r * pitch
        hidden_h = grid.rect.bottom - hidden_y
        if hidden_h > 0 and grid.rect.width > 0:
            pygame.draw.rect(
                surface,
                theme.masked_cell,
                pygame.Rect(grid.rect.x, hidden_y, grid.rect.width, hidden_h),
            )


def draw_arrow(
    surface: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
    color: ColorRGB,
    thickness: int = 2,
    head_len: int = 10,
) -> None:
    """Draw a crisp geometric line with filled triangular arrowhead from start to end."""
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    length = float(np.hypot(dx, dy))
    if length < 2.0:
        return
    ux, uy = dx / length, dy / length
    eff_head = min(float(head_len), length * 0.55)
    bx, by = ex - ux * eff_head, ey - uy * eff_head
    pygame.draw.line(
        surface, color, (round(sx), round(sy)), (round(bx), round(by)), max(1, thickness)
    )
    px, py = -uy * (eff_head * 0.55), ux * (eff_head * 0.55)
    pts = [
        (round(ex), round(ey)),
        (round(bx + px), round(by + py)),
        (round(bx - px), round(by - py)),
    ]
    pygame.draw.polygon(surface, color, pts)


def draw_operator_symbol(
    surface: pygame.Surface,
    kind: str,
    center: tuple[int, int],
    radius: int,
    color: ColorRGB,
    thickness: int = 3,
) -> None:
    """Draw geometric '+', 'x', or '=' operator centered at center without font glyph reliance."""
    cx, cy = int(center[0]), int(center[1])
    r = max(4, int(radius))
    w = max(2, int(thickness))
    if kind == "+":
        pygame.draw.line(surface, color, (cx - r, cy), (cx + r, cy), w)
        pygame.draw.line(surface, color, (cx, cy - r), (cx, cy + r), w)
    elif kind == "x":
        d = max(3, int(r * 0.72))
        pygame.draw.line(surface, color, (cx - d, cy - d), (cx + d, cy + d), w)
        pygame.draw.line(surface, color, (cx - d, cy + d), (cx + d, cy - d), w)
    elif kind == "=":
        off = max(2, int(r * 0.42))
        pygame.draw.line(surface, color, (cx - r, cy - off), (cx + r, cy - off), w)
        pygame.draw.line(surface, color, (cx - r, cy + off), (cx + r, cy + off), w)


def draw_bezier_arc(
    surface: pygame.Surface,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: ColorRGB,
    thickness: int = 2,
    steps: int = 20,
) -> None:
    """Draw an anti-aliased quadratic Bezier arc from p0 through control p1 to p2."""
    n = max(4, steps)
    pts: list[tuple[int, int]] = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1]
        pts.append((round(x), round(y)))
    if thickness <= 1:
        pygame.draw.aalines(surface, color, False, pts)
    else:
        pygame.draw.lines(surface, color, False, pts, thickness)
        pygame.draw.aalines(surface, color, False, pts)


def blend_rgb(c1: ColorRGB, c2: ColorRGB, t: float) -> ColorRGB:
    """Linearly interpolate between two RGB colors with t clamped to [0, 1]."""
    u = max(0.0, min(1.0, float(t)))
    return (
        round(c1[0] * (1.0 - u) + c2[0] * u),
        round(c1[1] * (1.0 - u) + c2[1] * u),
        round(c1[2] * (1.0 - u) + c2[2] * u),
    )


def place_grid_in_card(
    card_rect: pygame.Rect,
    grid: TileGrid,
    header_h: int,
    pad: int,
) -> None:
    """Center a TileGrid cleanly inside a card below its header height."""
    avail_w = max(1, card_rect.width - 2 * pad)
    avail_h = max(1, card_rect.height - header_h - pad)
    gx = card_rect.x + pad + max(0, (avail_w - grid.rect.width) // 2)
    gy = card_rect.y + header_h + max(0, (avail_h - grid.rect.height) // 2)
    grid.set_position(gx, gy)


def format_token_chip(tok: str) -> str:
    """Convert display token (leading dot '·', newline '↵') to compact readable chip text."""
    return tok.replace("↵", "\\n")
