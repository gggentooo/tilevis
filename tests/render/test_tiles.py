"""Unit tests for TileGrid rendering, downsampling, and cell_at() hit testing."""

from __future__ import annotations

import numpy as np
import pygame

from tilevis.core.trace import AxisInfo
from tilevis.render.colormap import DIVERGING_BLUE_ORANGE
from tilevis.render.tiles import TileGrid


def test_tilegrid_hit_testing_exact_roundtrip_and_gaps() -> None:
    """cell_at() maps every cell center to (r, c) & AxisInfo.indices, and None in gaps/outside."""
    data = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0
    row_indices = np.array([1, 4, 9], dtype=np.int64)
    col_indices = np.array([2, 10, 22, 31], dtype=np.int64)
    axes = (
        AxisInfo("token", 12, row_indices, ("t1", "t4", "t9")),
        AxisInfo("model_dim", 64, col_indices, None),
    )

    grid = TileGrid(
        data=data,
        axes=axes,
        colormap=DIVERGING_BLUE_ORANGE,
        scale=10.0,
        cell_px=12,
        gap_px=2,
        pos=(40, 60),
    )
    assert not grid.was_downsampled
    assert grid.rect.topleft == (40, 60)

    # Every cell center maps to exact row, col, orig indices, value, and labels
    for r in range(3):
        for c in range(4):
            cx, cy = grid.cell_center(r, c)
            hit = grid.cell_at((cx, cy))
            assert hit is not None
            assert (hit.row, hit.col) == (r, c)
            assert (hit.orig_row, hit.orig_col) == (int(row_indices[r]), int(col_indices[c]))
            assert hit.value == float(data[r, c])
            assert hit.row_label == ("t1", "t4", "t9")[r]

    # Gap pixels between cols 0 and 1 must return None
    gap_x = 40 + 12  # first gap column starts at offset 12
    assert grid.cell_at((gap_x, 60 + 5)) is None
    # Gap pixels between rows 0 and 1 must return None
    gap_y = 60 + 12
    assert grid.cell_at((40 + 5, gap_y)) is None
    # Outside bounds
    assert grid.cell_at((0, 0)) is None
    assert grid.cell_at((grid.rect.right + 1, grid.rect.bottom + 1)) is None


def test_tilegrid_auto_downsampling_preserves_min_size_and_indices() -> None:
    """Oversized arrays auto-downsample to keep tiles legible and note 'showing N of M'."""
    data = np.linspace(-1.0, 1.0, 24 * 64, dtype=np.float32).reshape(24, 64)
    row_ax = AxisInfo("token", 24, np.arange(24, dtype=np.int64))
    col_ax = AxisInfo("model_dim", 128, np.arange(0, 128, 2, dtype=np.int64))

    # Constrain into a small 200x120 stage area where 64 cols cannot fit at 8 px each
    grid = TileGrid(
        data=data,
        axes=(row_ax, col_ax),
        colormap=DIVERGING_BLUE_ORANGE,
        scale=1.0,
        cell_px=16,
        gap_px=1,
        min_cell_px=8,
        max_display_size=(200, 120),
    )

    assert grid.was_downsampled
    assert "showing" in grid.downsample_note
    assert " of " in grid.downsample_note
    assert grid.cell_px >= 8
    assert grid.rect.width <= 200
    assert grid.rect.height <= 120

    # Hit test rightmost displayed tile maps cleanly back into col_ax.indices
    hit = grid.cell_at(grid.cell_center(grid.disp_rows - 1, grid.disp_cols - 1))
    assert hit is not None
    assert hit.orig_col == int(col_ax.indices[-1])


def test_tilegrid_1d_and_highlight_draw() -> None:
    """1-D arrays, tiny gapless cells, highlights, and reveal masks draw cleanly."""
    arr = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    ax = AxisInfo("vocab", 50257, np.array([10, 20, 30], dtype=np.int64), ("a", "b", "c"))
    grid = TileGrid(arr, (ax,), DIVERGING_BLUE_ORANGE, scale=1.0, cell_px=4, gap_px=1, signed=False)
    assert grid.effective_gap_px == 0  # forced to 0 when cell < 6 px

    hit = grid.cell_at(grid.cell_center(0, 1))
    assert hit is not None
    assert hit.orig_col == 20
    assert hit.col_label == "b"

    surf = pygame.Surface((100, 50))
    mask = np.array([[True, False, False]])
    grid.draw(
        surf,
        pos=(10, 10),
        highlight_rows=[0],
        highlight_cols=[1],
        highlight_cells=[(0, 2)],
        reveal_mask=mask,
    )
