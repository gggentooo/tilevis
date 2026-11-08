"""TileGrid rendering, surface caching, and cell hit-testing.

Single responsibility: convert numerical arrays into cached colored pygame surfaces.
Allowed dependencies: pygame, numpy, tilevis.core, tilevis.render, stdlib.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pygame

from tilevis.core.trace import AxisInfo
from tilevis.render.colormap import Colormap, ColorRGB


@dataclass(frozen=True, slots=True)
class TileHit:
    """Result of hit-testing a pixel coordinate inside a TileGrid."""

    row: int
    col: int
    orig_row: int
    orig_col: int
    value: float
    row_label: str | None = None
    col_label: str | None = None


def _downsample_indices(full_len: int, keep_len: int) -> np.ndarray:
    """Select strictly increasing indices in [0, full_len) for display downsampling."""
    if keep_len >= full_len:
        return np.arange(full_len, dtype=np.int64)
    if keep_len <= 1:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, full_len - 1, keep_len, dtype=np.int64)


class TileGrid:
    """Cached colored-tile matrix with exact hit-testing and highlight overlays."""

    def __init__(
        self,
        data: np.ndarray,
        axes: tuple[AxisInfo, ...],
        colormap: Colormap,
        scale: float,
        cell_px: int = 16,
        gap_px: int = 1,
        *,
        signed: bool = True,
        min_cell_px: int = 8,
        max_display_size: tuple[int, int] | None = None,
        gap_color: ColorRGB = (26, 31, 44),
        pos: tuple[int, int] = (0, 0),
    ) -> None:
        raw = np.asarray(data, dtype=np.float32)
        if raw.ndim == 1:
            if len(axes) != 1:
                raise ValueError(f"1-D data requires 1 AxisInfo, got {len(axes)}")
            arr2d = raw.reshape(1, -1)
            row_axis: AxisInfo | None = None
            col_axis = axes[0]
        elif raw.ndim == 2:
            if len(axes) != 2:
                raise ValueError(f"2-D data requires 2 AxisInfo descriptors, got {len(axes)}")
            arr2d = raw
            row_axis = axes[0]
            col_axis = axes[1]
        else:
            raise ValueError(f"TileGrid supports only 1-D or 2-D arrays, got ndim={raw.ndim}")

        orig_rows, orig_cols = int(arr2d.shape[0]), int(arr2d.shape[1])
        if orig_rows <= 0 or orig_cols <= 0:
            raise ValueError(f"TileGrid cannot render empty shape {arr2d.shape}")

        safe_scale = float(scale) if np.isfinite(scale) and scale > 1e-12 else 1.0
        req_cell = max(1, int(cell_px))
        req_gap = max(0, int(gap_px))
        eff_min_cell = max(1, int(min_cell_px))

        # Fit or display-downsample to fit within max_display_size when constrained.
        sel_r = np.arange(orig_rows, dtype=np.int64)
        sel_c = np.arange(orig_cols, dtype=np.int64)
        chosen_cell = req_cell

        if max_display_size is not None:
            avail_w = max(1, int(max_display_size[0]))
            avail_h = max(1, int(max_display_size[1]))

            fit_w = (avail_w - max(0, orig_cols - 1) * req_gap) // orig_cols
            fit_h = (avail_h - max(0, orig_rows - 1) * req_gap) // orig_rows
            fit_cell = min(req_cell, fit_w, fit_h)

            local_min = max(1, min(eff_min_cell, avail_w, avail_h))
            if fit_cell < local_min:
                probe_gap = req_gap if local_min >= 6 else 0
                max_cols = max(1, (avail_w + probe_gap) // (local_min + probe_gap))
                max_rows = max(1, (avail_h + probe_gap) // (local_min + probe_gap))
                sel_r = _downsample_indices(orig_rows, min(orig_rows, max_rows))
                sel_c = _downsample_indices(orig_cols, min(orig_cols, max_cols))

                disp_r = len(sel_r)
                disp_c = len(sel_c)
                fit_w2 = (avail_w - max(0, disp_c - 1) * probe_gap) // disp_c
                fit_h2 = (avail_h - max(0, disp_r - 1) * probe_gap) // disp_r
                chosen_cell = max(1, min(req_cell, fit_w2, fit_h2))
            else:
                chosen_cell = max(local_min, fit_cell)

        self.cell_px: int = int(chosen_cell)
        self.effective_gap_px: int = req_gap if self.cell_px >= 6 else 0
        self.signed: bool = signed
        self.scale: float = safe_scale
        self.colormap: Colormap = colormap
        self.gap_color: ColorRGB = gap_color

        # Sub-select display slice while preserving exact original axis indices.
        disp_data = np.asarray(arr2d[np.ix_(sel_r, sel_c)], dtype=np.float32)
        disp_data.setflags(write=False)
        self.disp_data: np.ndarray = disp_data
        self.disp_rows, self.disp_cols = disp_data.shape
        self.orig_rows = orig_rows
        self.orig_cols = orig_cols

        self.row_orig_indices: np.ndarray = (
            np.asarray(row_axis.indices[sel_r], dtype=np.int64)
            if row_axis is not None
            else np.array([0], dtype=np.int64)
        )
        self.col_orig_indices: np.ndarray = np.asarray(col_axis.indices[sel_c], dtype=np.int64)

        self.row_labels: tuple[str, ...] | None = (
            tuple(row_axis.labels[int(i)] for i in sel_r)
            if row_axis is not None and row_axis.labels is not None
            else None
        )
        self.col_labels: tuple[str, ...] | None = (
            tuple(col_axis.labels[int(i)] for i in sel_c) if col_axis.labels is not None else None
        )

        # Downsampling description if display resolution had to drop rows/cols
        self.was_downsampled: bool = (self.disp_rows < orig_rows) or (self.disp_cols < orig_cols)
        self.downsample_note: str = self._format_downsample_note(
            orig_rows, orig_cols, self.disp_rows, self.disp_cols
        )

        # Build cached pixel surface
        pixel_w = self.disp_cols * self.cell_px + max(0, self.disp_cols - 1) * self.effective_gap_px
        pixel_h = self.disp_rows * self.cell_px + max(0, self.disp_rows - 1) * self.effective_gap_px
        self.rect: pygame.Rect = pygame.Rect(pos[0], pos[1], pixel_w, pixel_h)
        self._surface: pygame.Surface = self._build_surface()

    @staticmethod
    def _format_downsample_note(
        orig_rows: int, orig_cols: int, disp_rows: int, disp_cols: int
    ) -> str:
        """Build human-readable downsampling indicator containing 'showing N of M'."""
        parts: list[str] = []
        if disp_rows < orig_rows:
            parts.append(f"showing {disp_rows} of {orig_rows} rows")
        if disp_cols < orig_cols:
            if parts:
                parts.append(f"{disp_cols} of {orig_cols} cols")
            else:
                parts.append(f"showing {disp_cols} of {orig_cols}")
        return ", ".join(parts)

    def _build_surface(self) -> pygame.Surface:
        """Render the cached Surface from RGB-mapped cells using vectorized NumPy."""
        clean = np.nan_to_num(self.disp_data, nan=0.0)
        norm = (
            np.clip(clean / self.scale, -1.0, 1.0)
            if self.signed
            else np.clip(clean / self.scale, 0.0, 1.0)
        )
        rgb_cells = self.colormap.map(norm)  # (disp_rows, disp_cols, 3) uint8

        # Compact path for tiny cells (< 6 px, no gap) via nearest-neighbor scale
        if self.effective_gap_px == 0:
            base = pygame.surfarray.make_surface(rgb_cells.transpose(1, 0, 2))
            if self.cell_px == 1:
                return base
            return pygame.transform.scale(
                base, (self.disp_cols * self.cell_px, self.disp_rows * self.cell_px)
            )

        # Gap-visible path built via C-speed NumPy block broadcast
        pitch = self.cell_px + self.effective_gap_px
        buf = np.empty((self.disp_rows, pitch, self.disp_cols, pitch, 3), dtype=np.uint8)
        buf[:] = self.gap_color
        buf[:, : self.cell_px, :, : self.cell_px, :] = rgb_cells[:, None, :, None, :]
        flat = buf.reshape(self.disp_rows * pitch, self.disp_cols * pitch, 3)
        cropped = flat[: self.rect.height, : self.rect.width, :]
        return pygame.surfarray.make_surface(cropped.transpose(1, 0, 2))

    @property
    def surface(self) -> pygame.Surface:
        """Return cached Surface representing the tile grid."""
        return self._surface

    def set_position(self, x: int, y: int) -> None:
        """Move top-left coordinates without regenerating cached surface."""
        self.rect.topleft = (int(x), int(y))

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Return viewport pixel Rect of display cell (row, col)."""
        pitch = self.cell_px + self.effective_gap_px
        return pygame.Rect(
            self.rect.x + col * pitch,
            self.rect.y + row * pitch,
            self.cell_px,
            self.cell_px,
        )

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        """Return pixel center of display cell (row, col)."""
        return self.cell_rect(row, col).center

    def cell_at(self, pos: tuple[int, int]) -> TileHit | None:
        """Hit-test screen coordinates, mapping back to exact original AxisInfo.indices.

        Returns None when coordinates lie in a gap between tiles or outside the grid.
        """
        px_x, px_y = int(pos[0]), int(pos[1])
        if not self.rect.collidepoint(px_x, px_y):
            return None

        rel_x = px_x - self.rect.x
        rel_y = px_y - self.rect.y
        pitch = self.cell_px + self.effective_gap_px

        col = rel_x // pitch
        row = rel_y // pitch

        if col < 0 or col >= self.disp_cols or row < 0 or row >= self.disp_rows:
            return None

        if self.effective_gap_px > 0 and (
            (rel_x % pitch) >= self.cell_px or (rel_y % pitch) >= self.cell_px
        ):
            return None

        row_lbl = self.row_labels[row] if self.row_labels is not None else None
        col_lbl = self.col_labels[col] if self.col_labels is not None else None

        return TileHit(
            row=int(row),
            col=int(col),
            orig_row=int(self.row_orig_indices[row]),
            orig_col=int(self.col_orig_indices[col]),
            value=float(self.disp_data[row, col]),
            row_label=row_lbl,
            col_label=col_lbl,
        )

    def draw(
        self,
        surface: pygame.Surface,
        *,
        pos: tuple[int, int] | None = None,
        highlight_rows: Iterable[int] = (),
        highlight_cols: Iterable[int] = (),
        highlight_cells: Iterable[tuple[int, int]] = (),
        reveal_mask: np.ndarray | None = None,
        highlight_color: ColorRGB = (255, 200, 87),
        mask_color: ColorRGB = (48, 54, 68),
    ) -> None:
        """Blit cached tiles onto surface along with optional mask and highlights."""
        if pos is not None:
            self.set_position(pos[0], pos[1])

        surface.blit(self._surface, self.rect.topleft)

        # Mask unrevealed cells
        if reveal_mask is not None:
            mask = np.asarray(reveal_mask, dtype=bool)
            if mask.shape == (self.disp_rows, self.disp_cols):
                hidden_coords = np.argwhere(~mask)
                for r, c in hidden_coords:
                    pygame.draw.rect(surface, mask_color, self.cell_rect(int(r), int(c)))

        pitch = self.cell_px + self.effective_gap_px
        stroke = max(1, min(3, self.cell_px // 5))

        for r in highlight_rows:
            if 0 <= r < self.disp_rows:
                row_rect = pygame.Rect(
                    self.rect.x - 1,
                    self.rect.y + r * pitch - 1,
                    self.rect.width + 2,
                    self.cell_px + 2,
                )
                pygame.draw.rect(surface, highlight_color, row_rect, width=stroke)

        for c in highlight_cols:
            if 0 <= c < self.disp_cols:
                col_rect = pygame.Rect(
                    self.rect.x + c * pitch - 1,
                    self.rect.y - 1,
                    self.cell_px + 2,
                    self.rect.height + 2,
                )
                pygame.draw.rect(surface, highlight_color, col_rect, width=stroke)

        for r, c in highlight_cells:
            if 0 <= r < self.disp_rows and 0 <= c < self.disp_cols:
                pygame.draw.rect(
                    surface,
                    highlight_color,
                    self.cell_rect(r, c).inflate(2, 2),
                    width=stroke,
                )
