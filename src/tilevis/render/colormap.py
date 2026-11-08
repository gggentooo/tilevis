"""Colormap registry and vectorized lookup table mapping.

Single responsibility: map normalized float values to (R, G, B) color arrays.
Allowed dependencies: numpy, tilevis.core, stdlib.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tilevis.core.registry import Registry

logger = logging.getLogger("tilevis.render.colormap")

ColorRGB = tuple[int, int, int]


def _build_diverging_lut(neg: ColorRGB, mid: ColorRGB, pos: ColorRGB) -> np.ndarray:
    """Build a (256, 3) uint8 diverging LUT with exact zero at index 127."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    neg_arr = np.array(neg, dtype=np.float64)
    mid_arr = np.array(mid, dtype=np.float64)
    pos_arr = np.array(pos, dtype=np.float64)

    for idx in range(255):
        if idx <= 127:
            t = (127 - idx) / 127.0
            rgb = mid_arr * (1.0 - t) + neg_arr * t
        else:
            t = (idx - 127) / 127.0
            rgb = mid_arr * (1.0 - t) + pos_arr * t
        lut[idx] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    lut[255] = lut[254]
    lut.setflags(write=False)
    return lut


def _build_sequential_lut(stops: tuple[ColorRGB, ...]) -> np.ndarray:
    """Build a piecewise-linear (256, 3) uint8 sequential LUT across color stops."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    n_seg = len(stops) - 1
    stops_arr = np.array(stops, dtype=np.float64)

    for idx in range(256):
        u = idx / 255.0
        pos = min(u * n_seg, float(n_seg))
        seg = min(int(pos), n_seg - 1)
        frac = pos - seg
        rgb = stops_arr[seg] * (1.0 - frac) + stops_arr[seg + 1] * frac
        lut[idx] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    lut.setflags(write=False)
    return lut


@dataclass(frozen=True, slots=True)
class Colormap:
    """Vectorized 256-entry RGB lookup table colormap."""

    name: str
    lut: np.ndarray
    kind: str = "diverging"  # "diverging" | "sequential"

    def __post_init__(self) -> None:
        arr = np.asarray(self.lut, dtype=np.uint8)
        if arr.shape != (256, 3):
            raise ValueError(f"Colormap LUT must have shape (256, 3), got {arr.shape}")
        arr.setflags(write=False)
        object.__setattr__(self, "lut", arr)

    def map(self, values: np.ndarray | float) -> np.ndarray:
        """Map normalized values onto (..., 3) uint8 RGB colors.

        For diverging colormaps, expects values in [-1, 1] with 0.0 mapped exactly to
        index 127 and strictly odd index reflection around zero.
        For sequential colormaps, expects values in [0, 1] mapped onto indices [0, 255].
        """
        arr = np.asarray(values, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.kind == "sequential":
            clipped = np.clip(arr, 0.0, 1.0)
            idx = np.rint(clipped * 255.0).astype(np.intp)
            idx = np.clip(idx, 0, 255)
        else:
            clipped = np.clip(arr, -1.0, 1.0)
            # Multiply before shifting by 127 so rint(-x) == -rint(x) holds bit-for-bit.
            offset = np.rint(clipped * 127.0).astype(np.intp)
            idx = np.clip(127 + offset, 0, 254)

        return np.asarray(self.lut[idx], dtype=np.uint8)


COLORMAPS: Registry[str, Colormap] = Registry("COLORMAPS")

DIVERGING_BLUE_ORANGE = Colormap(
    name="diverging_blue_orange",
    kind="diverging",
    lut=_build_diverging_lut(
        neg=(30, 110, 188),
        mid=(242, 243, 245),
        pos=(224, 96, 12),
    ),
)

SEQUENTIAL_PURPLE = Colormap(
    name="sequential_purple",
    kind="sequential",
    lut=_build_sequential_lut(
        (
            (244, 241, 249),
            (188, 181, 224),
            (126, 102, 184),
            (63, 18, 125),
        )
    ),
)

DIVERGING_TEAL_MAGENTA = Colormap(
    name="diverging_teal_magenta",
    kind="diverging",
    lut=_build_diverging_lut(
        neg=(0, 147, 146),
        mid=(242, 243, 245),
        pos=(207, 89, 126),
    ),
)

COLORMAPS.register("diverging_blue_orange")(DIVERGING_BLUE_ORANGE)
COLORMAPS.register("sequential_purple")(SEQUENTIAL_PURPLE)
COLORMAPS.register("diverging_teal_magenta")(DIVERGING_TEAL_MAGENTA)


def get_colormap(name: str) -> Colormap:
    """Retrieve colormap by name, falling back to 'diverging_blue_orange'."""
    key = name.strip().lower()
    if key in COLORMAPS:
        return COLORMAPS.get(key)
    logger.warning("Unknown colormap %r; falling back to 'diverging_blue_orange'", name)
    return DIVERGING_BLUE_ORANGE
