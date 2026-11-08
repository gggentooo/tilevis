"""Built-in colorblind-safe themes (dark, light).

Single responsibility: provide default dark and high-contrast light presentation themes.
Allowed dependencies: tilevis.core, tilevis.render, stdlib.
"""

from __future__ import annotations

import logging

from tilevis.core.registry import Registry
from tilevis.render.themes.base import Theme

logger = logging.getLogger("tilevis.render.themes")

THEMES: Registry[str, Theme] = Registry("THEMES")

DARK_THEME = Theme(
    name="dark",
    background=(15, 18, 25),
    panel=(26, 31, 44),
    panel_alt=(36, 43, 60),
    border=(68, 80, 108),
    text_primary=(240, 244, 250),
    text_secondary=(176, 188, 210),
    accent=(86, 180, 233),
    accent_muted=(45, 92, 128),
    masked_cell=(48, 54, 68),
    highlight=(255, 200, 87),
    success=(92, 214, 141),
    warning=(246, 193, 119),
    error=(235, 111, 146),
    confetti=True,
)

LIGHT_THEME = Theme(
    name="light",
    background=(244, 246, 250),
    panel=(255, 255, 255),
    panel_alt=(232, 236, 244),
    border=(185, 195, 212),
    text_primary=(20, 25, 36),
    text_secondary=(68, 76, 92),
    accent=(0, 114, 178),
    accent_muted=(178, 214, 238),
    masked_cell=(210, 214, 222),
    highlight=(217, 119, 6),
    success=(22, 163, 74),
    warning=(202, 138, 4),
    error=(220, 38, 38),
    confetti=True,
)

THEMES.register("dark")(DARK_THEME)
THEMES.register("light")(LIGHT_THEME)


def get_theme(name: str) -> Theme:
    """Retrieve theme by name, falling back to 'dark' if unrecognized."""
    key = name.strip().lower()
    if key in THEMES:
        return THEMES.get(key)
    logger.warning("Unknown theme %r; falling back to 'dark'", name)
    return THEMES.get("dark")
