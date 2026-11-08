"""Configuration dataclasses and TOML loading helpers.

Single responsibility: define presenter and recorder settings with defaults.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tilevis.core.errors import TilevisError

logger = logging.getLogger("tilevis.core.config")


def load_toml(path: Path | str) -> dict[str, Any]:
    """Load and parse a TOML file with friendly error messages and utf-8 encoding.

    Args:
        path: Path to TOML file.

    Returns:
        Parsed dictionary.

    Raises:
        TilevisError: If the file does not exist or contains invalid TOML syntax.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise TilevisError(f"Configuration file not found: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        raise TilevisError(f"Invalid TOML format in {file_path.name}: {exc}") from exc
    except Exception as exc:
        raise TilevisError(f"Error reading {file_path.name}: {exc}") from exc


def warn_unknown_keys(data: dict[str, Any], allowed_keys: set[str], section_name: str) -> None:
    """Log a warning if unknown configuration keys are present in a section."""
    for key in data:
        if key not in allowed_keys:
            logger.warning("Unknown configuration key %r found in section [%s]", key, section_name)


@dataclass
class PresenterSettings:
    """Settings controlling window presentation, playlist detail, and keybindings."""

    window_mode: str = "borderless"  # borderless | windowed | exclusive
    window_display: int = 0
    window_fps: int = 60
    theme: str = "dark"
    detail_level: int = 2
    detail_blocks: str | list[int] = "first"  # "first" | "first_last" | "all" | "none" | [1, 2]
    later_pass_level: int = 1
    include_title: bool = True
    include_end: bool = True
    nerd_mode: bool = False
    animation_speed: float = 1.0
    advance_skips_animation: bool = True
    normalizer: str = "shared_step"  # shared_step | per_tensor | global_pass
    locale: str = "en"
    keys: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path | str) -> PresenterSettings:
        """Create PresenterSettings from a TOML configuration file."""
        data = load_toml(path)
        settings = cls()

        # [window]
        if "window" in data and isinstance(data["window"], dict):
            w = data["window"]
            warn_unknown_keys(w, {"mode", "display", "fps"}, "window")
            if "mode" in w:
                settings.window_mode = str(w["mode"])
            if "display" in w:
                settings.window_display = int(w["display"])
            if "fps" in w:
                settings.window_fps = int(w["fps"])

        # [view]
        if "view" in data and isinstance(data["view"], dict):
            v = data["view"]
            warn_unknown_keys(
                v,
                {
                    "theme",
                    "detail_level",
                    "detail_blocks",
                    "later_pass_level",
                    "include_title",
                    "include_end",
                    "nerd_mode",
                    "animation_speed",
                    "advance_skips_animation",
                    "normalizer",
                    "locale",
                },
                "view",
            )
            if "theme" in v:
                settings.theme = str(v["theme"])
            if "detail_level" in v:
                settings.detail_level = int(v["detail_level"])
            if "detail_blocks" in v:
                settings.detail_blocks = v["detail_blocks"]
            if "later_pass_level" in v:
                settings.later_pass_level = int(v["later_pass_level"])
            if "include_title" in v:
                settings.include_title = bool(v["include_title"])
            if "include_end" in v:
                settings.include_end = bool(v["include_end"])
            if "nerd_mode" in v:
                settings.nerd_mode = bool(v["nerd_mode"])
            if "animation_speed" in v:
                settings.animation_speed = float(v["animation_speed"])
            if "advance_skips_animation" in v:
                settings.advance_skips_animation = bool(v["advance_skips_animation"])
            if "normalizer" in v:
                settings.normalizer = str(v["normalizer"])
            if "locale" in v:
                settings.locale = str(v["locale"])

        # [keys]
        if "keys" in data and isinstance(data["keys"], dict):
            settings.keys = {
                k: [str(item) for item in v] if isinstance(v, list) else [str(v)]
                for k, v in data["keys"].items()
            }

        return settings


@dataclass(frozen=True, slots=True)
class RecordSettings:
    """Settings controlling forward recording, reduction limits, and sampling."""

    new_tokens: int = 3
    strategy: str = "greedy"  # greedy | sample
    temperature: float = 0.0
    top_k: int | None = None
    seed: int = 123
    top_k_display: int = 10
    max_features: int = 64
    max_tokens: int = 24
    max_heads: int = 12
    device: str = "cpu"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordSettings:
        """Create RecordSettings from dictionary."""
        kwargs: dict[str, Any] = {}
        for k in (
            "new_tokens",
            "strategy",
            "temperature",
            "top_k",
            "seed",
            "top_k_display",
            "max_features",
            "max_tokens",
            "max_heads",
            "device",
        ):
            if k in data:
                kwargs[k] = data[k]
        return cls(**kwargs)
