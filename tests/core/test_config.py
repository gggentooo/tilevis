"""Unit tests for configuration dataclasses and TOML loading."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tilevis.core.config import PresenterSettings, RecordSettings, load_toml, warn_unknown_keys
from tilevis.core.errors import TilevisError


def test_presenter_settings_defaults() -> None:
    settings = PresenterSettings()
    assert settings.window_mode == "borderless"
    assert settings.window_fps == 60
    assert settings.detail_level == 2
    assert settings.detail_blocks == "first"
    assert settings.theme == "dark"
    assert settings.animation_speed == 1.0


def test_record_settings_defaults() -> None:
    settings = RecordSettings()
    assert settings.new_tokens == 3
    assert settings.strategy == "greedy"
    assert settings.seed == 123
    assert settings.max_features == 64
    assert settings.max_tokens == 24


def test_presenter_settings_from_toml(tmp_path: Path) -> None:
    toml_path = tmp_path / "custom.toml"
    toml_path.write_text(
        """
[window]
mode = "windowed"
display = 1
fps = 30

[view]
theme = "light"
detail_level = 1
detail_blocks = "all"
animation_speed = 2.0
nerd_mode = true

[keys]
next = ["space", "right"]
""",
        encoding="utf-8",
    )

    settings = PresenterSettings.from_toml(toml_path)
    assert settings.window_mode == "windowed"
    assert settings.window_display == 1
    assert settings.window_fps == 30
    assert settings.theme == "light"
    assert settings.detail_level == 1
    assert settings.detail_blocks == "all"
    assert settings.animation_speed == 2.0
    assert settings.nerd_mode is True
    assert settings.keys["next"] == ["space", "right"]


def test_load_toml_errors(tmp_path: Path) -> None:
    # Missing file
    with pytest.raises(TilevisError, match="Configuration file not found"):
        load_toml(tmp_path / "does_not_exist.toml")

    # Malformed syntax
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("[section\nkey = ", encoding="utf-8")
    with pytest.raises(TilevisError, match="Invalid TOML format"):
        load_toml(bad_toml)


def test_warn_unknown_keys(caplog: pytest.LogCaptureFixture) -> None:
    data = {"known_a": 1, "unrecognized_key": 2}
    with caplog.at_level(logging.WARNING):
        warn_unknown_keys(data, {"known_a"}, "my_section")

    assert (
        "Unknown configuration key 'unrecognized_key' found in section [my_section]" in caplog.text
    )
