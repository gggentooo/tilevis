"""Unit tests for generic Registry implementation."""

from __future__ import annotations

import pytest

from tilevis.core.errors import TilevisError
from tilevis.core.registry import Registry


def test_registry_register_and_get() -> None:
    reg: Registry[str, int] = Registry("TestRegistry")

    @reg.register("alpha")
    def alpha() -> int:
        return 1

    assert reg.get("alpha") is alpha
    assert "alpha" in reg
    assert reg.keys() == ["alpha"]
    assert len(reg) == 1


def test_registry_duplicate_rejection() -> None:
    reg: Registry[str, str] = Registry("PluginRegistry")
    reg.register("plugin_a")("First")

    msg = "Duplicate registration for key 'plugin_a' in PluginRegistry"
    with pytest.raises(TilevisError, match=msg):
        reg.register("plugin_a")("Second")


def test_registry_override() -> None:
    reg: Registry[str, str] = Registry("OverrideRegistry")
    reg.register("key_1")("First")
    reg.register("key_1", override=True)("Overwritten")

    assert reg.get("key_1") == "Overwritten"


def test_registry_missing_key_helpful_error() -> None:
    reg: Registry[str, str] = Registry("ColorRegistry")
    reg.register("red")("#ff0000")
    reg.register("green")("#00ff00")

    pattern = r"Key 'blue' not found in ColorRegistry\. Available keys: \['red', 'green'\]"
    with pytest.raises(KeyError, match=pattern):
        reg.get("blue")

    # If default is provided, returns default without raising
    assert reg.get("blue", default="#0000ff") == "#0000ff"


def test_registry_iteration_and_methods() -> None:
    reg: Registry[str, int] = Registry("NumberRegistry")
    reg.register("one")(1)
    reg.register("two")(2)

    assert list(reg) == ["one", "two"]
    assert reg.values() == [1, 2]
    assert reg.items() == [("one", 1), ("two", 2)]
    assert "NumberRegistry" in repr(reg)
