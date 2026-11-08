"""Generic registry implementation for tilevis extension points.

Single responsibility: manage named extensions via decorators or explicit registration.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from tilevis.core.errors import TilevisError

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    """Generic registry mapping keys to values or factory callables."""

    def __init__(self, name: str = "Registry") -> None:
        self.name = name
        self._entries: dict[K, V] = {}

    def register(self, key: K, *, override: bool = False) -> Callable[[V], V]:
        """Decorator to register a value or class under a key."""

        def decorator(item: V) -> V:
            if key in self._entries and not override:
                raise TilevisError(
                    f"Duplicate registration for key {key!r} in {self.name}. "
                    "Use override=True to replace an existing entry."
                )
            self._entries[key] = item
            return item

        return decorator

    def get(self, key: K, default: V | None = None) -> V:
        """Retrieve a registered item by key, or return default if provided."""
        if key in self._entries:
            return self._entries[key]
        if default is not None:
            return default
        available = ", ".join(repr(k) for k in self._entries) or "none"
        raise KeyError(f"Key {key!r} not found in {self.name}. Available keys: [{available}]")

    def keys(self) -> list[K]:
        """Return a list of all registered keys."""
        return list(self._entries.keys())

    def values(self) -> list[V]:
        """Return a list of all registered values."""
        return list(self._entries.values())

    def items(self) -> list[tuple[K, V]]:
        """Return key-value pairs."""
        return list(self._entries.items())

    def __contains__(self, key: K) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[K]:
        return iter(self._entries)

    def __repr__(self) -> str:
        return f"<{self.name} with {len(self._entries)} entries: {self.keys()!r}>"
