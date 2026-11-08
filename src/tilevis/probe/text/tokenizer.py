"""Tokenizer protocol and TiktokenTokenizer implementation.

Single responsibility: token encoding, decoding, and display formatting (·, ↵, <0xAB>).
Allowed dependencies: stdlib, tiktoken, tilevis.core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import tiktoken

from tilevis.core.errors import ModelLoadError


class Tokenizer(Protocol):
    """Protocol for text tokenizers used by probe and recorder."""

    name: str
    eos_id: int | None

    def encode(self, text: str) -> list[int]:
        """Convert string text into a sequence of integer token IDs."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert a sequence of token IDs back into string text."""
        ...

    def token_display(self, token_id: int) -> str:
        """Format an individual token ID for UI display.

        Leading spaces are displayed as '·', newlines as '↵', and partial/invalid
        UTF-8 bytes as '<0xAB>'.
        """
        ...


class TiktokenTokenizer:
    """GPT-2 BPE tokenizer wrapper around tiktoken."""

    def __init__(self, name: str = "gpt2") -> None:
        self.name = name
        try:
            self._enc = tiktoken.get_encoding(name)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load tiktoken encoding {name!r}: {exc}") from exc

        self.eos_id: int | None = 50256 if name == "gpt2" else getattr(self._enc, "eot_token", None)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs allowing '<|endoftext|>' special token."""
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs into string text."""
        return self._enc.decode(token_ids)

    def token_display(self, token_id: int) -> str:
        """Format single token ID into display string."""
        try:
            raw_b = self._enc.decode_single_token_bytes(token_id)
        except KeyError:
            return f"<{token_id}>"

        try:
            s = raw_b.decode("utf-8")
            s = s.replace("\n", "↵")
            if s.startswith(" "):
                s = "·" + s[1:]
            return s
        except UnicodeDecodeError:
            return "".join(f"<0x{b:02X}>" for b in raw_b)

    def validate_token_ids(self, token_ids: Sequence[int], vocab_size: int) -> None:
        """Validate that all token IDs fall within the model's vocabulary."""
        validate_token_ids(token_ids, vocab_size)


def validate_token_ids(token_ids: Sequence[int], vocab_size: int) -> None:
    """Validate that all token IDs fall within the model's vocabulary.

    Args:
        token_ids: List of encoded token IDs.
        vocab_size: Model embedding matrix vocabulary size.

    Raises:
        ModelLoadError: If any token ID >= vocab_size or < 0.
    """
    for tid in token_ids:
        if tid >= vocab_size or tid < 0:
            raise ModelLoadError(
                f"Token ID {tid} exceeds model vocabulary size ({vocab_size}). "
                "Tokenizer and model vocabulary mismatch."
            )
