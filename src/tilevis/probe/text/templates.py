"""Prompt template formatting and registry.

Single responsibility: format user input prompts into raw or instruction-tuned templates.
Allowed dependencies: stdlib, tilevis.core.
"""

from __future__ import annotations

from typing import Protocol

from tilevis.core.registry import Registry


class PromptTemplate(Protocol):
    """Protocol for prompt formatting templates."""

    name: str

    def format(self, prompt: str) -> str:
        """Apply template to a raw prompt string."""
        ...


class RawPromptTemplate:
    """Identity template returning the unmodified prompt."""

    name = "raw"

    def format(self, prompt: str) -> str:
        return prompt


class AlpacaPromptTemplate:
    """Alpaca-style instruction tuning template matching ch07 gpt_instruction_finetuning."""

    name = "alpaca"

    def format(self, prompt: str) -> str:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{prompt}"
            "\n\n### Response:\n"
        )


TEMPLATES: Registry[str, PromptTemplate] = Registry("templates")
TEMPLATES.register("raw")(RawPromptTemplate())
TEMPLATES.register("alpaca")(AlpacaPromptTemplate())
