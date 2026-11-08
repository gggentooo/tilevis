"""Step builder protocol and context definitions.

Single responsibility: define the common protocol and execution context for constructing
Trace Step records from captured model activations and weights.
Allowed dependencies: stdlib, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch.nn as nn

from tilevis.core.registry import Registry
from tilevis.core.trace import Prediction, Step
from tilevis.probe.adapters.base import ModelSpec
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.reduce import IndexPlan
from tilevis.probe.text.tokenizer import Tokenizer


@dataclass
class StepBuildContext:
    """Execution context provided to step builders for one forward pass."""

    pass_index: int
    input_ids: tuple[int, ...]
    input_tokens: tuple[str, ...]
    text_before: str
    tokenizer: Tokenizer
    step_id_prefix: str = ""
    prediction: Prediction | None = None
    pass_meta: dict[str, Any] = field(default_factory=dict)
    _counter: int = 0

    def next_step_id(self, suffix: str) -> str:
        """Generate a deterministic, unique step id."""
        return f"p{self.pass_index}.{suffix}"


class StepBuilder(Protocol):
    """Protocol for building Step records from captured forward activations."""

    def build(
        self,
        spec: ModelSpec,
        capture: CaptureBuffer,
        model: nn.Module,
        plan: IndexPlan,
        ctx: StepBuildContext,
        **kwargs: Any,
    ) -> list[Step]:
        """Construct one or more Step objects for this calculation stage."""
        ...


STEP_BUILDERS: Registry[str, StepBuilder] = Registry("step_builders")
