"""Autoregressive generation loop mirroring previous_chapters.generate().

Single responsibility: manage multi-token generation with greedy argmax, top-k sampling,
context window cropping, and EOS termination.
Allowed dependencies: stdlib, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from tilevis.core.config import RecordSettings
from tilevis.core.trace import Prediction
from tilevis.probe.text.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationPassResult:
    """Outcome of a single generation pass."""

    pass_index: int
    input_ids: tuple[int, ...]
    input_tokens: tuple[str, ...]
    text_before: str
    prediction: Prediction
    is_eos: bool
    context_cropped: bool


def select_next_token(
    logits: torch.Tensor,
    strategy: str = "greedy",
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Select next token index according to strategy, temperature, and top-k filtering.

    Semantics strictly match previous_chapters.generate().
    """
    # Filter logits with top_k sampling if requested
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        top_logits, _ = torch.topk(logits, k)
        min_val = top_logits[-1]
        filtered_logits = torch.where(
            logits < min_val,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )
    else:
        filtered_logits = logits

    if strategy == "sample" and temperature > 0.0:
        scaled_logits = filtered_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        if generator is not None:
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
        return int(idx_next.item())

    # Greedy argmax
    idx_next = torch.argmax(filtered_logits, dim=-1)
    return int(idx_next.item())


def generate_passes(
    model: nn.Module,
    prompt_ids: list[int],
    tokenizer: Tokenizer,
    settings: RecordSettings,
    context_size: int | None = None,
    forward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[GenerationPassResult]:
    """Execute autoregressive generation across passes, yielding results per pass.

    Args:
        model: PyTorch model.
        prompt_ids: Token IDs of initial prompt.
        tokenizer: Tokenizer for token display and decoding.
        settings: RecordSettings with new_tokens, strategy, seed, etc.
        context_size: Context length limit for left-cropping.
        forward_fn: Optional custom forward callable (e.g. running under hooks).

    Yields:
        GenerationPassResult for each pass.
    """
    device = next(model.parameters()).device
    generator = None
    if settings.strategy == "sample":
        generator = torch.Generator(device=str(device) if device.type != "mps" else "cpu")
        generator.manual_seed(settings.seed)

    current_ids = list(prompt_ids)

    for pass_idx in range(settings.new_tokens):
        # Context cropping
        cropped = False
        if context_size is not None and len(current_ids) > context_size:
            feed_ids = current_ids[-context_size:]
            cropped = True
        else:
            feed_ids = list(current_ids)

        feed_tensor = torch.tensor([feed_ids], device=device)

        # Execute forward pass
        with torch.inference_mode():
            logits_out = forward_fn(feed_tensor) if forward_fn is not None else model(feed_tensor)

        # Logits for the last token position
        last_logits = logits_out[0, -1, :].detach()
        if not torch.all(torch.isfinite(last_logits)):
            last_logits = torch.nan_to_num(last_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Top-k candidate probabilities from true raw forward logits
        raw_probs = torch.softmax(last_logits, dim=-1)
        k_disp = min(settings.top_k_display, last_logits.shape[-1])
        top_probs, top_indices = torch.topk(raw_probs, k_disp)

        cand_ids = tuple(int(i.item()) for i in top_indices)
        cand_probs = tuple(float(p.item()) for p in top_probs)
        cand_tokens = tuple(tokenizer.token_display(i) for i in cand_ids)

        # Token selection
        chosen_id = select_next_token(
            logits=last_logits,
            strategy=settings.strategy,
            temperature=settings.temperature,
            top_k=settings.top_k,
            generator=generator,
        )
        chosen_token = tokenizer.token_display(chosen_id)

        prediction = Prediction(
            candidate_ids=cand_ids,
            candidate_tokens=cand_tokens,
            candidate_probs=cand_probs,
            chosen_id=chosen_id,
            chosen_token=chosen_token,
            strategy=settings.strategy,
        )

        input_tokens = tuple(tokenizer.token_display(i) for i in feed_ids)
        text_before = tokenizer.decode(feed_ids)
        is_eos = bool(tokenizer.eos_id is not None and chosen_id == tokenizer.eos_id)

        yield GenerationPassResult(
            pass_index=pass_idx,
            input_ids=tuple(feed_ids),
            input_tokens=input_tokens,
            text_before=text_before,
            prediction=prediction,
            is_eos=is_eos,
            context_cropped=cropped,
        )

        if is_eos:
            logger.info("EOS token reached at pass %d; terminating generation.", pass_idx)
            break

        current_ids.append(chosen_id)
