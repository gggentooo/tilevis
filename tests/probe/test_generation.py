"""Unit tests for autoregressive generation loop.

Single responsibility: test multi-token generation, greedy vs sampling, context cropping,
EOS early stopping, and equivalence to previous_chapters.generate().
"""

from __future__ import annotations

import pytest
import torch
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import build_matrix_model, get_previous_chapters_path

from tilevis.core.config import RecordSettings
from tilevis.probe.generation import generate_passes, select_next_token
from tilevis.probe.loading.code_import import import_model_module
from tilevis.probe.text.tokenizer import TiktokenTokenizer


@pytest.fixture
def tiny_model():
    """Build a tiny_ch07 model."""
    entry = COMPAT_MATRIX["tiny_ch07"]
    return build_matrix_model(entry, seed=123)


def test_select_next_token() -> None:
    """Test greedy and sampling token selection logic."""
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
    # Greedy must pick index 1
    chosen = select_next_token(logits, strategy="greedy")
    assert chosen == 1

    # Sampling with seed
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    c1 = select_next_token(logits, strategy="sample", temperature=1.0, generator=gen1)
    c2 = select_next_token(logits, strategy="sample", temperature=1.0, generator=gen2)
    assert c1 == c2


def test_generate_passes_equivalence_to_reference(tiny_model) -> None:
    """Verify that greedy generation matches previous_chapters.generate() token-for-token."""
    ch07_mod = import_model_module(get_previous_chapters_path())
    ref_generate = ch07_mod.generate

    tokenizer = TiktokenTokenizer("gpt2")
    prompt = "Every effort"
    prompt_ids = tokenizer.encode(prompt)
    context_size = 120
    new_tokens = 3

    # Run reference generate
    idx_in = torch.tensor([prompt_ids])
    ref_out = ref_generate(
        tiny_model,
        idx_in,
        max_new_tokens=new_tokens,
        context_size=context_size,
        temperature=0.0,
    )
    ref_tokens = ref_out[0].tolist()

    # Run tilevis generate_passes
    settings = RecordSettings(new_tokens=new_tokens, strategy="greedy", temperature=0.0)
    tilevis_tokens = list(prompt_ids)
    for res in generate_passes(
        tiny_model,
        prompt_ids,
        tokenizer,
        settings,
        context_size=context_size,
    ):
        tilevis_tokens.append(res.prediction.chosen_id)

    assert tilevis_tokens == ref_tokens


def test_generation_context_cropping(tiny_model) -> None:
    """Test that context cropping occurs when prompt exceeds context_size."""
    tokenizer = TiktokenTokenizer("gpt2")
    # Prompt with 10 tokens
    prompt_ids = [100 + i for i in range(10)]
    # Set context_size to 5
    settings = RecordSettings(new_tokens=2, strategy="greedy")

    results = list(
        generate_passes(
            tiny_model,
            prompt_ids,
            tokenizer,
            settings,
            context_size=5,
        )
    )

    assert len(results) == 2
    for r in results:
        assert r.context_cropped is True
        assert len(r.input_ids) == 5
        # Must be the last 5 tokens of the accumulated sequence
        assert r.input_ids[-1] == prompt_ids[-1] or r.pass_index > 0


def test_generation_eos_stopping(tiny_model) -> None:
    """Test that generation stops immediately when an EOS token is produced."""
    tokenizer = TiktokenTokenizer("gpt2")
    prompt_ids = [100, 200]
    # Set tokenizer.eos_id to whatever chosen token will be
    settings = RecordSettings(new_tokens=5, strategy="greedy")

    # Run pass 0 to find out what token is generated
    first_res = next(
        generate_passes(
            tiny_model,
            prompt_ids,
            tokenizer,
            settings,
            context_size=64,
        )
    )
    chosen_id = first_res.prediction.chosen_id

    # Now make that token the EOS token
    object.__setattr__(tokenizer, "eos_id", chosen_id)

    # Re-run: must stop after pass 0!
    results = list(
        generate_passes(
            tiny_model,
            prompt_ids,
            tokenizer,
            settings,
            context_size=64,
        )
    )
    assert len(results) == 1
    assert results[0].is_eos is True
