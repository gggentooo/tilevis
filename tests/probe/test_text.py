"""Unit tests for tokenizers, display rules, and prompt templates."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from tilevis.core.errors import ModelLoadError
from tilevis.probe.loading.code_import import import_model_module
from tilevis.probe.text.templates import TEMPLATES
from tilevis.probe.text.tokenizer import TiktokenTokenizer, validate_token_ids


def test_tiktoken_tokenizer_encoding_and_eos() -> None:
    tok = TiktokenTokenizer("gpt2")
    assert tok.eos_id == 50256

    text = "Hello world <|endoftext|>"
    ids = tok.encode(text)
    assert ids[-1] == 50256
    decoded = tok.decode(ids)
    assert decoded == text


def test_token_display_rules() -> None:
    tok = TiktokenTokenizer("gpt2")

    # 1. Leading space displayed as '·'
    space_word_id = tok.encode(" effort")[0]
    disp_space = tok.token_display(space_word_id)
    assert disp_space.startswith("·")
    assert "effort" in disp_space

    # 2. Newline displayed as '↵'
    newline_id = tok.encode("\n")[0]
    disp_newline = tok.token_display(newline_id)
    assert disp_newline == "↵"

    # 3. Partial / multi-byte UTF-8 bytes displayed as <0xAB>
    korean_ids = tok.encode("한")
    # In GPT-2 BPE, '한' is split into 2 or 3 byte tokens
    for kid in korean_ids:
        raw_b = tok._enc.decode_single_token_bytes(kid)
        try:
            raw_b.decode("utf-8")
        except UnicodeDecodeError:
            disp_korean = tok.token_display(kid)
            assert disp_korean.startswith("<0x")
            assert disp_korean.endswith(">")


def test_alpaca_template_matches_ch07_format_input() -> None:
    """Verify that the alpaca template produces text identical to ch07 format_input."""
    root = Path(__file__).resolve().parent.parent.parent
    finetune_file = root / "ch07" / "01_main-chapter-code" / "gpt_instruction_finetuning.py"

    # gpt_instruction_finetuning imports gpt_download which imports tensorflow.
    # We mock tensorflow/gpt_download so importing gpt_instruction_finetuning succeeds.
    with mock.patch.dict(
        "sys.modules", {"tensorflow": mock.MagicMock(), "gpt_download": mock.MagicMock()}
    ):
        mod = import_model_module(finetune_file)
        original_format_input = mod.format_input

        test_prompt = "Explain quantum computing to high school students."
        entry = {"instruction": test_prompt, "input": ""}
        expected_text = original_format_input(entry) + "\n\n### Response:\n"

        alpaca_tmpl = TEMPLATES.get("alpaca")
        actual_text = alpaca_tmpl.format(test_prompt)

        assert actual_text == expected_text


def test_raw_template_identity() -> None:
    raw_tmpl = TEMPLATES.get("raw")
    prompt = "Simple raw text prompt."
    assert raw_tmpl.format(prompt) == prompt


def test_wrong_vocab_raises_model_load_error() -> None:
    tok = TiktokenTokenizer("gpt2")
    token_ids = tok.encode("Hello world")  # e.g. [15496, 995]
    custom_small_vocab = 1000  # Tokenizer ID 15496 exceeds 1000

    with pytest.raises(ModelLoadError, match="exceeds model vocabulary size"):
        validate_token_ids(token_ids, vocab_size=custom_small_vocab)
