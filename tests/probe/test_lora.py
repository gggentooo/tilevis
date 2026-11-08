"""Unit tests for LoRA tensor detection, merging, and numerical equivalence."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tests.fixtures.tiny_models import (
    LinearWithLoRA,
    apply_lora_wrapper,
    get_previous_chapters_path,
)

from tilevis.probe.loading.code_import import import_model_module
from tilevis.probe.loading.lora import detect_lora_tensors, merge_lora_state_dict


def test_lora_detection() -> None:
    plain_sd = {"linear.weight": torch.ones(4, 4)}
    assert detect_lora_tensors(plain_sd) is False

    lora_sd = {
        "linear.weight": torch.ones(4, 4),
        "linear.lora.A": torch.ones(4, 2),
        "linear.lora.B": torch.ones(2, 4),
    }
    assert detect_lora_tensors(lora_sd) is True


def test_lora_merge_math_equivalence() -> None:
    in_dim, out_dim, rank, alpha = 6, 8, 2, 4.0

    torch.manual_seed(42)
    linear = nn.Linear(in_dim, out_dim, bias=True)
    wrapped = LinearWithLoRA(linear, rank=rank, alpha=alpha)

    # Extract state dict as saved by exercise_experiments
    state_dict = {f"submod.{k}": v for k, v in wrapped.state_dict().items()}
    assert detect_lora_tensors(state_dict) is True

    # Merge state dict
    merged, was_merged = merge_lora_state_dict(state_dict, alpha=alpha)
    assert was_merged is True
    assert "submod.weight" in merged
    assert "submod.bias" in merged
    assert "submod.linear.weight" not in merged
    assert "submod.lora.A" not in merged

    # Compare forward pass outputs
    merged_linear = nn.Linear(in_dim, out_dim, bias=True)
    merged_linear.weight.data.copy_(merged["submod.weight"])
    merged_linear.bias.data.copy_(merged["submod.bias"])

    x = torch.randn(3, in_dim)
    with torch.no_grad():
        out_wrapped = wrapped(x)
        out_merged = merged_linear(x)

    max_diff = torch.max(torch.abs(out_wrapped - out_merged)).item()
    assert max_diff < 1e-5, f"LoRA merge output discrepancy: {max_diff}"


def test_lora_full_gpt_model_logits_within_tolerance() -> None:
    """Verify that a LoRA-wrapped GPTModel produces logits matching the merged model within 1e-5."""
    code_path = get_previous_chapters_path()
    mod = import_model_module(code_path)
    gpt_cls = mod.GPTModel

    cfg = {
        "vocab_size": 100,
        "context_length": 16,
        "emb_dim": 16,
        "n_layers": 2,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

    torch.manual_seed(99)
    base_model = gpt_cls(cfg)
    base_model.eval()

    # Wrap linear layers in LoRA
    rank, alpha = 4, 8.0
    apply_lora_wrapper(base_model, rank=rank, alpha=alpha)
    wrapped_sd = base_model.state_dict()

    # Merge state dict
    merged_sd, was_merged = merge_lora_state_dict(wrapped_sd, alpha=alpha)
    assert was_merged is True

    # Construct clean base model and load merged weights
    clean_model = gpt_cls(cfg)
    clean_model.eval()
    clean_model.load_state_dict(merged_sd, strict=False)

    # Test forward pass
    in_idx = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits_wrapped = base_model(in_idx)
        logits_merged = clean_model(in_idx)

    max_diff = torch.max(torch.abs(logits_wrapped - logits_merged)).item()
    assert max_diff < 1e-5, f"Full model LoRA logits discrepancy: {max_diff}"


def test_lora_missing_alpha_logs_warning_and_uses_default(caplog: pytest.LogCaptureFixture) -> None:
    lora_sd = {
        "fc.linear.weight": torch.ones(2, 2),
        "fc.lora.A": torch.ones(2, 1),
        "fc.lora.B": torch.ones(1, 2),
    }
    with caplog.at_level("WARNING"):
        _, was_merged = merge_lora_state_dict(lora_sd, alpha=None)

    assert was_merged is True
    assert any("LoRA checkpoint without explicit" in r.message for r in caplog.records)
