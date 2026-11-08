"""Compatibility test verifying all compat_matrix variants load and match logits bit-for-bit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.compat_matrix import COMPAT_MATRIX
from tests.fixtures.tiny_models import (
    build_matrix_model,
    get_previous_chapters_path,
    save_matrix_checkpoint,
)

from tilevis.probe.adapters import ADAPTERS
from tilevis.probe.loading.factory import build_model
from tilevis.probe.loading.profile import (
    AdapterSettings,
    ModelLoraConfig,
    ModelProfile,
    ModelSettings,
    TextSettings,
    load_profile,
)


@pytest.mark.parametrize("matrix_key", list(COMPAT_MATRIX.keys()))
def test_compat_matrix_entry_profile_and_checkpoint_loading(
    matrix_key: str, tmp_path: Path
) -> None:
    entry = COMPAT_MATRIX[matrix_key]
    vocab_size = 256  # compact vocab for swift CPU testing

    # 1. Build ground-truth in-memory model and save checkpoint + profile
    ckpt_path, profile_path = save_matrix_checkpoint(
        entry,
        tmp_path / matrix_key,
        vocab_size=vocab_size,
        seed=42,
    )

    # Reconstruct original in-memory model for forward evaluation
    orig_model = build_matrix_model(entry, vocab_size=vocab_size, seed=42)

    # 2. Test loading via profile
    profile = load_profile(profile_path)
    built_from_profile = build_model(profile)

    # (a) Check configuration matches
    true_cfg = entry.to_config_dict(vocab_size=vocab_size)
    for key in ("vocab_size", "emb_dim", "context_length", "n_layers", "qkv_bias"):
        assert built_from_profile.cfg[key] == true_cfg[key], f"Mismatch in {key}"

    # (b) Adapter selection confidence
    assert built_from_profile.spec.adapter == "gpt_scratch"
    assert built_from_profile.adapter_confidence >= 0.90

    # (c) Bit-for-bit logits match on CPU (1e-5 tolerance for LoRA floating-point additions)
    torch.manual_seed(999)
    input_idx = torch.randint(0, vocab_size, (1, min(4, entry.context_length)))
    with torch.no_grad():
        orig_logits = orig_model(input_idx)
        profile_logits = built_from_profile.model(input_idx)

    if entry.variant == "lora":
        diff = torch.max(torch.abs(orig_logits - profile_logits)).item()
        assert diff < 1e-5, f"LoRA profile logits discrepancy: {diff}"
    else:
        assert torch.equal(orig_logits, profile_logits), (
            f"Bit-for-bit logit mismatch in {entry.name}"
        )

    # (d) Check generic adapter produces sensible leaf mappings when forced
    generic_adapter = ADAPTERS.get("generic")
    generic_spec = generic_adapter.describe(built_from_profile.model)
    assert generic_spec.adapter == "generic"
    assert generic_spec.token_embedding == "tok_emb"
    assert generic_spec.lm_head == "out_head"

    # 3. Test loading directly via checkpoint (simulating `tilevis inspect --checkpoint`)
    lora_cfg = (
        ModelLoraConfig(alpha=float(entry.extras.get("lora_alpha", 16.0)))
        if entry.variant == "lora" and entry.extras
        else None
    )
    ckpt_profile = ModelProfile(
        name=f"{entry.name}_direct",
        model=ModelSettings(
            code=get_previous_chapters_path(),
            cls_name="GPTModel",
            checkpoint=ckpt_path,
            lora=lora_cfg,
        ),
        text=TextSettings(tokenizer="gpt2", template="raw"),
        adapter=AdapterSettings(name="auto"),
        source_path=ckpt_path,
    )

    # (a) When n_heads is not given, inferred cfg equals true cfg on all structural fields
    built_without_heads = build_model(ckpt_profile)
    for key in ("vocab_size", "emb_dim", "context_length", "n_layers", "qkv_bias"):
        assert built_without_heads.cfg[key] == true_cfg[key], (
            f"Checkpoint inferred mismatch in {key}"
        )

    # (b) When n_heads is provided via CLI/override, logits equal original model bit-for-bit on CPU
    built_from_ckpt = build_model(ckpt_profile, overrides={"n_heads": entry.n_heads})

    with torch.no_grad():
        ckpt_logits = built_from_ckpt.model(input_idx)

    if entry.variant == "lora":
        diff = torch.max(torch.abs(orig_logits - ckpt_logits)).item()
        assert diff < 1e-5, f"LoRA checkpoint logits discrepancy: {diff}"
    else:
        assert torch.equal(orig_logits, ckpt_logits), (
            f"Bit-for-bit logit mismatch in ckpt for {entry.name}"
        )
