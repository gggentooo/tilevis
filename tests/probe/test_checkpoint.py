"""Unit tests for checkpoint loading, prefix stripping, and container unwrapping."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tilevis.core.errors import ModelLoadError
from tilevis.probe.loading.checkpoint import (
    load_checkpoint,
    strip_tensor_prefixes,
    unwrap_checkpoint_container,
)


def test_strip_tensor_prefixes() -> None:
    raw = {
        "module.tok_emb.weight": torch.ones(2, 2),
        "_orig_mod.trf_blocks.0.att.W_query.weight": torch.ones(2, 2),
        "module._orig_mod.final_norm.scale": torch.ones(2),
        "plain.weight": torch.ones(2, 2),
    }
    cleaned = strip_tensor_prefixes(raw)
    assert set(cleaned.keys()) == {
        "tok_emb.weight",
        "trf_blocks.0.att.W_query.weight",
        "final_norm.scale",
        "plain.weight",
    }


def test_unwrap_checkpoint_containers() -> None:
    # 1. Plain state dict
    plain = {"tok_emb.weight": torch.ones(2, 2)}
    sd, mod = unwrap_checkpoint_container(plain)
    assert sd == plain
    assert mod is None

    # 2. Nested model_state_dict
    nested = {"model_state_dict": {"out.weight": torch.ones(1)}}
    sd, mod = unwrap_checkpoint_container(nested)
    assert "out.weight" in sd
    assert mod is None

    # 3. nn.Module instance
    lin = nn.Linear(4, 2)
    sd, mod = unwrap_checkpoint_container(lin)
    assert "weight" in sd
    assert mod is lin


def test_load_checkpoint_roundtrip_with_korean_path(tmp_path: Path) -> None:
    korean_dir = tmp_path / "한국어_체크포인트_디렉토리"
    korean_dir.mkdir()
    ckpt_path = korean_dir / "모델.pth"

    expected_sd = {"tok_emb.weight": torch.randn(10, 8)}
    torch.save(expected_sd, ckpt_path)

    loaded = load_checkpoint(ckpt_path)
    assert loaded.source_path == ckpt_path.resolve()
    assert torch.equal(loaded.state_dict["tok_emb.weight"], expected_sd["tok_emb.weight"])


def test_missing_checkpoint_raises_model_load_error(tmp_path: Path) -> None:
    missing = tmp_path / "not_found.pth"
    with pytest.raises(ModelLoadError, match="Checkpoint file not found"):
        load_checkpoint(missing)


def test_pickled_full_module_rejection_with_helpful_hint(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "full_model.pth"
    model = nn.Linear(4, 2)
    # Save the full nn.Module instance (pickled object)
    torch.save(model, ckpt_path)

    with pytest.raises(ModelLoadError, match=r"allow_pickle|pickled objects"):
        load_checkpoint(ckpt_path, allow_pickle=False)

    # When allow_pickle=True, it should successfully unwrap
    loaded = load_checkpoint(ckpt_path, allow_pickle=True)
    assert "weight" in loaded.state_dict
    assert isinstance(loaded.module, nn.Linear)
