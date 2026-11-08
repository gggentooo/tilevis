"""Unit tests for state-dict config inference and contradiction detection."""

from __future__ import annotations

import pytest
import torch

from tilevis.core.errors import ConfigInferenceError
from tilevis.probe.loading.config_infer import (
    infer_config_from_state_dict,
    infer_n_heads_provisional,
)


def test_infer_all_rules_from_state_dict() -> None:
    state_dict = {
        "tok_emb.weight": torch.empty(50257, 128),
        "pos_emb.weight": torch.empty(512, 128),
        "trf_blocks.0.att.W_query.weight": torch.empty(128, 128),
        "trf_blocks.0.att.W_query.bias": torch.empty(128),
        "trf_blocks.1.att.W_query.weight": torch.empty(128, 128),
        "trf_blocks.2.att.W_query.weight": torch.empty(128, 128),
    }

    inferred = infer_config_from_state_dict(state_dict)
    cfg = inferred.cfg

    assert cfg["vocab_size"] == 50257
    assert cfg["emb_dim"] == 128
    assert cfg["context_length"] == 512
    assert cfg["n_layers"] == 3
    assert cfg["qkv_bias"] is True
    assert cfg["drop_rate"] == 0.0
    # Provisional n_heads for 128 dim (128 // 64 = 2)
    assert cfg["n_heads"] == 2
    assert "n_heads" in inferred.inferred_keys


def test_infer_n_heads_provisional() -> None:
    assert infer_n_heads_provisional(768) == 12  # 768 // 64
    assert infer_n_heads_provisional(64) == 1
    assert infer_n_heads_provisional(48) == 8  # divisible by 8
    assert infer_n_heads_provisional(12) == 4  # divisible by 4
    assert infer_n_heads_provisional(10) == 2  # divisible by 2
    assert infer_n_heads_provisional(7) == 1  # prime


def test_contradiction_in_emb_dim_includes_both_values() -> None:
    state_dict = {"tok_emb.weight": torch.empty(1000, 64)}
    explicit = {"emb_dim": 128}

    with pytest.raises(ConfigInferenceError) as exc_info:
        infer_config_from_state_dict(state_dict, explicit)

    msg = str(exc_info.value)
    assert "128" in msg
    assert "64" in msg
    assert "emb_dim" in msg


def test_contradiction_in_n_layers_includes_both_values() -> None:
    state_dict = {
        "trf_blocks.0.weight": torch.empty(4, 4),
        "trf_blocks.1.weight": torch.empty(4, 4),
        "trf_blocks.2.weight": torch.empty(4, 4),
    }
    explicit = {"n_layers": 5}

    with pytest.raises(ConfigInferenceError) as exc_info:
        infer_config_from_state_dict(state_dict, explicit)

    msg = str(exc_info.value)
    assert "5" in msg
    assert "3" in msg
    assert "n_layers" in msg


def test_contradiction_in_qkv_bias() -> None:
    state_dict = {
        "trf_blocks.0.att.W_query.weight": torch.empty(4, 4),
        "trf_blocks.0.att.W_query.bias": torch.empty(4),
    }
    explicit = {"qkv_bias": False}

    with pytest.raises(ConfigInferenceError, match="qkv_bias"):
        infer_config_from_state_dict(state_dict, explicit)
