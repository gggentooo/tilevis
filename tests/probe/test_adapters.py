"""Unit tests for architecture adapters and role resolution."""

from __future__ import annotations

import torch.nn as nn
from tests.fixtures.tiny_models import get_previous_chapters_path

from tilevis.probe.adapters import select_adapter
from tilevis.probe.adapters.generic import GenericAdapter
from tilevis.probe.adapters.gpt_scratch import GptScratchAdapter
from tilevis.probe.loading.code_import import import_model_module


def test_gpt_scratch_adapter_resolution() -> None:
    code_path = get_previous_chapters_path()
    mod = import_model_module(code_path)
    gpt_cls = mod.GPTModel

    cfg = {
        "vocab_size": 200,
        "context_length": 32,
        "emb_dim": 16,
        "n_layers": 2,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = gpt_cls(cfg)

    adapter = GptScratchAdapter()
    conf = adapter.matches(model)
    assert conf >= 0.95

    spec = adapter.describe(model)
    assert spec.adapter == "gpt_scratch"
    assert spec.token_embedding == "tok_emb"
    assert spec.position_embedding == "pos_emb"
    assert spec.final_norm == "final_norm"
    assert spec.lm_head == "out_head"
    assert len(spec.blocks) == 2
    assert spec.blocks[0].norm1 == "trf_blocks.0.norm1"
    assert spec.blocks[0].q == "trf_blocks.0.att.W_query"
    assert spec.blocks[0].ff_in == "trf_blocks.0.ff.layers.0"
    assert spec.blocks[0].ff_act == "trf_blocks.0.ff.layers.1"
    assert spec.blocks[0].ff_out == "trf_blocks.0.ff.layers.2"


def test_model_without_pos_emb_sets_none() -> None:
    """Verify that a model without position embedding sets pos_emb to None."""
    code_path = get_previous_chapters_path()
    mod = import_model_module(code_path)
    gpt_cls = mod.GPTModel

    cfg = {
        "vocab_size": 100,
        "context_length": 16,
        "emb_dim": 16,
        "n_layers": 1,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = gpt_cls(cfg)
    # Remove pos_emb
    model.pos_emb = None  # type: ignore[assignment]

    adapter = GptScratchAdapter()
    spec = adapter.describe(model)
    assert spec.position_embedding is None


def test_adapter_roles_overrides() -> None:
    class CustomModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.my_tok_emb = nn.Embedding(50, 16)
            self.my_head = nn.Linear(16, 50)

    model = CustomModel()
    adapter = GenericAdapter()
    spec = adapter.describe(
        model,
        overrides={"token_embedding": "my_tok_emb", "lm_head": "my_head"},
    )
    assert spec.token_embedding == "my_tok_emb"
    assert spec.lm_head == "my_head"


def test_generic_adapter_fallback() -> None:
    # A completely non-GPT model (e.g. MLP)
    mlp = nn.Sequential(
        nn.Embedding(50, 16),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 50),
    )

    chosen_adapter, spec, _ = select_adapter(mlp, "auto")
    assert chosen_adapter.name == "generic"
    assert spec.blocks == ()
    assert spec.residual_style == "unknown"
