"""Unit tests for weight extraction and presentation orientation."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from tests.fixtures.tiny_models import LinearWithLoRA

from tilevis.probe.weights import extract_weights


def test_linear_weight_transposed_to_in_features_by_out_features() -> None:
    in_features, out_features = 10, 20
    lin = nn.Linear(in_features, out_features, bias=True)

    weights = extract_weights(lin)
    assert weights is not None
    assert weights.weight is not None
    assert weights.bias is not None

    # Visualizer convention: (in_features, out_features)
    assert weights.weight.shape == (in_features, out_features)
    assert weights.bias.shape == (out_features,)

    # Verify transpose: element (i, j) in visual weight == lin.weight[j, i]
    assert np.isclose(weights.weight[3, 7], lin.weight[7, 3].item())


def test_lora_wrapper_weight_extraction() -> None:
    in_dim, out_dim, rank, alpha = 8, 12, 2, 4.0
    lin = nn.Linear(in_dim, out_dim, bias=True)
    wrapped = LinearWithLoRA(lin, rank=rank, alpha=alpha)

    weights = extract_weights(wrapped)
    assert weights is not None
    assert weights.weight is not None
    assert weights.bias is not None

    assert weights.weight.shape == (in_dim, out_dim)

    # Compute expected effective weight: (W + alpha * (A @ B).T).T = W.T + alpha * (A @ B)
    expected_eff = (lin.weight + alpha * (wrapped.lora.A @ wrapped.lora.B).t()).t()
    expected_np = expected_eff.detach().cpu().numpy()
    assert np.allclose(weights.weight, expected_np, atol=1e-5)


def test_embedding_weight_extraction() -> None:
    num_embeddings, embedding_dim = 100, 32
    emb = nn.Embedding(num_embeddings, embedding_dim)

    weights = extract_weights(emb)
    assert weights is not None
    assert weights.weight is not None
    assert weights.bias is None
    assert weights.weight.shape == (num_embeddings, embedding_dim)


def test_layernorm_scale_and_shift_extraction() -> None:
    dim = 16
    ln = nn.LayerNorm(dim)

    weights = extract_weights(ln)
    assert weights is not None
    assert weights.scale is not None
    assert weights.shift is not None
    assert weights.scale.shape == (dim,)
    assert weights.shift.shape == (dim,)
