"""Generic architecture adapter fallback for arbitrary nn.Module models.

Single responsibility: inspect leaf modules and identify basic embedding and projection roles.
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

from tilevis.probe.adapters.base import ModelSpec


class GenericAdapter:
    """Fallback adapter for unknown or non-standard model architectures."""

    name = "generic"

    def matches(self, model: nn.Module) -> float:
        """Always return 0.5 as baseline fallback confidence."""
        return 0.5

    def describe(self, model: nn.Module, overrides: Mapping[str, str] | None = None) -> ModelSpec:
        """Produce a ModelSpec mapping basic embedding and LM head roles."""
        ov = dict(overrides or {})

        # Find leaf modules
        embeddings: list[str] = []
        linears: list[str] = []
        final_norm_candidate: str | None = None

        for name, mod in model.named_modules():
            if not name:
                continue
            if isinstance(mod, nn.Embedding):
                embeddings.append(name)
            elif isinstance(mod, nn.Linear):
                linears.append(name)
            elif isinstance(mod, nn.LayerNorm):
                final_norm_candidate = name

        # 1. Token embedding
        tok_emb_path = ov.get("token_embedding")
        if tok_emb_path is None:
            tok_emb_path = embeddings[0] if embeddings else "embedding"

        # 2. Position embedding
        pos_emb_path = ov.get("position_embedding")
        if pos_emb_path is None:
            pos_emb_path = embeddings[1] if len(embeddings) > 1 else None

        # 3. LM Head
        lm_head_path = ov.get("lm_head")
        if lm_head_path is None:
            lm_head_path = linears[-1] if linears else "linear"

        # 4. Final Norm
        final_norm_path = ov.get("final_norm", final_norm_candidate)

        # Dimensions inference
        vocab_size = 50257
        emb_dim = 64
        context_length: int | None = None

        if hasattr(model, tok_emb_path):
            tok_emb = getattr(model, tok_emb_path)
            if hasattr(tok_emb, "num_embeddings"):
                vocab_size = tok_emb.num_embeddings
            if hasattr(tok_emb, "embedding_dim"):
                emb_dim = tok_emb.embedding_dim

        if pos_emb_path and hasattr(model, pos_emb_path):
            pos_emb = getattr(model, pos_emb_path)
            if hasattr(pos_emb, "num_embeddings"):
                context_length = pos_emb.num_embeddings

        if hasattr(model, lm_head_path):
            lm_head = getattr(model, lm_head_path)
            if hasattr(lm_head, "out_features") and not embeddings:
                vocab_size = lm_head.out_features
            if hasattr(lm_head, "in_features") and not embeddings:
                emb_dim = lm_head.in_features

        return ModelSpec(
            adapter=self.name,
            token_embedding=tok_emb_path,
            position_embedding=pos_emb_path,
            blocks=(),
            final_norm=final_norm_path,
            lm_head=lm_head_path,
            emb_dim=emb_dim,
            vocab_size=vocab_size,
            context_length=context_length,
            n_heads=None,
            residual_style="unknown",
        )
