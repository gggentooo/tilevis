"""ArchitectureAdapter for previous_chapters-style GPTModel implementations.

Single responsibility: map standard GPTModel submodules to semantic pipeline roles.
Allowed dependencies: stdlib, torch, tilevis.core.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

from tilevis.core.errors import ModelLoadError
from tilevis.probe.adapters.base import BlockSpec, ModelSpec


class GptScratchAdapter:
    """Adapter for 'Build a Large Language Model From Scratch' GPTModel architecture."""

    name = "gpt_scratch"

    def matches(self, model: nn.Module) -> float:
        """Check confidence that model matches the gpt_scratch architecture."""
        score = 0.0

        # Check token embedding
        if hasattr(model, "tok_emb"):
            score += 0.25

        # Check transformer blocks
        trf = getattr(model, "trf_blocks", None)
        if trf is not None and isinstance(trf, (nn.Sequential, nn.ModuleList, list, tuple)):
            score += 0.35
            # Inspect first block if available
            try:
                first_block = next(iter(trf))
                if hasattr(first_block, "att") and hasattr(first_block, "ff"):
                    score += 0.15
            except Exception:
                pass

        # Check final norm
        if hasattr(model, "final_norm"):
            score += 0.10

        # Check LM head
        if hasattr(model, "out_head") or hasattr(model, "lm_head"):
            score += 0.15

        return min(1.0, score)

    def describe(self, model: nn.Module, overrides: Mapping[str, str] | None = None) -> ModelSpec:
        """Extract ModelSpec from a GPTModel instance, applying optional overrides."""
        ov = dict(overrides or {})

        # 1. Embeddings
        tok_emb_path = ov.get("token_embedding", "tok_emb")
        if not hasattr(model, tok_emb_path):
            raise ModelLoadError(f"Model missing required token embedding module: {tok_emb_path!r}")
        tok_emb = getattr(model, tok_emb_path)

        pos_emb_path = ov.get("position_embedding", "pos_emb")
        has_pos = hasattr(model, pos_emb_path) and getattr(model, pos_emb_path) is not None
        actual_pos_path = pos_emb_path if has_pos else None

        # 2. Dimensions
        emb_dim = getattr(tok_emb, "embedding_dim", None)
        vocab_size = getattr(tok_emb, "num_embeddings", None)

        if has_pos:
            pos_emb = getattr(model, pos_emb_path)
            context_length = getattr(pos_emb, "num_embeddings", None)
        else:
            context_length = None

        # 3. LM Head
        lm_head_path = ov.get("lm_head")
        if lm_head_path is None:
            if hasattr(model, "out_head"):
                lm_head_path = "out_head"
            elif hasattr(model, "lm_head"):
                lm_head_path = "lm_head"
            else:
                raise ModelLoadError("Model missing LM head module ('out_head' or 'lm_head')")

        lm_head = getattr(model, lm_head_path)
        if vocab_size is None and hasattr(lm_head, "out_features"):
            vocab_size = lm_head.out_features
        if emb_dim is None and hasattr(lm_head, "in_features"):
            emb_dim = lm_head.in_features

        # 4. Final Norm
        final_norm_path = ov.get("final_norm", "final_norm")
        actual_final_norm = final_norm_path if hasattr(model, final_norm_path) else None

        # 5. Transformer Blocks
        blocks_container_path = ov.get("blocks", "trf_blocks")
        if not hasattr(model, blocks_container_path):
            raise ModelLoadError(
                f"Model missing transformer blocks container: {blocks_container_path!r}"
            )

        blocks_container = getattr(model, blocks_container_path)
        blocks: list[BlockSpec] = []
        n_heads: int | None = None

        for b_idx, block in enumerate(blocks_container):
            b_prefix = f"{blocks_container_path}.{b_idx}"

            # Attention paths
            att_path = f"{b_prefix}.att" if hasattr(block, "att") else None
            q_path = (
                f"{b_prefix}.att.W_query" if att_path and hasattr(block.att, "W_query") else None
            )
            k_path = f"{b_prefix}.att.W_key" if att_path and hasattr(block.att, "W_key") else None
            v_path = (
                f"{b_prefix}.att.W_value" if att_path and hasattr(block.att, "W_value") else None
            )
            out_proj_path = (
                f"{b_prefix}.att.out_proj" if att_path and hasattr(block.att, "out_proj") else None
            )

            if n_heads is None and att_path and hasattr(block.att, "num_heads"):
                n_heads = block.att.num_heads

            # Normalization paths
            norm1_path = f"{b_prefix}.norm1" if hasattr(block, "norm1") else None
            norm2_path = f"{b_prefix}.norm2" if hasattr(block, "norm2") else None

            # Feedforward paths
            ff_path = f"{b_prefix}.ff" if hasattr(block, "ff") else None
            ff_in: str | None = None
            ff_act: str | None = None
            ff_out: str | None = None

            if (
                ff_path
                and hasattr(block.ff, "layers")
                and isinstance(block.ff.layers, nn.Sequential)
            ):
                layers = list(block.ff.layers)
                if len(layers) >= 3:
                    ff_in = f"{ff_path}.layers.0"
                    ff_act = f"{ff_path}.layers.1"
                    ff_out = f"{ff_path}.layers.{len(layers) - 1}"

            blocks.append(
                BlockSpec(
                    index=b_idx,
                    path=b_prefix,
                    norm1=norm1_path,
                    attention=att_path,
                    q=q_path,
                    k=k_path,
                    v=v_path,
                    out_proj=out_proj_path,
                    norm2=norm2_path,
                    ff=ff_path,
                    ff_in=ff_in,
                    ff_act=ff_act,
                    ff_out=ff_out,
                )
            )

        return ModelSpec(
            adapter=self.name,
            token_embedding=tok_emb_path,
            position_embedding=actual_pos_path,
            blocks=tuple(blocks),
            final_norm=actual_final_norm,
            lm_head=lm_head_path,
            emb_dim=int(emb_dim or 64),
            vocab_size=int(vocab_size or 50257),
            context_length=int(context_length) if context_length is not None else None,
            n_heads=n_heads,
            residual_style="pre_norm",
        )
