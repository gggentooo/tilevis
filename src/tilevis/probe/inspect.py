"""Model inspection utility for probe diagnostic CLI.

Single responsibility: inspect model profiles or checkpoints and print architecture,
inferred config, adapter bindings, and parameter counts.
Allowed dependencies: stdlib, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from tilevis.core.errors import ModelLoadError
from tilevis.probe.loading.checkpoint import load_checkpoint
from tilevis.probe.loading.factory import build_model
from tilevis.probe.loading.lora import detect_lora_tensors
from tilevis.probe.loading.profile import (
    AdapterSettings,
    ModelLoraConfig,
    ModelProfile,
    ModelSettings,
    TextSettings,
    load_profile,
)


def inspect_model(
    profile_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    stream: TextIO | None = None,
) -> int:
    """Inspect model profile or checkpoint file and output diagnostic details.

    Args:
        profile_path: Optional path to TOML model profile.
        checkpoint_path: Optional path to raw .pth checkpoint.
        stream: Target text stream for output (defaults to sys.stdout).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    out = stream if stream is not None else sys.stdout

    if profile_path is not None:
        profile = load_profile(profile_path)
    elif checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path).resolve()
        # Find default previous_chapters.py relative to repo root
        repo_code = (
            Path.cwd() / "ch07" / "01_main-chapter-code" / "previous_chapters.py"
        ).resolve()
        if not repo_code.exists():
            # Try relative to parent directories
            for parent in ckpt_p.parents:
                cand = (parent / "ch07" / "01_main-chapter-code" / "previous_chapters.py").resolve()
                if cand.exists():
                    repo_code = cand
                    break

        # Check if checkpoint has LoRA tensors to enable default alpha
        lora_cfg = None
        try:
            raw_ckpt = load_checkpoint(ckpt_p, allow_pickle=False)
            if detect_lora_tensors(raw_ckpt.state_dict):
                lora_cfg = ModelLoraConfig(alpha=16.0)
        except Exception:
            pass

        profile = ModelProfile(
            name=ckpt_p.stem,
            model=ModelSettings(
                code=repo_code,
                cls_name="GPTModel",
                checkpoint=ckpt_p,
                lora=lora_cfg,
            ),
            text=TextSettings(tokenizer="gpt2", template="raw"),
            adapter=AdapterSettings(name="auto"),
            source_path=ckpt_p,
        )
    else:
        raise ModelLoadError("Either --profile or --checkpoint must be provided for inspection.")

    built = build_model(profile)
    total_params = sum(p.numel() for p in built.model.parameters())
    trainable_params = sum(p.numel() for p in built.model.parameters() if p.requires_grad)

    out.write(f"\n{'=' * 60}\n")
    out.write(f"Tilevis Model Inspection: {profile.name}\n")
    out.write(f"{'=' * 60}\n")
    out.write(f"Code Path:        {profile.model.code}\n")
    out.write(f"Checkpoint:       {profile.model.checkpoint or 'None (Seeded Random Init)'}\n")
    out.write(f"Class Name:       {profile.model.cls_name or built.model.__class__.__name__}\n")
    out.write(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)\n")
    out.write(f"Trainable Params: {trainable_params:,}\n")

    out.write(f"\n{'-' * 60}\n")
    out.write("Architecture & Adapter Mapping\n")
    out.write(f"{'-' * 60}\n")
    out.write(
        f"Adapter:          {built.spec.adapter} (confidence: {built.adapter_confidence:.2f})\n"
    )
    out.write(f"Embedding Dim:    {built.spec.emb_dim}\n")
    out.write(f"Vocab Size:       {built.spec.vocab_size:,}\n")
    out.write(f"Context Length:   {built.spec.context_length}\n")
    heads_str = built.spec.n_heads if built.spec.n_heads is not None else "Provisional/Auto"
    out.write(f"Attention Heads:  {heads_str}\n")
    out.write(f"Residual Style:   {built.spec.residual_style}\n")
    out.write(f"Token Embedding:  {built.spec.token_embedding}\n")
    out.write(f"Pos Embedding:    {built.spec.position_embedding or 'None'}\n")
    out.write(f"Final LayerNorm:  {built.spec.final_norm or 'None'}\n")
    out.write(f"LM Output Head:   {built.spec.lm_head}\n")

    out.write(f"\n{'-' * 60}\n")
    out.write("Inferred Configuration\n")
    out.write(f"{'-' * 60}\n")
    for k, v in sorted(built.cfg.items()):
        source = "explicit" if k in profile.model.config else "inferred"
        out.write(f"  {k:<16}: {v!s:<12} ({source})\n")

    if built.spec.blocks:
        out.write(f"\n{'-' * 60}\n")
        out.write(f"Transformer Blocks ({len(built.spec.blocks)} total)\n")
        out.write(f"{'-' * 60}\n")
        header = (
            f"{'Block':<6} {'Norm1':<20} {'Attention Q/K/V/Out':<28} "
            f"{'Norm2':<20} {'FeedForward':<20}\n"
        )
        out.write(header)
        out.write(f"{'-' * 94}\n")
        for b in built.spec.blocks[:6]:  # Show first 6 blocks
            qkv = "W_q,k,v" if b.q else "none"
            att_col = f"{b.attention} ({qkv})"
            out.write(f"{b.index:<6} {b.norm1!s:<20} {att_col:<28} {b.norm2!s:<20} {b.ff!s:<20}\n")
        if len(built.spec.blocks) > 6:
            out.write(f"... and {len(built.spec.blocks) - 6} more blocks\n")

    if built.notes:
        out.write(f"\n{'-' * 60}\n")
        out.write("Notes & Diagnostics\n")
        out.write(f"{'-' * 60}\n")
        for note in built.notes:
            out.write(f"  * {note}\n")
    out.write(f"{'=' * 60}\n\n")

    return 0
