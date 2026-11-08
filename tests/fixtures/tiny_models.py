"""Tiny model construction and checkpoint serialization fixtures for testing.

Single responsibility: build and save every compat_matrix variant in all checkpoint styles
(plain, LoRA-wrapped, DataParallel-wrapped).
Allowed dependencies: stdlib, torch, tests.compat_matrix, tilevis.probe.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from tests.compat_matrix import ModelMatrixEntry
from tilevis.probe.loading.code_import import import_model_module


class LoRALayer(nn.Module):
    """LoRA parameter layer matching ch07 exercise_experiments."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Linear wrapper with LoRA bypass matching ch07 exercise_experiments."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def apply_lora_wrapper(model: nn.Module, rank: int = 4, alpha: float = 8.0) -> None:
    """Recursively wrap all nn.Linear modules with LinearWithLoRA."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            apply_lora_wrapper(module, rank, alpha)


def get_previous_chapters_path() -> Path:
    """Return resolved path to read-only ch07 previous_chapters.py."""
    root = Path(__file__).resolve().parent.parent.parent
    return (root / "ch07" / "01_main-chapter-code" / "previous_chapters.py").resolve()


def build_matrix_model(
    entry: ModelMatrixEntry,
    vocab_size: int = 50257,
    seed: int = 42,
) -> nn.Module:
    """Construct a GPTModel instance from a compat_matrix entry."""
    torch.manual_seed(seed)
    code_path = get_previous_chapters_path()
    mod = import_model_module(code_path)
    gpt_cls = mod.GPTModel

    cfg = entry.to_config_dict(vocab_size=vocab_size)
    model: nn.Module = gpt_cls(cfg)
    model.eval()
    return model


def save_matrix_checkpoint(
    entry: ModelMatrixEntry,
    out_dir: Path,
    vocab_size: int = 50257,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Build and save a checkpoint and profile matching a compat_matrix entry.

    Returns:
        Tuple of (checkpoint_path, profile_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{entry.name}.pth"
    profile_path = out_dir / f"{entry.name}.toml"

    model = build_matrix_model(entry, vocab_size=vocab_size, seed=seed)

    lora_alpha = 16.0
    if entry.variant == "lora":
        rank = int(entry.extras.get("lora_rank", 4)) if entry.extras else 4
        lora_alpha = float(entry.extras.get("lora_alpha", 8.0)) if entry.extras else 8.0
        apply_lora_wrapper(model, rank=rank, alpha=lora_alpha)
        torch.save(model.state_dict(), ckpt_path)

    elif entry.variant == "wrapped":
        # Simulates ch05 training checkpoints: nested dict and DataParallel module. prefix
        state_dict = {f"module.{k}": v for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": {},
                "epoch": 1,
            },
            ckpt_path,
        )
    else:  # plain
        torch.save(model.state_dict(), ckpt_path)

    # Write corresponding TOML profile
    code_path = get_previous_chapters_path()
    profile_content = [
        "[profile]",
        f'name = "{entry.name}"',
        "",
        "[model]",
        f'code = "{code_path.as_posix()}"',
        'class = "GPTModel"',
        f'checkpoint = "{ckpt_path.as_posix()}"',
        f"seed = {seed}",
        "",
        "[model.config]",
        f"n_heads = {entry.n_heads}",
    ]

    if entry.variant == "lora":
        profile_content.extend(
            [
                "",
                "[model.lora]",
                f"alpha = {lora_alpha}",
            ]
        )

    profile_content.extend(
        [
            "",
            "[text]",
            'tokenizer = "gpt2"',
            'template = "raw"',
            "",
            "[adapter]",
            'name = "auto"',
        ]
    )

    profile_path.write_text("\n".join(profile_content) + "\n", encoding="utf-8")
    return ckpt_path, profile_path
