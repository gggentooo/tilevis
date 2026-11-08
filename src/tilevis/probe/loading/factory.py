"""Model factory assembling ready nn.Module instances and ModelSpecs.

Single responsibility: coordinate code import, checkpoint loading, config inference,
LoRA folding, model instantiation, and adapter role assignment.
Allowed dependencies: stdlib, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tilevis.core.errors import ModelLoadError
from tilevis.probe.adapters import ModelSpec, select_adapter
from tilevis.probe.loading.checkpoint import load_checkpoint
from tilevis.probe.loading.code_import import import_model_module
from tilevis.probe.loading.config_infer import infer_config_from_state_dict
from tilevis.probe.loading.lora import merge_lora_state_dict
from tilevis.probe.loading.profile import ModelProfile
from tilevis.probe.text.templates import TEMPLATES, PromptTemplate
from tilevis.probe.text.tokenizer import TiktokenTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuiltModel:
    """Fully instantiated, configured, and bound model ready for probe analysis."""

    model: nn.Module
    spec: ModelSpec
    cfg: dict[str, Any]
    tokenizer: Tokenizer
    template: PromptTemplate
    adapter_confidence: float
    notes: tuple[str, ...]


def resolve_device(device_request: str = "cpu") -> torch.device:
    """Resolve device string ('auto', 'cpu', 'cuda', 'mps') to a torch.device."""
    req = device_request.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def build_model(
    profile: ModelProfile,
    overrides: Mapping[str, Any] | None = None,
    device: str = "cpu",
) -> BuiltModel:
    """Build a complete, ready-to-run PyTorch model from a ModelProfile.

    Args:
        profile: Parsed ModelProfile.
        overrides: Optional runtime CLI configuration overrides.
        device: Target execution device ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        BuiltModel containing model, spec, tokenizer, and diagnostic notes.

    Raises:
        ModelLoadError: If loading, importing, or weight assignment fails.
    """
    ov = dict(overrides or {})
    notes: list[str] = []

    # 1. Merge overrides with profile config
    merged_config = dict(profile.model.config)
    merged_config.update(ov)

    # 2. Tokenizer & Template
    tokenizer = TiktokenTokenizer(profile.text.tokenizer)
    template = TEMPLATES.get(profile.text.template)

    # 3. Model construction paths
    model: nn.Module
    cfg: dict[str, Any]
    target_device = resolve_device(device)

    # Path (a): factory function
    if profile.model.factory:
        factory_str = profile.model.factory
        notes.append(f"Loaded via custom factory: {factory_str}")
        if ":" in factory_str:
            file_part, fn_name = factory_str.split(":", 1)
            factory_file = (
                (profile.source_path.parent / file_part).resolve()
                if profile.source_path
                else Path(file_part).resolve()
            )
            module = import_model_module(factory_file)
        else:
            module = import_model_module(profile.model.code)
            fn_name = factory_str

        if not hasattr(module, fn_name):
            raise ModelLoadError(f"Factory function {fn_name!r} not found in {module}")
        factory_fn = getattr(module, fn_name)
        model = factory_fn(merged_config)
        cfg = merged_config

    # Path (b): class + checkpoint
    elif profile.model.checkpoint is not None:
        ckpt = load_checkpoint(
            profile.model.checkpoint,
            allow_pickle=profile.model.allow_pickle,
        )
        notes.append(f"Loaded checkpoint weights from: {ckpt.source_path.name}")

        # Check for and fold LoRA weights
        lora_alpha = profile.model.lora.alpha if profile.model.lora else None
        merged_state_dict, was_lora = merge_lora_state_dict(ckpt.state_dict, alpha=lora_alpha)
        if was_lora:
            notes.append(f"Merged LoRA parameters using alpha={lora_alpha or 16.0}")

        # Infer config
        infer_res = infer_config_from_state_dict(merged_state_dict, merged_config)
        cfg = infer_res.cfg
        n_inf = len(infer_res.inferred_keys)
        n_exp = len(infer_res.explicit_keys)
        notes.append(f"Config inferred ({n_inf} inferred, {n_exp} explicit)")

        # Import model class
        code_mod = import_model_module(profile.model.code)
        cls_name = profile.model.cls_name or "GPTModel"
        if not hasattr(code_mod, cls_name):
            raise ModelLoadError(f"Model class {cls_name!r} not found in {profile.model.code}")
        model_cls = getattr(code_mod, cls_name)

        torch.manual_seed(profile.model.seed)
        model = model_cls(cfg)

        # Load weights with tolerance for derivable .mask buffers
        incompatible = model.load_state_dict(merged_state_dict, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.endswith(".mask")]
        unexpected = [k for k in incompatible.unexpected_keys if not k.endswith(".mask")]

        if missing or unexpected:
            hints: list[str] = []
            if any(".lora." in k for k in unexpected):
                hints.append(
                    "Checkpoint contains unmerged LoRA weights; specify [model.lora] in profile."
                )
            if any("trf_blocks." in k for k in missing + unexpected):
                hints.append(
                    "Layer count or block structure mismatch between model class and checkpoint."
                )

            err_msg = (
                f"State dict mismatch loading checkpoint {ckpt.source_path.name}.\n"
                f"Missing keys ({len(missing)}): {missing[:10]}\n"
                f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}"
            )
            if hints:
                err_msg += "\nHints:\n  - " + "\n  - ".join(hints)
            raise ModelLoadError(err_msg)

    # Path (c): class without checkpoint (seeded random initialization)
    else:
        infer_res = infer_config_from_state_dict({}, merged_config)
        cfg = infer_res.cfg
        notes.append("Initialized model randomly with seed (no checkpoint provided)")

        code_mod = import_model_module(profile.model.code)
        cls_name = profile.model.cls_name or "GPTModel"
        if not hasattr(code_mod, cls_name):
            raise ModelLoadError(f"Model class {cls_name!r} not found in {profile.model.code}")
        model_cls = getattr(code_mod, cls_name)

        torch.manual_seed(profile.model.seed)
        model = model_cls(cfg)

    # 4. Preparation for evaluation
    model.eval()
    model.requires_grad_(False)
    model.to(target_device)

    # 5. Architecture Adapter Selection
    _, spec, confidence = select_adapter(
        model,
        adapter_name=profile.adapter.name,
        overrides=profile.adapter.roles,
    )

    return BuiltModel(
        model=model,
        spec=spec,
        cfg=cfg,
        tokenizer=tokenizer,
        template=template,
        adapter_confidence=confidence,
        notes=tuple(notes),
    )
