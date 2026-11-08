"""Model profile loading and parsing.

Single responsibility: load and validate TOML model profiles into ModelProfile dataclasses.
Allowed dependencies: stdlib and tilevis.core only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tilevis.core.config import load_toml, warn_unknown_keys
from tilevis.core.errors import ModelLoadError


@dataclass(frozen=True, slots=True)
class ModelLoraConfig:
    """LoRA configuration parameters."""

    alpha: float = 16.0


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Model code, checkpoint, class/factory and instantiation settings."""

    code: Path
    cls_name: str | None = "GPTModel"
    checkpoint: Path | None = None
    factory: str | None = None
    allow_pickle: bool = False
    seed: int = 123
    config: Mapping[str, Any] = field(default_factory=dict)
    lora: ModelLoraConfig | None = None


@dataclass(frozen=True, slots=True)
class TextSettings:
    """Tokenizer and prompt templating settings."""

    tokenizer: str = "gpt2"
    template: str = "raw"


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    """Architecture adapter selection and role overrides."""

    name: str = "auto"
    roles: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Complete specification describing a teammate model configuration."""

    name: str
    model: ModelSettings
    text: TextSettings = field(default_factory=TextSettings)
    adapter: AdapterSettings = field(default_factory=AdapterSettings)
    source_path: Path | None = None


_TOP_LEVEL_KEYS = {"profile", "model", "text", "adapter"}
_MODEL_KEYS = {"code", "class", "checkpoint", "factory", "allow_pickle", "seed", "config", "lora"}
_TEXT_KEYS = {"tokenizer", "template"}
_ADAPTER_KEYS = {"name", "roles"}


def load_profile(path: Path | str) -> ModelProfile:
    """Load and parse a model profile from a TOML file.

    Relative paths in the profile (for code and checkpoint) are resolved
    relative to the profile file itself.

    Args:
        path: Path to TOML profile file.

    Returns:
        Validated ModelProfile instance.

    Raises:
        ModelLoadError: If required fields are missing or corrupted.
    """
    profile_path = Path(path).resolve()
    if not profile_path.exists() or not profile_path.is_file():
        raise ModelLoadError(f"Model profile not found: {profile_path}")

    raw_data = load_toml(profile_path)
    warn_unknown_keys(raw_data, _TOP_LEVEL_KEYS, str(profile_path))

    # 1. Profile metadata
    p_meta = raw_data.get("profile", {})
    profile_name = p_meta.get("name", profile_path.stem)

    # 2. Model configuration
    m_raw = raw_data.get("model", {})
    warn_unknown_keys(m_raw, _MODEL_KEYS, f"{profile_path}: [model]")

    profile_dir = profile_path.parent

    # Resolve model code path
    code_raw = m_raw.get("code")
    if code_raw is not None:
        code_path = (profile_dir / code_raw).resolve()
    else:
        # Default to previous_chapters.py in repo
        code_path = (
            profile_dir / ".." / "ch07" / "01_main-chapter-code" / "previous_chapters.py"
        ).resolve()

    # Resolve checkpoint path if provided
    ckpt_raw = m_raw.get("checkpoint")
    ckpt_path = (profile_dir / ckpt_raw).resolve() if ckpt_raw is not None else None

    # LoRA config
    lora_raw = m_raw.get("lora")
    lora_cfg = None
    if lora_raw is not None:
        alpha_val = float(lora_raw.get("alpha", 16.0))
        lora_cfg = ModelLoraConfig(alpha=alpha_val)

    model_settings = ModelSettings(
        code=code_path,
        cls_name=m_raw.get("class", "GPTModel"),
        checkpoint=ckpt_path,
        factory=m_raw.get("factory"),
        allow_pickle=bool(m_raw.get("allow_pickle", False)),
        seed=int(m_raw.get("seed", 123)),
        config=dict(m_raw.get("config", {})),
        lora=lora_cfg,
    )

    # 3. Text configuration
    t_raw = raw_data.get("text", {})
    warn_unknown_keys(t_raw, _TEXT_KEYS, f"{profile_path}: [text]")
    text_settings = TextSettings(
        tokenizer=t_raw.get("tokenizer", "gpt2"),
        template=t_raw.get("template", "raw"),
    )

    # 4. Adapter configuration
    a_raw = raw_data.get("adapter", {})
    warn_unknown_keys(a_raw, _ADAPTER_KEYS, f"{profile_path}: [adapter]")
    adapter_settings = AdapterSettings(
        name=a_raw.get("name", "auto"),
        roles=dict(a_raw.get("roles", {})),
    )

    return ModelProfile(
        name=profile_name,
        model=model_settings,
        text=text_settings,
        adapter=adapter_settings,
        source_path=profile_path,
    )
