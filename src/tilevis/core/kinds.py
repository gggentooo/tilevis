"""Step kinds, semantics, detail levels, stages, and tensor roles.

Single responsibility: enumerate visual and narrative vocabulary for Trace representation.
Allowed dependencies: stdlib only.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class StepKind(StrEnum):
    """Visual classification determining which Scene renderer is chosen."""

    PROMPT = "PROMPT"
    TOKENIZE = "TOKENIZE"
    EMBEDDING = "EMBEDDING"
    LAYERNORM = "LAYERNORM"
    LINEAR = "LINEAR"
    QKV_PROJECTION = "QKV_PROJECTION"
    ATTENTION_PATTERN = "ATTENTION_PATTERN"
    ATTENTION_MIX = "ATTENTION_MIX"
    ACTIVATION = "ACTIVATION"
    RESIDUAL_ADD = "RESIDUAL_ADD"
    BLOCK_SUMMARY = "BLOCK_SUMMARY"
    LOGITS = "LOGITS"
    NEXT_TOKEN = "NEXT_TOKEN"
    GENERIC = "GENERIC"


class Semantic(StrEnum):
    """Semantic meaning of a calculation step used for caption lookup."""

    PROMPT = "prompt"
    TOKENIZE = "tokenize"
    EMBED = "embed"
    BLOCK_NORM1 = "block.norm1"
    BLOCK_ATT_QKV = "block.att.qkv"
    BLOCK_ATT_SCORES = "block.att.scores"
    BLOCK_ATT_MIX = "block.att.mix"
    BLOCK_ATT_OUT = "block.att.out"
    BLOCK_RESID1 = "block.resid1"
    BLOCK_NORM2 = "block.norm2"
    BLOCK_FF_EXPAND = "block.ff.expand"
    BLOCK_FF_ACT = "block.ff.act"
    BLOCK_FF_CONTRACT = "block.ff.contract"
    BLOCK_RESID2 = "block.resid2"
    BLOCK_SUMMARY = "block.summary"
    FINAL_NORM = "final_norm"
    LM_HEAD = "lm_head"
    NEXT_TOKEN = "next_token"
    GENERIC = "generic"


class DetailLevel(IntEnum):
    """Presentation detail level controlling step inclusion."""

    STORY = 1
    STANDARD = 2
    DEEP = 3


class Stage(StrEnum):
    """High-level model pipeline stage used for minimap navigation."""

    INPUT = "INPUT"
    BLOCK = "BLOCK"
    OUTPUT = "OUTPUT"


class TensorRole(StrEnum):
    """Semantic role of a tensor within a computation step."""

    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    BIAS = "bias"
    SCALE = "scale"
    SHIFT = "shift"
    SHORTCUT = "shortcut"
    UPDATE = "update"
    QUERIES = "queries"
    KEYS = "keys"
    VALUES = "values"
    ATTN_WEIGHTS = "attn_weights"
    TOKEN_VECTORS = "token_vectors"
    POSITION_VECTORS = "position_vectors"
    LOGITS = "logits"
    PROBS = "probs"
    EXTRA = "extra"


def parse_kind(value: str) -> StepKind | str:
    """Parse a string into a StepKind enum member, tolerating unknown forward-compatible kinds.

    Args:
        value: String representation of a step kind.

    Returns:
        The matching StepKind enum member if recognized, otherwise the original string unchanged.
    """
    normalized = value.strip().upper()
    try:
        return StepKind(normalized)
    except ValueError:
        pass
    try:
        return StepKind[normalized]
    except KeyError:
        return value
