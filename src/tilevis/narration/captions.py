"""Caption provider for translating steps and semantics into friendly explanations.

Single responsibility: format caption titles, bodies, and nerd-mode details.
Allowed dependencies: tilevis.core and stdlib only.
"""

from __future__ import annotations

import logging
import string
import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import Any

from tilevis.core.trace import PassRecord, Step, Trace

logger = logging.getLogger("tilevis.narration.captions")

_FORMATTER = string.Formatter()


@dataclass(frozen=True, slots=True)
class Caption:
    """Formatted human-friendly and nerd-mode explanations for a single step."""

    title: str
    body: str
    nerd: str


class _SafeMissing:
    """Sentinel representing an unavailable template placeholder."""


_MISSING = _SafeMissing()


def _extract_ff_dim(step: Step, pass_record: PassRecord, trace: Trace) -> int | None:
    """Resolve feed-forward hidden dimension from step, pass tensors, or trace metadata."""
    meta_val = step.meta.get("ff_dim")
    if isinstance(meta_val, int):
        return meta_val

    for snap in step.tensors:
        for ax in snap.axes:
            if ax.name == "ff_dim":
                return int(ax.full_size)

    for s in pass_record.steps:
        for snap in s.tensors:
            for ax in snap.axes:
                if ax.name == "ff_dim":
                    return int(ax.full_size)

    ff_mult = trace.model.extras.get("ff_mult") if trace.model.extras else None
    if isinstance(ff_mult, (int, float)):
        return round(trace.model.emb_dim * ff_mult)

    return None


def _build_placeholders(step: Step, trace: Trace, pass_record: PassRecord) -> dict[str, Any]:
    """Build placeholder dictionary for caption template substitution."""
    block_num: Any = step.block_index + 1 if step.block_index is not None else _MISSING
    n_heads_raw = step.meta.get("n_heads")
    n_heads: Any = n_heads_raw if n_heads_raw is not None else trace.model.n_heads
    if n_heads is None:
        n_heads = _MISSING

    emb_dim_raw = step.meta.get("emb_dim")
    emb_dim = emb_dim_raw if emb_dim_raw is not None else trace.model.emb_dim

    ff_dim_val = _extract_ff_dim(step, pass_record, trace)
    ff_dim: Any = ff_dim_val if ff_dim_val is not None else _MISSING

    if pass_record.input_tokens:
        token: Any = pass_record.input_tokens[-1]
    elif step.meta.get("chosen_token") is not None:
        token = str(step.meta["chosen_token"])
    else:
        token = _MISSING

    top_word: Any = _MISSING
    top_prob: Any = _MISSING
    if pass_record.prediction is not None:
        pred = pass_record.prediction
        if pred.candidate_tokens:
            top_word = pred.candidate_tokens[0]
        elif pred.chosen_token:
            top_word = pred.chosen_token
        if pred.candidate_probs:
            top_prob = float(pred.candidate_probs[0])
    else:
        meta_cands = step.meta.get("candidate_tokens")
        meta_probs = step.meta.get("candidate_probs")
        if isinstance(meta_cands, (list, tuple)) and meta_cands:
            top_word = str(meta_cands[0])
        elif isinstance(step.meta.get("chosen_token"), str):
            top_word = step.meta["chosen_token"]
        if isinstance(meta_probs, (list, tuple)) and meta_probs:
            top_prob = float(meta_probs[0])

    return {
        "block_num": block_num,
        "n_blocks": trace.model.n_layers,
        "n_heads": n_heads,
        "emb_dim": emb_dim,
        "ff_dim": ff_dim,
        "token": token,
        "top_word": top_word,
        "top_prob": top_prob,
    }


def _safe_format(template: str, values: dict[str, Any], key_ctx: str) -> str:
    """Format template string safely, replacing missing placeholders with '?'."""
    parts: list[str] = []
    for literal_text, field_name, format_spec, conversion in _FORMATTER.parse(template):
        if literal_text:
            parts.append(literal_text)
        if field_name is None:
            continue

        root_field = field_name.split(".")[0].split("[")[0]
        val = values.get(root_field, _MISSING)
        if isinstance(val, _SafeMissing) or val is None:
            logger.warning(
                "Missing placeholder {%s} while formatting caption %r; rendering as '?'",
                field_name,
                key_ctx,
            )
            parts.append("?")
            continue

        try:
            obj: Any = val
            if conversion == "r":
                obj = repr(obj)
            elif conversion == "s":
                obj = str(obj)
            elif conversion == "a":
                obj = ascii(obj)
            parts.append(format(obj, format_spec or ""))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to format placeholder {%s:%s} on %r (%s); rendering as '?'",
                field_name,
                format_spec,
                key_ctx,
                exc,
            )
            parts.append("?")

    return "".join(parts)


class CaptionProvider:
    """Provides audience-friendly titles, bodies, and nerd details for trace steps."""

    def __init__(self, locale: str = "en") -> None:
        self.locale = locale
        self._entries: dict[str, dict[str, str]] = {}
        self._entries_lower: dict[str, str] = {}
        self._load_locale(locale)

    def _load_locale(self, locale: str) -> None:
        """Load caption TOML file for requested locale with English fallback."""
        pkg = resources.files("tilevis.narration.locales")
        target = pkg.joinpath(f"{locale}.toml")
        if not target.is_file():
            logger.warning("Caption locale %r not found; falling back to 'en'", locale)
            target = pkg.joinpath("en.toml")

        raw_text = target.read_text(encoding="utf-8")
        data = tomllib.loads(raw_text)
        for key, val in data.items():
            if isinstance(val, dict):
                self._entries[key] = {
                    "title": str(val.get("title", "Computation Step")),
                    "body": str(val.get("body", "")),
                    "nerd": str(val.get("nerd", "")),
                }
                self._entries_lower[key.lower()] = key

    def _lookup_entry(self, step: Step) -> tuple[str, dict[str, str]]:
        """Resolve template entry using priority: semantic -> kind -> generic."""
        candidates = [
            step.semantic,
            step.semantic.lower() if step.semantic else "",
            step.kind,
            step.kind.upper() if step.kind else "",
            step.kind.lower() if step.kind else "",
        ]
        for cand in candidates:
            if not cand:
                continue
            if cand in self._entries:
                return cand, self._entries[cand]
            mapped = self._entries_lower.get(cand.lower())
            if mapped is not None:
                return mapped, self._entries[mapped]

        logger.warning(
            "No caption entry found for semantic=%r kind=%r; using 'generic' fallback",
            step.semantic,
            step.kind,
        )
        fallback = self._entries.get(
            "generic",
            {
                "title": "Computation Step",
                "body": "The model processes vectors through this layer.",
                "nerd": "Generic computation step.",
            },
        )
        return "generic", fallback

    def caption(self, step: Step, trace: Trace, pass_record: PassRecord) -> Caption:
        """Produce formatted Caption for the given step within trace and pass context."""
        key_used, entry = self._lookup_entry(step)
        placeholders = _build_placeholders(step, trace, pass_record)

        title = _safe_format(entry["title"], placeholders, f"{key_used}.title")
        body = _safe_format(entry["body"], placeholders, f"{key_used}.body")
        nerd_base = _safe_format(entry["nerd"], placeholders, f"{key_used}.nerd")

        # Append extra contextual stats for nerd mode when available
        extras: list[str] = []
        if step.module_path:
            extras.append(f"module={step.module_path}")
        if step.fidelity.status != "exact" or step.fidelity.max_abs_err is not None:
            err_s = (
                f", max_err={step.fidelity.max_abs_err:.2e}"
                if step.fidelity.max_abs_err is not None
                else ""
            )
            extras.append(f"fidelity={step.fidelity.status}{err_s}")

        nerd = f"{nerd_base} ({', '.join(extras)})" if extras else nerd_base
        return Caption(title=title, body=body, nerd=nerd)
