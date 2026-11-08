"""Playlist filtering and StepRef generation.

Single responsibility: filter full Trace steps into a tailored presentation sequence.
Allowed dependencies: tilevis.core, stdlib.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel, Semantic, Stage, StepKind
from tilevis.core.trace import Fidelity, ModelSpecInfo, PassRecord, Step, Trace

logger = logging.getLogger("tilevis.render.playlist")


@dataclass(frozen=True, slots=True)
class StepRef:
    """Lightweight reference to either a recorded Trace step or a synthetic presentation card."""

    pass_index: int | None
    step_index: int | None
    step_id: str
    kind: str
    semantic: str
    stage: str
    block_index: int | None
    detail_level: int
    is_synthetic: bool = False


def _resolve_expanded_blocks(detail_blocks: str | list[int], n_layers: int) -> set[int]:
    """Resolve PresenterSettings.detail_blocks into 0-based block index set."""
    if n_layers <= 0:
        return set()

    if isinstance(detail_blocks, list):
        return {int(b) for b in detail_blocks if 0 <= int(b) < n_layers}

    key = str(detail_blocks).strip().lower()
    if key == "first":
        return {0}
    if key == "first_last":
        return {0, n_layers - 1}
    if key == "all":
        return set(range(n_layers))
    if key == "none":
        return set()

    logger.warning("Unknown detail_blocks value %r; defaulting to 'first'", detail_blocks)
    return {0}


def build_playlist(trace: Trace, settings: PresenterSettings) -> list[StepRef]:
    """Filter a recorded Trace according to PresenterSettings into an ordered StepRef playlist.

    Rules:
    - STORY detail collapses every block to its BLOCK_SUMMARY step.
    - STANDARD / DEEP expands blocks selected by detail_blocks (excluding BLOCK_SUMMARY) and
      collapses remaining blocks to BLOCK_SUMMARY.
    - Passes after pass 0 use settings.later_pass_level.
    - include_title / include_end prepend/append synthetic TITLE / END cards.
    """
    refs: list[StepRef] = []
    n_layers = trace.model.n_layers
    expanded_blocks = _resolve_expanded_blocks(settings.detail_blocks, n_layers)

    if settings.include_title:
        refs.append(
            StepRef(
                pass_index=None,
                step_index=None,
                step_id="__title__",
                kind="TITLE",
                semantic="title",
                stage=Stage.INPUT.value,
                block_index=None,
                detail_level=int(DetailLevel.STORY),
                is_synthetic=True,
            )
        )

    for pass_rec in trace.passes:
        pass_level = settings.detail_level if pass_rec.index == 0 else settings.later_pass_level
        blocks_with_summary = {
            s.block_index
            for s in pass_rec.steps
            if s.block_index is not None
            and (
                s.semantic == Semantic.BLOCK_SUMMARY.value or s.kind == StepKind.BLOCK_SUMMARY.value
            )
        }

        for s_idx, step in enumerate(pass_rec.steps):
            b_idx = step.block_index
            if b_idx is None:
                if step.detail_level > pass_level:
                    continue
            else:
                is_summary = (
                    step.semantic == Semantic.BLOCK_SUMMARY.value
                    or step.kind == StepKind.BLOCK_SUMMARY.value
                )
                is_expanded = (pass_level > int(DetailLevel.STORY)) and (b_idx in expanded_blocks)
                if is_expanded:
                    if is_summary or step.detail_level > pass_level:
                        continue
                else:
                    if b_idx in blocks_with_summary:
                        if not is_summary:
                            continue
                    elif step.detail_level > int(DetailLevel.STORY):
                        continue

            refs.append(
                StepRef(
                    pass_index=pass_rec.index,
                    step_index=s_idx,
                    step_id=step.id,
                    kind=step.kind,
                    semantic=step.semantic,
                    stage=step.stage,
                    block_index=b_idx,
                    detail_level=step.detail_level,
                    is_synthetic=False,
                )
            )

    if settings.include_end:
        refs.append(
            StepRef(
                pass_index=None,
                step_index=None,
                step_id="__end__",
                kind="END",
                semantic="end",
                stage=Stage.OUTPUT.value,
                block_index=None,
                detail_level=int(DetailLevel.STORY),
                is_synthetic=True,
            )
        )

    return refs


def _fallback_pass(trace: Trace, *, at_end: bool = False) -> PassRecord:
    """Return representative PassRecord for synthetic cards without crashing on empty passes."""
    if trace.passes:
        return trace.passes[-1] if at_end else trace.passes[0]
    return PassRecord(index=0, input_ids=(), input_tokens=(), text_before=trace.prompt, steps=())


def resolve_step(trace: Trace, ref: StepRef) -> tuple[Step, PassRecord]:
    """Resolve a StepRef to its concrete (Step, PassRecord) pair, synthesizing TITLE/END cards."""
    if not ref.is_synthetic and ref.pass_index is not None and ref.step_index is not None:
        pass_rec = trace.passes[ref.pass_index]
        return pass_rec.steps[ref.step_index], pass_rec

    model_info: ModelSpecInfo = trace.model
    if ref.kind == "TITLE" or ref.step_id == "__title__":
        pass_rec = _fallback_pass(trace, at_end=False)
        step = Step(
            id=ref.step_id,
            kind="TITLE",
            semantic="title",
            stage=Stage.INPUT.value,
            block_index=None,
            detail_level=int(DetailLevel.STORY),
            module_path=None,
            tensors=(),
            meta={
                "model_name": model_info.name,
                "prompt": trace.prompt,
                "n_layers": model_info.n_layers,
                "n_heads": model_info.n_heads,
                "emb_dim": model_info.emb_dim,
                "n_passes": len(trace.passes),
            },
            fidelity=Fidelity("exact", 0.0),
        )
        return step, pass_rec

    # Synthetic END card
    pass_rec = _fallback_pass(trace, at_end=True)
    generated_tokens = [p.prediction.chosen_token for p in trace.passes if p.prediction is not None]
    step = Step(
        id=ref.step_id,
        kind="END",
        semantic="end",
        stage=Stage.OUTPUT.value,
        block_index=None,
        detail_level=int(DetailLevel.STORY),
        module_path=None,
        tensors=(),
        meta={
            "model_name": model_info.name,
            "prompt": trace.prompt,
            "generated_tokens": generated_tokens,
            "n_passes": len(trace.passes),
        },
        fidelity=Fidelity("exact", 0.0),
    )
    return step, pass_rec
