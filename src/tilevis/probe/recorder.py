"""Trace recorder coordinating forward capture, step builders, and Trace assembly.

Single responsibility: orchestrate multi-pass model generation under HookManager,
invoke step builders in deterministic template order, and return a validated Trace.
Allowed dependencies: stdlib, numpy, torch, tilevis.core, tilevis.probe.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from datetime import UTC, datetime

import torch
import torch.nn as nn

from tilevis import __version__
from tilevis.core.config import RecordSettings
from tilevis.core.kinds import Semantic
from tilevis.core.trace import (
    TRACE_SCHEMA_VERSION,
    ModelSpecInfo,
    PassRecord,
    Step,
    Trace,
)
from tilevis.core.validation import validate_trace
from tilevis.probe.adapters.base import ModelSpec
from tilevis.probe.builders.attention import AttentionStepBuilder
from tilevis.probe.builders.base import STEP_BUILDERS, StepBuildContext
from tilevis.probe.capture import CaptureBuffer
from tilevis.probe.generation import generate_passes
from tilevis.probe.hooks import HookManager
from tilevis.probe.loading.factory import BuiltModel
from tilevis.probe.reduce import IndexPlan
from tilevis.probe.text.tokenizer import validate_token_ids

logger = logging.getLogger(__name__)


def _gather_hook_paths(spec: ModelSpec, model: nn.Module) -> tuple[list[str], list[str]]:
    """Determine submodules requiring pre-hooks (inputs) and post-hooks (outputs)."""
    if spec.adapter == "generic":
        # Hook all leaf modules
        leaf_paths: list[str] = []
        for name, mod in model.named_modules():
            if len(list(mod.children())) == 0 and name:
                leaf_paths.append(name)
        return leaf_paths, leaf_paths

    pre_paths: list[str] = []
    post_paths: list[str] = []

    # Embeddings
    if spec.token_embedding:
        post_paths.append(spec.token_embedding)
    if spec.position_embedding:
        post_paths.append(spec.position_embedding)

    # Per block
    for block in spec.blocks:
        pre_paths.append(block.path)  # Shortcut input
        post_paths.append(block.path)  # Block output

        if block.norm1:
            post_paths.append(block.norm1)
        if block.q:
            post_paths.append(block.q)
        if block.k:
            post_paths.append(block.k)
        if block.v:
            post_paths.append(block.v)
        if block.out_proj:
            pre_paths.append(block.out_proj)  # ctx_true
            post_paths.append(block.out_proj)
        if block.attention:
            post_paths.append(block.attention)

        if block.norm2:
            pre_paths.append(block.norm2)  # resid1 output
            post_paths.append(block.norm2)

        if block.ff_in:
            pre_paths.append(block.ff_in)
            post_paths.append(block.ff_in)
        if block.ff_act:
            post_paths.append(block.ff_act)
        if block.ff_out:
            post_paths.append(block.ff_out)
        if block.ff:
            post_paths.append(block.ff)

    # Final norm & LM head
    if spec.final_norm:
        pre_paths.append(spec.final_norm)
        post_paths.append(spec.final_norm)
    if spec.lm_head:
        pre_paths.append(spec.lm_head)
        post_paths.append(spec.lm_head)

    # Deduplicate while preserving order
    unique_pre = list(dict.fromkeys(p for p in pre_paths if p))
    unique_post = list(dict.fromkeys(p for p in post_paths if p))
    return unique_pre, unique_post


class TraceRecorder:
    """Orchestrates model execution under hooks and assembles validated Traces."""

    def __init__(self, built_model: BuiltModel, settings: RecordSettings | None = None) -> None:
        self.built_model = built_model
        self.model = built_model.model
        self.spec = built_model.spec
        self.settings = settings or RecordSettings()
        self.tokenizer = built_model.tokenizer
        self.template = built_model.template

    def record(self, prompt: str) -> Trace:
        """Record model execution starting from prompt into an immutable Trace.

        Args:
            prompt: Text prompt to seed generation.

        Returns:
            Validated immutable Trace object.
        """
        formatted_prompt = self.template.format(prompt)
        prompt_ids = self.tokenizer.encode(formatted_prompt)

        # Validate token IDs against vocab size
        validate_token_ids(prompt_ids, self.spec.vocab_size)

        buffer = CaptureBuffer()
        pre_paths, post_paths = _gather_hook_paths(self.spec, self.model)
        hook_mgr = HookManager(self.model, buffer, pre_paths, post_paths)

        index_plan = IndexPlan.from_model_spec(
            self.spec,
            max_features=self.settings.max_features,
            max_tokens=self.settings.max_tokens,
            max_heads=self.settings.max_heads,
        )

        pass_records: list[PassRecord] = []
        spec = self.spec

        logger.info(
            "Recording trace for %d new tokens with strategy=%s",
            self.settings.new_tokens,
            self.settings.strategy,
        )

        with hook_mgr:

            def forward_under_hooks(input_tensor: torch.Tensor) -> torch.Tensor:
                buffer.clear()
                hook_mgr.reset_counts()
                out = self.model(input_tensor)
                assert isinstance(out, torch.Tensor)
                return out

            passes_iter = generate_passes(
                model=self.model,
                prompt_ids=prompt_ids,
                tokenizer=self.tokenizer,
                settings=self.settings,
                context_size=spec.context_length,
                forward_fn=forward_under_hooks,
            )

            for pass_res in passes_iter:
                ctx = StepBuildContext(
                    pass_index=pass_res.pass_index,
                    input_ids=pass_res.input_ids,
                    input_tokens=pass_res.input_tokens,
                    text_before=pass_res.text_before,
                    tokenizer=self.tokenizer,
                    prediction=pass_res.prediction,
                    pass_meta={"context_cropped": pass_res.context_cropped},
                )

                steps: list[Step] = []

                if spec.adapter == "generic":
                    # Generic builder flow
                    b_gen = STEP_BUILDERS.get(Semantic.GENERIC.value)
                    steps.extend(b_gen.build(spec, buffer, self.model, index_plan, ctx))

                    # Next token step
                    b_out = STEP_BUILDERS.get(Semantic.NEXT_TOKEN.value)
                    steps.extend(
                        b_out.build(
                            spec,
                            buffer,
                            self.model,
                            index_plan,
                            ctx,
                            semantic=Semantic.NEXT_TOKEN.value,
                        )
                    )
                else:
                    # Template order (§6.1.11)
                    # 1. prompt
                    steps.extend(
                        STEP_BUILDERS.get(Semantic.PROMPT.value).build(
                            spec, buffer, self.model, index_plan, ctx
                        )
                    )
                    # 2. tokenize
                    steps.extend(
                        STEP_BUILDERS.get(Semantic.TOKENIZE.value).build(
                            spec, buffer, self.model, index_plan, ctx
                        )
                    )
                    # 3. embed
                    steps.extend(
                        STEP_BUILDERS.get(Semantic.EMBED.value).build(
                            spec, buffer, self.model, index_plan, ctx
                        )
                    )

                    # 4. For each block b:
                    att_builder = STEP_BUILDERS.get(Semantic.BLOCK_ATT_SCORES.value)
                    for b in spec.blocks:
                        # summary (STORY)
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_SUMMARY.value).build(
                                spec, buffer, self.model, index_plan, ctx, block=b
                            )
                        )
                        # norm1
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_NORM1.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_NORM1.value,
                            )
                        )
                        # att (att.qkv, att.scores, att.mix, att.out)
                        steps.extend(
                            att_builder.build(spec, buffer, self.model, index_plan, ctx, block=b)
                        )
                        # resid1
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_RESID1.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_RESID1.value,
                            )
                        )
                        # norm2
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_NORM2.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_NORM2.value,
                            )
                        )
                        # ff.expand
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_FF_EXPAND.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_FF_EXPAND.value,
                            )
                        )
                        # ff.act
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_FF_ACT.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_FF_ACT.value,
                            )
                        )
                        # ff.contract
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_FF_CONTRACT.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_FF_CONTRACT.value,
                            )
                        )
                        # resid2
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.BLOCK_RESID2.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                block=b,
                                semantic=Semantic.BLOCK_RESID2.value,
                            )
                        )

                    # Update detected n_heads if attention builder discovered it
                    if (
                        isinstance(att_builder, AttentionStepBuilder)
                        and att_builder.detected_n_heads
                    ):
                        spec = replace(spec, n_heads=att_builder.detected_n_heads)

                    # 5. final_norm
                    if spec.final_norm:
                        steps.extend(
                            STEP_BUILDERS.get(Semantic.FINAL_NORM.value).build(
                                spec,
                                buffer,
                                self.model,
                                index_plan,
                                ctx,
                                semantic=Semantic.FINAL_NORM.value,
                            )
                        )

                    # 6. lm_head
                    steps.extend(
                        STEP_BUILDERS.get(Semantic.LM_HEAD.value).build(
                            spec,
                            buffer,
                            self.model,
                            index_plan,
                            ctx,
                            semantic=Semantic.LM_HEAD.value,
                        )
                    )

                    # 7. next_token
                    steps.extend(
                        STEP_BUILDERS.get(Semantic.NEXT_TOKEN.value).build(
                            spec,
                            buffer,
                            self.model,
                            index_plan,
                            ctx,
                            semantic=Semantic.NEXT_TOKEN.value,
                        )
                    )

                pass_records.append(
                    PassRecord(
                        index=pass_res.pass_index,
                        input_ids=pass_res.input_ids,
                        input_tokens=pass_res.input_tokens,
                        text_before=pass_res.text_before,
                        steps=tuple(steps),
                        prediction=pass_res.prediction,
                    )
                )

        # Assemble Trace
        n_params = sum(p.numel() for p in self.model.parameters())
        n_layers = len(spec.blocks) if len(spec.blocks) > 0 else 1
        model_info = ModelSpecInfo(
            name=self.built_model.cfg.get("name", "gpt_model"),
            adapter=spec.adapter,
            n_layers=n_layers,
            emb_dim=spec.emb_dim,
            n_heads=spec.n_heads,
            vocab_size=spec.vocab_size,
            context_length=spec.context_length,
            n_params=n_params,
            extras={"notes": list(self.built_model.notes)},
        )

        trace = Trace(
            schema_version=TRACE_SCHEMA_VERSION,
            tilevis_version=__version__,
            created_at=datetime.now(UTC).isoformat(),
            model=model_info,
            prompt=prompt,
            settings=asdict(self.settings),
            passes=tuple(pass_records),
        )

        # Strict validation
        validate_trace(trace, strict=True)
        return trace
