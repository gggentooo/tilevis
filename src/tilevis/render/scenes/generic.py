"""Fallback generic scene rendering all tensors in a step as labeled tile grids.

Single responsibility: robust fallback visualization for custom or unsupported operations.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from tilevis.core.kinds import TensorRole
from tilevis.core.trace import AxisInfo, Step, TensorSnapshot
from tilevis.render.animation import Timeline
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext, Tooltip
from tilevis.render.tiles import TileGrid

_UNSIGNED_ROLES: frozenset[str] = frozenset(
    {
        TensorRole.ATTN_WEIGHTS.value,
        TensorRole.PROBS.value,
    }
)


@dataclass(slots=True)
class _TensorPanel:
    """Internal layout entry holding one tensor card and its cached TileGrid."""

    label: str
    sublabel: str
    card_rect: pygame.Rect
    grid: TileGrid
    note: str


class GenericScene:
    """Universal scene displaying any Step's tensors (or card metadata) inside stage."""

    def __init__(self) -> None:
        self._step: Step | None = None
        self._ctx: SceneContext | None = None
        self._timeline: Timeline = Timeline(duration=0.0)
        self._panels: list[_TensorPanel] = []
        self._cached_stage_size: tuple[int, int] | None = None
        self._cached_head: int | None = None
        self._head_index: int = 0
        self._head_count: int = 0
        self._average_heads: bool = False
        self.downsample_notes: list[str] = []

    def enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        """Initialize state and animation timeline for the new step."""
        self._step = step
        self._ctx = ctx
        self._panels = []
        self._cached_stage_size = None
        self._cached_head = None
        self.downsample_notes = []

        # Detect if any snapshot is 3-D attention heads (H, T, T)
        self._head_count = 0
        for snap in step.tensors:
            if snap.data.ndim == 3 and snap.data.shape[0] > 0:
                self._head_count = int(snap.data.shape[0])
                break

        if ctx.active_head is not None and self._head_count > 0:
            self._head_index = ctx.active_head % self._head_count
            self._average_heads = False
        else:
            self._head_index = 0
            self._average_heads = False

        speed = max(0.0, float(ctx.animation_speed) * ctx.theme.animation_speed_factor)
        duration = 0.55 if (animate and speed > 0.0) else 0.0
        self._timeline = Timeline(duration=duration, speed=speed)
        if not animate:
            self._timeline.finish()

    def update(self, dt: float) -> None:
        """Advance scene timeline by dt seconds."""
        self._timeline.update(dt)

    def finish(self) -> None:
        """Jump immediately to completed state."""
        self._timeline.finish()

    @property
    def is_animating(self) -> bool:
        """True while step animation has not completed."""
        return not self._timeline.is_finished

    def handle_command(self, cmd: Command) -> bool:
        """Handle optional head selection commands if 3-D tensors are present."""
        if self._head_count <= 0:
            return False

        if cmd == Command.CYCLE_HEAD:
            self._average_heads = False
            self._head_index = (self._head_index + 1) % self._head_count
            self._cached_stage_size = None
            return True
        if cmd == Command.CYCLE_HEAD_PREV:
            self._average_heads = False
            self._head_index = (self._head_index - 1) % self._head_count
            self._cached_stage_size = None
            return True
        if cmd == Command.AVG_HEADS:
            self._average_heads = not self._average_heads
            self._cached_stage_size = None
            return True
        return False

    def hit_test(self, pos: tuple[int, int]) -> Tooltip | None:
        """Return Tooltip for tile hovered under cursor position."""
        for panel in self._panels:
            hit = panel.grid.cell_at(pos)
            if hit is not None:
                row_part = (
                    f"{hit.row_label}" if hit.row_label is not None else f"row {hit.orig_row}"
                )
                col_part = (
                    f"{hit.col_label}" if hit.col_label is not None else f"dim {hit.orig_col}"
                )
                return Tooltip(
                    title=panel.label,
                    lines=(f"{row_part} · {col_part}", f"value: {hit.value:+.4f}"),
                    pos=pos,
                )
        return None

    def _slice_snapshot(self, snap: TensorSnapshot) -> tuple[np.ndarray, tuple[AxisInfo, ...], str]:
        """Prepare 1-D/2-D slice and axis metadata (slicing 3-D attention heads safely)."""
        if snap.data.ndim != 3:
            shape_str = "x".join(str(s) for s in snap.full_shape)
            return snap.data, snap.axes, shape_str

        h_total = int(snap.data.shape[0])
        if self._average_heads:
            arr2d = np.mean(snap.data, axis=0, dtype=np.float32)
            head_tag = f"avg of {h_total} heads"
        else:
            idx = self._head_index % max(1, h_total)
            arr2d = snap.data[idx]
            head_tag = f"head {idx + 1}/{h_total}"
        axes2 = (snap.axes[1], snap.axes[2]) if len(snap.axes) >= 3 else snap.axes[-2:]
        shape_str = f"{head_tag} · {snap.full_shape[-2]}x{snap.full_shape[-1]}"
        return arr2d, axes2, shape_str

    def _rebuild_panels(self, stage: pygame.Rect, scale: UiScale) -> None:
        """Lay out tensor TileGrid panels to fit stage cleanly without overlap."""
        self._panels = []
        self.downsample_notes = []
        self._cached_stage_size = stage.size
        self._cached_head = None if self._average_heads else self._head_index

        step = self._step
        ctx = self._ctx
        if step is None or ctx is None or not step.tensors:
            return

        tensors = step.tensors
        n = len(tensors)
        cols = min(3, n) if stage.width >= stage.height else min(2, n)
        cols = max(1, cols)
        rows = max(1, math.ceil(n / cols))

        if stage.width >= 320 and stage.height >= 240:
            outer_pad = max(4, min(scale.px(18, minimum=4), stage.width // 20, stage.height // 20))
        else:
            outer_pad = 2

        gap = outer_pad
        avail_w = max(cols, stage.width - 2 * outer_pad - gap * (cols - 1))
        avail_h = max(rows, stage.height - 2 * outer_pad - gap * (rows - 1))
        cell_w = max(1, avail_w // cols)
        cell_h = max(1, avail_h // rows)

        header_h = min(max(16, scale.px(44, minimum=16)), max(16, cell_h // 3))
        inner_pad = max(2, min(scale.px(12, minimum=2), cell_w // 12, cell_h // 12))

        for i, snap in enumerate(tensors):
            r = i // cols
            c = i % cols
            rx = stage.x + outer_pad + c * (cell_w + gap)
            ry = stage.y + outer_pad + r * (cell_h + gap)
            card_rect = pygame.Rect(rx, ry, cell_w, cell_h)

            arr2d, axes2, shape_str = self._slice_snapshot(snap)
            if arr2d.size == 0 or any(dim <= 0 for dim in arr2d.shape):
                continue
            signed = snap.role not in _UNSIGNED_ROLES
            cmap_name = ctx.theme.diverging_cmap if signed else ctx.theme.sequential_cmap
            cmap = (
                ctx.colormaps.get(cmap_name)
                if cmap_name in ctx.colormaps
                else ctx.colormaps.get("diverging_blue_orange")
            )
            norm_scale = ctx.normalizer.compute_scale(snap, step, ctx.pass_record)

            grid_w = max(1, cell_w - 2 * inner_pad)
            grid_h = max(1, cell_h - header_h - 2 * inner_pad)

            min_tile_px = max(1, min(8, grid_w, grid_h))
            pref_cell_px = max(min_tile_px, scale.px(20, minimum=8))

            grid = TileGrid(
                data=arr2d,
                axes=axes2,
                colormap=cmap,
                scale=norm_scale,
                cell_px=pref_cell_px,
                gap_px=ctx.theme.tile_gap_px,
                signed=signed,
                min_cell_px=min_tile_px,
                max_display_size=(grid_w, grid_h),
                gap_color=ctx.theme.panel,
            )

            # Center TileGrid within the lower panel space
            gx = card_rect.x + inner_pad + max(0, (grid_w - grid.rect.width) // 2)
            gy = card_rect.y + header_h + inner_pad + max(0, (grid_h - grid.rect.height) // 2)
            grid.set_position(gx, gy)

            note = grid.downsample_note
            if note:
                self.downsample_notes.append(note)

            role_label = snap.role.replace("_", " ").title()
            title = f"{role_label}: {snap.name}"
            sub = f"shape ({shape_str})" + (f" · {note}" if note else "")

            self._panels.append(
                _TensorPanel(
                    label=title,
                    sublabel=sub,
                    card_rect=card_rect,
                    grid=grid,
                    note=note,
                )
            )

    def _draw_text_card(self, surface: pygame.Surface, stage: pygame.Rect, scale: UiScale) -> None:
        """Render clean summary card when a step carries metadata only (e.g. PROMPT/TITLE/END)."""
        step = self._step
        ctx = self._ctx
        if step is None or ctx is None:
            return

        theme = ctx.theme
        margin = max(8, min(scale.px(48, minimum=8), stage.width // 10, stage.height // 10))
        box = pygame.Rect(
            stage.x + margin,
            stage.y + margin,
            max(1, stage.width - 2 * margin),
            max(1, stage.height - 2 * margin),
        )
        pygame.draw.rect(
            surface, theme.panel, box, border_radius=max(2, scale.px(theme.corner_radius))
        )
        pygame.draw.rect(
            surface,
            theme.border,
            box,
            width=max(1, scale.px(2)),
            border_radius=max(2, scale.px(theme.corner_radius)),
        )

        # Animated progress accent stripe
        prog = self._timeline.eased("ease_out_cubic")
        bar_h = max(3, scale.px(6))
        bar_w = max(1, int((box.width - 2 * margin) * prog))
        pygame.draw.rect(
            surface,
            theme.accent,
            pygame.Rect(box.x + margin, box.y + margin // 2, bar_w, bar_h),
            border_radius=max(1, bar_h // 2),
        )

        # Select primary heading & details from metadata
        meta = step.meta
        if step.kind == "TITLE":
            heading = str(meta.get("model_name", ctx.trace.model.name))
            n_l = meta.get("n_layers", ctx.trace.model.n_layers)
            n_h = meta.get("n_heads", ctx.trace.model.n_heads)
            emb = meta.get("emb_dim", ctx.trace.model.emb_dim)
            heads_txt = f"{n_h} heads" if n_h is not None else "dynamic heads"
            subtitle = f"{n_l} layers · {heads_txt} · {emb} numbers per token"
            body_text = f'Prompt: "{meta.get("prompt", ctx.trace.prompt)}"'
        elif step.kind == "END":
            heading = "Generated Output"
            gen = meta.get("generated_tokens")
            gen_str = (
                "".join(str(t) for t in gen).replace("·", " ") if isinstance(gen, list) else ""
            )
            subtitle = f"Completed {meta.get('n_passes', len(ctx.trace.passes))} forward pass(es)"
            body_text = f'"{ctx.trace.prompt}{gen_str}"'
        else:
            heading = f"Step: {step.kind}"
            prompt_val = meta.get("text") or meta.get("prompt") or ctx.pass_record.text_before
            subtitle = f"Tokens: {len(ctx.pass_record.input_tokens)}"
            body_text = f'"{prompt_val}"'

        title_px = min(scale.font_px(theme.title_font_px, minimum=14), max(14, box.height // 6))
        body_px = min(scale.font_px(theme.body_font_px, minimum=11), max(11, box.height // 10))
        pad = max(8, scale.px(32, minimum=8))
        content_w = max(20, box.width - 2 * pad)

        cur_y = box.y + pad + bar_h
        h_surf = ctx.fonts.render_text(
            heading, title_px, theme.text_primary, bold=True, max_width=content_w
        )
        surface.blit(h_surf, (box.x + pad, cur_y))
        cur_y += h_surf.get_height() + max(4, scale.px(10))

        sub_surf = ctx.fonts.render_text(subtitle, body_px, theme.accent, max_width=content_w)
        surface.blit(sub_surf, (box.x + pad, cur_y))
        cur_y += sub_surf.get_height() + max(6, scale.px(16))

        remaining_h = max(1, box.bottom - pad - cur_y)
        line_h = max(12, ctx.fonts.get(body_px).get_linesize())
        max_lines = max(1, remaining_h // line_h)
        for line in ctx.fonts.wrap_text(body_text, content_w, body_px, max_lines=max_lines):
            l_surf = ctx.fonts.render_text(line, body_px, theme.text_secondary)
            surface.blit(l_surf, (box.x + pad, cur_y))
            cur_y += line_h

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render step tile panels (or summary card) inside stage."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return

        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)

        if not self._step.tensors:
            self._draw_text_card(surface, stage, scale)
            return

        cur_head = None if self._average_heads else self._head_index
        if self._cached_stage_size != stage.size or self._cached_head != cur_head:
            self._rebuild_panels(stage, scale)

        if not self._panels:
            self._draw_text_card(surface, stage, scale)
            return

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.eased("ease_out_cubic")

        title_px = max(10, min(scale.font_px(22, minimum=10), max(10, stage.height // 24)))
        sub_px = max(9, min(scale.font_px(16, minimum=9), title_px - 2))

        for panel in self._panels:
            c_rect = panel.card_rect
            pygame.draw.rect(
                surface,
                theme.panel,
                c_rect,
                border_radius=max(2, scale.px(theme.corner_radius)),
            )
            pygame.draw.rect(
                surface,
                theme.border,
                c_rect,
                width=1,
                border_radius=max(2, scale.px(theme.corner_radius)),
            )

            pad = max(2, min(scale.px(10, minimum=2), c_rect.width // 16))
            text_w = max(10, c_rect.width - 2 * pad)
            t_surf = fonts.render_text(
                panel.label, title_px, theme.text_primary, bold=True, max_width=text_w
            )
            surface.blit(t_surf, (c_rect.x + pad, c_rect.y + pad // 2))

            s_surf = fonts.render_text(
                panel.sublabel, sub_px, theme.text_secondary, max_width=text_w
            )
            surface.blit(s_surf, (c_rect.x + pad, c_rect.y + pad // 2 + t_surf.get_height()))

            # Column-sweep reveal mask during animation
            reveal_mask: np.ndarray | None = None
            if prog < 0.999 and panel.grid.disp_cols > 1:
                vis_cols = max(1, math.ceil(panel.grid.disp_cols * prog))
                mask = np.zeros((panel.grid.disp_rows, panel.grid.disp_cols), dtype=bool)
                mask[:, :vis_cols] = True
                reveal_mask = mask

            panel.grid.draw(
                surface,
                reveal_mask=reveal_mask,
                highlight_color=theme.highlight,
                mask_color=theme.masked_cell,
            )
