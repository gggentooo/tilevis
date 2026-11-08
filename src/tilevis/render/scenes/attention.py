"""Attention pattern scene with causal-masked heatmap, spotlight arcs, and head cycling.

Single responsibility: render causal T x T attention weights, spotlight arcs, and head cycling.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from tilevis.core.kinds import StepKind, TensorRole
from tilevis.core.trace import Step, TensorSnapshot
from tilevis.render.animation import ease_out_cubic
from tilevis.render.commands import Command
from tilevis.render.layout import UiScale
from tilevis.render.scenes.base import SceneContext, Tooltip
from tilevis.render.scenes.common import (
    BaseScene,
    HitRegion,
    blend_rgb,
    chip_accent,
    draw_bezier_arc,
    draw_panel,
    draw_pill,
    find_snapshot,
    format_token_chip,
    make_grid,
    slice_2d,
)
from tilevis.render.scenes.registry import SCENES
from tilevis.render.tiles import TileGrid


@dataclass(slots=True)
class _ArcTokenNode:
    pill_rect: pygame.Rect
    label: str
    weight: float
    token_idx: int


@SCENES.register(StepKind.ATTENTION_PATTERN.value)
class AttentionPatternScene(BaseScene):
    """Draws causal-masked attention heatmap + spotlight arcs from newest token back in time."""

    DEFAULT_DURATION_SEC = 1.2

    def __init__(self) -> None:
        super().__init__()
        self._attn_snap: TensorSnapshot | None = None
        self._head_index: int = 0
        self._head_count: int = 1
        self._average_heads: bool = False
        self._heatmap_card: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._arc_card: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._grid: TileGrid | None = None
        self._masked_surface: pygame.Surface | None = None
        self._arc_nodes: list[_ArcTokenNode] = []
        self._row_tag_positions: list[tuple[int, int, str]] = []
        self._col_tag_positions: list[tuple[int, int, str]] = []
        self._last_weights: np.ndarray = np.zeros((0,), dtype=np.float32)

    @property
    def head_index(self) -> int:
        """Return currently selected 0-based attention head index."""
        return self._head_index

    @property
    def head_count(self) -> int:
        """Return total number of attention heads available in this step."""
        return self._head_count

    @property
    def average_heads(self) -> bool:
        """Return True when displaying the average over all attention heads."""
        return self._average_heads

    def _on_enter(self, step: Step, ctx: SceneContext, *, animate: bool) -> None:
        _ = animate
        snap = find_snapshot(step, TensorRole.ATTN_WEIGHTS)
        if snap is None and step.tensors:
            snap = step.tensors[0]
        self._attn_snap = snap

        if snap is not None and snap.data.ndim == 3 and snap.data.shape[0] > 0:
            self._head_count = int(snap.data.shape[0])
        else:
            raw_h = step.meta.get("n_heads")
            self._head_count = max(1, int(raw_h)) if isinstance(raw_h, int) and raw_h > 0 else 1

        if ctx.active_head is not None and self._head_count > 0:
            self._head_index = ctx.active_head % self._head_count
        else:
            self._head_index = 0
        self._average_heads = False

    def handle_command(self, cmd: Command) -> bool:
        """Handle ']' CYCLE_HEAD, '[' CYCLE_HEAD_PREV, and 'A' AVG_HEADS."""
        if self._attn_snap is None or self._head_count <= 0:
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

    def _head_badge_text(self) -> str:
        if self._average_heads:
            return f"Average of all {self._head_count} head(s)  (A to toggle, [ ] for single head)"
        return f"Spotlight Head {self._head_index + 1} of {self._head_count}  ([ ] cycle, A avg)"

    def _rebuild_layout(self, stage: pygame.Rect, scale: UiScale) -> None:
        self._cached_stage_size = stage.size
        self._hit_regions = []
        self._grid = None
        self._masked_surface = None
        self._arc_nodes = []
        self._row_tag_positions = []
        self._col_tag_positions = []
        self.downsample_notes = []

        step = self._step
        ctx = self._ctx
        snap = self._attn_snap
        if step is None or ctx is None or snap is None:
            return

        outer_pad = max(6, min(scale.px(22, minimum=6), stage.width // 24, stage.height // 24))
        gap = max(8, scale.px(20, minimum=6))
        avail_w = max(40, stage.width - 2 * outer_pad)
        avail_h = max(40, stage.height - 2 * outer_pad)

        left_w = max(24, int((avail_w - gap) * 0.54))
        right_w = max(24, avail_w - gap - left_w)
        self._heatmap_card = pygame.Rect(stage.x + outer_pad, stage.y + outer_pad, left_w, avail_h)
        self._arc_card = pygame.Rect(
            self._heatmap_card.right + gap, stage.y + outer_pad, right_w, avail_h
        )

        # Slice selected head (or average) into 2-D matrix
        arr2d, axes2 = slice_2d(snap, self._head_index, average_heads=self._average_heads)
        self._last_weights = (
            np.asarray(arr2d[-1], dtype=np.float32)
            if arr2d.shape[0] > 0
            else np.zeros((0,), dtype=np.float32)
        )

        hdr_h = min(max(36, scale.px(66, minimum=34)), max(34, self._heatmap_card.height // 4))
        pad = max(6, min(scale.px(14, minimum=4), left_w // 16))
        row_lbl_w = max(48, scale.px(110, minimum=48)) if left_w >= 300 else 0
        col_lbl_h = max(18, scale.px(26, minimum=16)) if avail_h >= 240 else 0
        legend_h = max(20, scale.px(28, minimum=18))

        gw = max(1, left_w - 2 * pad - row_lbl_w)
        gh = max(1, avail_h - hdr_h - 2 * pad - col_lbl_h - legend_h)
        grid = make_grid(
            snap,
            step,
            ctx,
            (gw, gh),
            cell_px=scale.px(36, minimum=8),
            min_cell_px=max(1, min(8, gw, gh)),
            signed=False,
            data_override=arr2d,
            axes_override=axes2,
        )
        gx = self._heatmap_card.x + pad + row_lbl_w + max(0, (gw - grid.rect.width) // 2)
        gy = self._heatmap_card.y + hdr_h + max(0, (gh - grid.rect.height) // 2)
        grid.set_position(gx, gy)
        self._grid = grid
        if grid.downsample_note:
            self.downsample_notes.append(grid.downsample_note)

        # Pre-bake causal mask (orig_col > orig_row) directly into cached heatmap surface
        masked_surf = grid.surface.copy()
        pitch = grid.cell_px + grid.effective_gap_px
        masked_col = ctx.theme.masked_cell
        for r in range(grid.disp_rows):
            orig_r = int(grid.row_orig_indices[r])
            for c in range(grid.disp_cols):
                if int(grid.col_orig_indices[c]) > orig_r:
                    pygame.draw.rect(
                        masked_surf,
                        masked_col,
                        pygame.Rect(c * pitch, r * pitch, grid.cell_px, grid.cell_px),
                    )
        self._masked_surface = masked_surf

        # Token labels alongside rows & columns when cells aren't cramped
        row_labels = grid.row_labels or tuple(ctx.pass_record.input_tokens)
        if row_lbl_w > 0 and grid.cell_px >= 14 and row_labels:
            for r in range(min(grid.disp_rows, len(row_labels))):
                cy = grid.cell_center(r, 0)[1]
                self._row_tag_positions.append((gx - 8, cy, format_token_chip(row_labels[r])))
        col_labels = grid.col_labels or row_labels
        if col_lbl_h > 0 and grid.cell_px >= 42 and col_labels:
            for c in range(min(grid.disp_cols, len(col_labels))):
                cx = grid.cell_center(0, c)[0]
                self._col_tag_positions.append(
                    (cx, grid.rect.bottom + 4, format_token_chip(col_labels[c]))
                )

        # Build right-side Spotlight Arc token nodes reserving percentage column on the right
        arc_hdr_h = hdr_h
        arc_pad = max(8, scale.px(16, minimum=6))
        nodes_top = self._arc_card.y + arc_hdr_h + 6
        nodes_h = max(24, self._arc_card.bottom - arc_pad - nodes_top)
        disp_t = arr2d.shape[0]
        if disp_t <= 0:
            return

        # Keep last <= 16 tokens for crisp arc readability if trace has huge T
        max_show = min(disp_t, max(1, nodes_h // max(18, scale.px(26, minimum=16))))
        start_tok = disp_t - max_show
        tok_labels = (
            axes2[0].labels if axes2 and axes2[0].labels else ctx.pass_record.input_tokens
        ) or tuple(f"t{i}" for i in range(disp_t))
        pct_col_w = scale.px(62, minimum=40) if right_w >= 260 else 0
        pill_w = max(54, min(scale.px(160, minimum=54), max(54, right_w // 2 - pct_col_w // 2)))
        pill_x = self._arc_card.right - arc_pad - pct_col_w - pill_w
        slot_h = nodes_h / max(1, max_show)
        pill_h = max(16, min(scale.px(36, minimum=16), int(slot_h * 0.78)))

        for slot_idx, tok_idx in enumerate(range(start_tok, disp_t)):
            lbl = (
                format_token_chip(tok_labels[tok_idx])
                if tok_idx < len(tok_labels)
                else f"t{tok_idx}"
            )
            cy = round(nodes_top + (slot_idx + 0.5) * slot_h)
            rect = pygame.Rect(pill_x, cy - pill_h // 2, pill_w, pill_h)
            w_val = float(self._last_weights[tok_idx]) if tok_idx < len(self._last_weights) else 0.0
            self._arc_nodes.append(
                _ArcTokenNode(pill_rect=rect, label=lbl, weight=w_val, token_idx=tok_idx)
            )
            is_newest = tok_idx == disp_t - 1
            tt_title = f'Token "{lbl}"' + (" (Newest)" if is_newest else "")
            self._hit_regions.append(
                HitRegion(
                    rect=rect,
                    tooltip_title=tt_title,
                    tooltip_lines=(
                        f"Spotlight received by newest token: {w_val:.1%}",
                        self._head_badge_text().split("  (")[0],
                    ),
                )
            )

    def hit_test(self, pos: tuple[int, int]) -> Tooltip | None:
        """Return hover Tooltip on attention cells (causal mask vs weight) or arc token pills."""
        for region in self._hit_regions:
            if region.rect.collidepoint(pos):
                return Tooltip(title=region.tooltip_title, lines=region.tooltip_lines, pos=pos)

        grid = self._grid
        if grid is None:
            return None
        hit = grid.cell_at(pos)
        if hit is None:
            return None
        q_lbl = f'"{hit.row_label}"' if hit.row_label is not None else f"token {hit.orig_row}"
        k_lbl = f'"{hit.col_label}"' if hit.col_label is not None else f"token {hit.orig_col}"
        if hit.orig_col > hit.orig_row:
            return Tooltip(
                title="Future Token (Masked)",
                lines=(f"{q_lbl} cannot peek ahead at {k_lbl}", "Upper triangle stays hidden."),
                pos=pos,
            )
        head_tag = (
            f"Avg ({self._head_count} heads)"
            if self._average_heads
            else f"Head {self._head_index + 1}/{self._head_count}"
        )
        return Tooltip(
            title=f"Attention Weight ({head_tag})",
            lines=(f"{q_lbl} -> {k_lbl}", f"attention: {hit.value:.1%} ({hit.value:.4f})"),
            pos=pos,
        )

    def draw(self, surface: pygame.Surface, stage: pygame.Rect) -> None:
        """Render causal attention heatmap and spotlight arcs cleanly."""
        if stage.width <= 0 or stage.height <= 0 or self._step is None or self._ctx is None:
            return
        sw, sh = surface.get_size()
        scale = UiScale.from_viewport(sw, sh)
        if self._cached_stage_size != stage.size:
            self._rebuild_layout(stage, scale)

        theme = self._ctx.theme
        fonts = self._ctx.fonts
        prog = self._timeline.progress

        # 1. Left Card: Causal Heatmap
        draw_panel(surface, self._heatmap_card, theme, scale)
        draw_panel(surface, self._arc_card, theme, scale)

        pad = max(8, scale.px(16, minimum=6))
        t_px = max(11, min(scale.font_px(22, minimum=11), self._heatmap_card.height // 18))
        sub_px = max(9, min(scale.font_px(16, minimum=9), t_px - 2))

        h_title = fonts.render_text(
            "Who Pays Attention to Whom?",
            t_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=max(20, self._heatmap_card.width - 2 * pad),
        )
        surface.blit(h_title, (self._heatmap_card.x + pad, self._heatmap_card.y + pad // 2))
        h_sub = fonts.render_text(
            self._head_badge_text(),
            sub_px,
            theme.accent,
            family=theme.font_family,
            max_width=max(20, self._heatmap_card.width - 2 * pad),
        )
        surface.blit(
            h_sub,
            (
                self._heatmap_card.x + pad,
                self._heatmap_card.y + pad // 2 + h_title.get_height() + 2,
            ),
        )

        grid = self._grid
        if grid is not None and self._masked_surface is not None:
            surface.blit(self._masked_surface, grid.rect.topleft)

            # Row-by-row animation sweep (O(1) single rect)
            if prog < 0.999:
                row_prog = ease_out_cubic(min(1.0, prog / 0.65))
                vis_rows = max(1, math.ceil(grid.disp_rows * row_prog))
                if vis_rows < grid.disp_rows:
                    pitch = grid.cell_px + grid.effective_gap_px
                    hy = grid.rect.y + vis_rows * pitch
                    hh = grid.rect.bottom - hy
                    if hh > 0:
                        pygame.draw.rect(
                            surface,
                            theme.background,
                            pygame.Rect(grid.rect.x, hy, grid.rect.width, hh),
                        )

            lbl_px = max(9, min(scale.font_px(15, minimum=9), grid.cell_px))
            max_lbl_w = max(24, grid.rect.x - self._heatmap_card.x - 10)
            for rx, ry, txt in self._row_tag_positions:
                rs = fonts.render_text(
                    txt, lbl_px, theme.text_secondary, family=theme.font_family, max_width=max_lbl_w
                )
                surface.blit(rs, (rx - rs.get_width(), ry - rs.get_height() // 2))
            for cx, cy, txt in self._col_tag_positions:
                cs = fonts.render_text(
                    txt,
                    lbl_px,
                    theme.text_secondary,
                    family=theme.font_family,
                    max_width=max(18, grid.cell_px - 4),
                )
                surface.blit(cs, (cx - cs.get_width() // 2, cy))

            # Causal mask legend swatch at bottom of heatmap card
            sw_size = max(10, sub_px - 2)
            sw_y = self._heatmap_card.bottom - pad // 2 - sw_size - 2
            sw_rect = pygame.Rect(self._heatmap_card.x + pad, sw_y, sw_size, sw_size)
            pygame.draw.rect(surface, theme.masked_cell, sw_rect, border_radius=2)
            pygame.draw.rect(surface, theme.border, sw_rect, width=1, border_radius=2)
            leg_txt = fonts.render_text(
                "Grey = hidden future tokens",
                sub_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=max(20, self._heatmap_card.width - 3 * pad - sw_size),
            )
            surface.blit(leg_txt, (sw_rect.right + 6, sw_rect.centery - leg_txt.get_height() // 2))

        # 2. Right Card: Spotlight Arcs from Newest Token
        a_title = fonts.render_text(
            "Newest Token Spotlight Arcs",
            t_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
            max_width=max(20, self._arc_card.width - 2 * pad),
        )
        surface.blit(a_title, (self._arc_card.x + pad, self._arc_card.y + pad // 2))
        a_sub = fonts.render_text(
            "Thicker arc = stronger spotlight on earlier word",
            sub_px,
            theme.text_secondary,
            family=theme.font_family,
            max_width=max(20, self._arc_card.width - 2 * pad),
        )
        surface.blit(
            a_sub,
            (self._arc_card.x + pad, self._arc_card.y + pad // 2 + a_title.get_height() + 2),
        )

        if not self._arc_nodes:
            return

        arc_t = ease_out_cubic(max(0.0, min(1.0, (prog - 0.35) / 0.65)))
        newest = self._arc_nodes[-1]
        pill_px = max(10, min(scale.font_px(18, minimum=10), max(10, newest.pill_rect.height - 6)))

        # Draw arcs first behind pills
        if arc_t > 0.04:
            if len(self._arc_nodes) == 1:
                # Edge case T=1: draw clean self-attention loop arc
                r = newest.pill_rect
                loop_w = max(24, min(scale.px(90, minimum=24), r.x - self._arc_card.x - pad))
                p0 = (float(r.x - 2), float(r.y + r.height * 0.28))
                p2 = (float(r.x - 2), float(r.y + r.height * 0.72))
                p1 = (float(r.x - loop_w), float(r.centery))
                draw_bezier_arc(surface, p0, p1, p2, theme.highlight, thickness=max(2, scale.px(4)))
                self_lbl = fonts.render_text(
                    "100% self",
                    max(9, pill_px - 2),
                    theme.highlight,
                    bold=True,
                    family=theme.font_family,
                )
                surface.blit(
                    self_lbl,
                    (
                        max(self._arc_card.x + pad, r.x - loop_w - self_lbl.get_width() // 2),
                        r.centery - self_lbl.get_height() // 2,
                    ),
                )
            else:
                src_x = float(newest.pill_rect.x - 3)
                src_y = float(newest.pill_rect.centery)
                avail_bow = max(24.0, float(newest.pill_rect.x - self._arc_card.x - pad - 12))
                n_nodes = len(self._arc_nodes)
                max_w = max(1e-6, max(node.weight for node in self._arc_nodes))

                for idx, node in enumerate(self._arc_nodes):
                    if node.weight <= 0.005:
                        continue
                    rel = node.weight / max_w
                    thick = max(1, min(9, round(1 + 7 * rel * arc_t)))
                    color = blend_rgb(theme.panel_alt, theme.highlight, 0.25 + 0.75 * rel)
                    dst_x = float(node.pill_rect.x - 3)
                    dst_y = float(node.pill_rect.centery)
                    if idx == n_nodes - 1:
                        # Compact self-attention curl on newest token
                        bow = min(avail_bow * 0.45, max(16.0, float(node.pill_rect.height) * 0.9))
                        p0 = (src_x, src_y - node.pill_rect.height * 0.2)
                        p2 = (src_x, src_y + node.pill_rect.height * 0.2)
                        p1 = (src_x - bow, src_y)
                    else:
                        dist_frac = (n_nodes - 1 - idx) / max(1, n_nodes - 1)
                        bow = avail_bow * (0.28 + 0.70 * dist_frac) * arc_t
                        p0 = (src_x, src_y)
                        p2 = (dst_x, dst_y)
                        p1 = (src_x - bow, (src_y + dst_y) * 0.5)
                    draw_bezier_arc(surface, p0, p1, p2, color, thickness=thick)

        # Draw token node pills + percentage readout
        for idx, node in enumerate(self._arc_nodes):
            is_newest = idx == len(self._arc_nodes) - 1
            bg = (
                theme.highlight
                if is_newest
                else blend_rgb(
                    theme.panel_alt,
                    chip_accent(node.token_idx, theme),
                    0.35 + 0.5 * min(1.0, node.weight * 2.0),
                )
            )
            fg = theme.background if is_newest else theme.text_primary
            draw_pill(
                surface,
                node.pill_rect,
                node.label,
                fonts,
                pill_px,
                fg,
                bg,
                theme.accent if is_newest else theme.border,
                bold=is_newest,
                family=theme.font_family,
            )
            if node.weight >= 0.01 and (self._arc_card.right - node.pill_rect.right) >= 36:
                pct_s = fonts.render_text(
                    f"{node.weight:.0%}",
                    max(9, pill_px - 3),
                    theme.highlight if is_newest else theme.text_secondary,
                    bold=node.weight >= 0.25,
                    family=theme.font_family,
                )
                surface.blit(
                    pct_s,
                    (
                        node.pill_rect.right + 8,
                        node.pill_rect.centery - pct_s.get_height() // 2,
                    ),
                )
