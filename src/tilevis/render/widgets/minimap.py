"""Pipeline minimap overview widget (Input -> B1 -> B2 ... -> Output).

Single responsibility: display clickable model pipeline progress with collapsing for deep models.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from tilevis.render.layout import UiScale
from tilevis.render.player import StepPlayer
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


@dataclass(frozen=True, slots=True)
class MinimapSegment:
    """A clickable segment in the pipeline minimap."""

    label: str
    target_type: str  # "input", "block", "output", "ellipsis"
    block_index: int | None
    rect: pygame.Rect
    is_active: bool
    is_clickable: bool


class MinimapWidget:
    """Top pipeline minimap with dynamic collapsing for deep models (12-48 layers)."""

    def __init__(self) -> None:
        self.segments: list[MinimapSegment] = []
        self._hovered_index: int | None = None

    def compute_segments(
        self,
        rect: pygame.Rect,
        player: StepPlayer,
        *,
        scale: UiScale,
        max_visible_blocks: int = 6,
    ) -> list[MinimapSegment]:
        """Compute layout of pipeline segments within rect, collapsing middle blocks if deep."""
        if rect.width <= 0 or rect.height <= 0:
            self.segments = []
            return []

        n_layers = max(1, player.trace.model.n_layers)
        cur_ref = player.current

        # Determine active stage/block
        active_type = "input"
        active_block: int | None = None
        if cur_ref is not None:
            if cur_ref.kind == "TITLE":
                active_type = "input"
            elif cur_ref.kind == "END":
                active_type = "output"
            elif cur_ref.stage == "BLOCK" and cur_ref.block_index is not None:
                active_type = "block"
                active_block = cur_ref.block_index
            elif cur_ref.stage == "OUTPUT":
                active_type = "output"
            else:
                active_type = "input"

        # Determine which block indices to show
        if n_layers <= max_visible_blocks:
            visible_blocks: list[int | None] = list(range(n_layers))
        else:
            # Deep model: always show 0, n_layers-1, and window around active_block
            focal = active_block if active_block is not None else 0
            needed: set[int] = {0, n_layers - 1}
            for offset in (-1, 0, 1):
                b = focal + offset
                if 0 <= b < n_layers:
                    needed.add(b)

            sorted_needed = sorted(needed)
            visible_blocks = []
            prev: int | None = None
            for b in sorted_needed:
                if prev is not None and b > prev + 1:
                    visible_blocks.append(None)  # Ellipsis
                visible_blocks.append(b)
                prev = b

        # Build list of items to place: "Input", blocks/ellipses, "Output"
        items: list[tuple[str, str, int | None, bool, bool]] = [
            ("Input", "input", None, active_type == "input", True)
        ]
        for b_cand in visible_blocks:
            if b_cand is None:
                items.append(("...", "ellipsis", None, False, False))
            else:
                is_act = active_type == "block" and active_block == b_cand
                items.append((f"B{b_cand + 1}", "block", b_cand, is_act, True))
        items.append(("Output", "output", None, active_type == "output", True))

        # Horizontal partitioning
        n_items = len(items)
        gap = max(2, scale.px(4, minimum=2))
        total_gaps = max(0, n_items - 1) * gap
        avail_w = max(1, rect.width - total_gaps)

        seg_w = max(12, avail_w // n_items)
        # Allocate rects
        segments: list[MinimapSegment] = []
        x = rect.x
        for i, (label, t_type, b_idx, is_act, is_click) in enumerate(items):
            # Give remainder pixels to the last item
            w = (rect.right - x) if i == n_items - 1 else seg_w
            seg_rect = pygame.Rect(x, rect.y, max(1, w), rect.height)
            segments.append(
                MinimapSegment(
                    label=label,
                    target_type=t_type,
                    block_index=b_idx,
                    rect=seg_rect,
                    is_active=is_act,
                    is_clickable=is_click,
                )
            )
            x += w + gap

        self.segments = segments
        return segments

    def update_hover(self, pos: tuple[int, int]) -> None:
        """Update hovered segment index based on mouse cursor position."""
        self._hovered_index = None
        for i, seg in enumerate(self.segments):
            if seg.is_clickable and seg.rect.collidepoint(pos):
                self._hovered_index = i
                break

    def hit_test(self, pos: tuple[int, int]) -> MinimapSegment | None:
        """Return clickable segment under pos, or None."""
        for seg in self.segments:
            if seg.is_clickable and seg.rect.collidepoint(pos):
                return seg
        return None

    def handle_click(self, pos: tuple[int, int], player: StepPlayer) -> bool:
        """Handle mouse click on minimap, jumping StepPlayer to selected block/stage."""
        seg = self.hit_test(pos)
        if seg is None:
            return False

        cur_pass = player.current.pass_index if player.current else 0
        p_idx = cur_pass if cur_pass is not None else 0

        if seg.target_type == "block" and seg.block_index is not None:
            # Jump to first playlist step of that block in current pass
            return player.goto_block(p_idx, seg.block_index)
        elif seg.target_type == "input":
            # Jump to first INPUT step in current pass
            for idx, ref in enumerate(player.playlist):
                if ref.pass_index == p_idx and ref.stage == "INPUT":
                    if player.cursor != idx:
                        player.cursor = idx
                        return True
                    return False
            # Fallback to first step of current pass
            for idx, ref in enumerate(player.playlist):
                if ref.pass_index == p_idx:
                    if player.cursor != idx:
                        player.cursor = idx
                        return True
                    return False
        elif seg.target_type == "output":
            # Jump to first OUTPUT step in current pass
            for idx, ref in enumerate(player.playlist):
                if ref.pass_index == p_idx and ref.stage == "OUTPUT":
                    if player.cursor != idx:
                        player.cursor = idx
                        return True
                    return False

        return False

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        player: StepPlayer,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
    ) -> None:
        """Render pipeline minimap into rect."""
        if rect.width <= 0 or rect.height <= 0:
            return

        segments = self.compute_segments(rect, player, scale=scale)
        font_px = max(9, min(scale.font_px(14, minimum=9), max(9, rect.height - 4)))

        radius = max(2, scale.px(4, minimum=2))
        for i, seg in enumerate(segments):
            if seg.rect.width <= 0 or seg.rect.height <= 0:
                continue

            # Determine colors
            is_hover = self._hovered_index == i
            if seg.is_active:
                bg_color = theme.accent
                txt_color = theme.background if theme.name == "dark" else (255, 255, 255)
                border_color = theme.highlight
            elif is_hover and seg.is_clickable:
                bg_color = theme.panel_alt
                txt_color = theme.text_primary
                border_color = theme.accent
            else:
                bg_color = theme.panel
                txt_color = theme.text_secondary if seg.is_clickable else theme.masked_cell
                border_color = theme.border

            pygame.draw.rect(surface, bg_color, seg.rect, border_radius=radius)
            pygame.draw.rect(surface, border_color, seg.rect, width=1, border_radius=radius)

            # Render text centered
            surf = fonts.render_text(
                seg.label,
                font_px,
                txt_color,
                bold=seg.is_active,
                family=theme.font_family,
                max_width=max(4, seg.rect.width - 2),
            )
            tx = seg.rect.x + max(0, (seg.rect.width - surf.get_width()) // 2)
            ty = seg.rect.y + max(0, (seg.rect.height - surf.get_height()) // 2)
            surface.blit(surf, (tx, ty))
