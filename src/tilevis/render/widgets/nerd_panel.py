"""Nerd Mode telemetry panel displaying technical parameters on demand.

Single responsibility: display full tensor shapes, module paths, reduction stats, and fidelity.
Allowed dependencies: tilevis.core, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import pygame

from tilevis.core.trace import Step
from tilevis.render.layout import UiScale
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme


class NerdPanelWidget:
    """Floating telemetry card showing internal dimensions, statistics, and fidelity."""

    def __init__(self) -> None:
        self.rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.close_btn_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

    def draw(
        self,
        surface: pygame.Surface,
        stage_rect: pygame.Rect,
        step: Step,
        *,
        theme: Theme,
        fonts: FontManager,
        scale: UiScale,
        normalizer_name: str = "shared_step",
        normalizer_scale: float = 1.0,
    ) -> None:
        """Render technical nerd mode panel floating inside stage_rect."""
        if stage_rect.width <= 0 or stage_rect.height <= 0:
            return

        pad_x = max(8, scale.px(12, minimum=8))
        pad_y = max(8, scale.px(10, minimum=8))
        w = max(240, min(scale.px(420, minimum=240), stage_rect.width - 2 * pad_x))
        h = max(180, min(scale.px(460, minimum=180), stage_rect.height - 2 * pad_y))

        # Anchor to the top-right of stage_rect
        x = stage_rect.right - w - pad_x
        y = stage_rect.y + pad_y
        self.rect = pygame.Rect(x, y, w, h)

        # Semi-transparent backing panel
        panel_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        panel_alpha = 240
        panel_color = (*theme.panel[:3], panel_alpha)
        panel_surf.fill(panel_color)
        surface.blit(panel_surf, (x, y))

        radius = max(2, scale.px(6, minimum=2))
        pygame.draw.rect(surface, theme.accent, self.rect, width=1, border_radius=radius)

        # Text dimensions
        title_font_px = max(10, scale.font_px(14, minimum=10))
        text_font_px = max(9, scale.font_px(11, minimum=9))
        code_font_px = max(9, scale.font_px(10, minimum=9))

        inner_x = self.rect.x + pad_x
        inner_w = self.rect.width - 2 * pad_x
        cur_y = self.rect.y + pad_y

        # Title & close badge
        title_surf = fonts.render_text(
            f"Nerd Mode: {step.id}",
            title_font_px,
            theme.accent,
            bold=True,
            family=theme.font_family,
            max_width=inner_w - 30,
        )
        surface.blit(title_surf, (inner_x, cur_y))

        close_surf = fonts.render_text(
            "[X]",
            text_font_px,
            theme.text_secondary,
            family=theme.font_family,
        )
        self.close_btn_rect = pygame.Rect(
            self.rect.right - pad_x - close_surf.get_width() - 4,
            cur_y,
            close_surf.get_width() + 8,
            close_surf.get_height() + 4,
        )
        surface.blit(close_surf, (self.close_btn_rect.x + 2, self.close_btn_rect.y + 2))

        cur_y += title_surf.get_height() + max(3, scale.px(4, minimum=3))
        pygame.draw.line(
            surface, theme.border, (inner_x, cur_y), (self.rect.right - pad_x, cur_y), width=1
        )
        cur_y += max(4, scale.px(6, minimum=4))

        # Module path (truncated gracefully if very long)
        mod_path = step.module_path or "N/A"
        mod_surf = fonts.render_text(
            f"Module: {mod_path}",
            text_font_px,
            theme.text_primary,
            family=theme.font_family,
            max_width=inner_w,
        )
        surface.blit(mod_surf, (inner_x, cur_y))
        cur_y += mod_surf.get_height() + 2

        # Semantic & Stage
        meta_surf = fonts.render_text(
            f"Semantic: {step.semantic}  ·  Kind: {step.kind}",
            text_font_px,
            theme.text_secondary,
            family=theme.font_family,
            max_width=inner_w,
        )
        surface.blit(meta_surf, (inner_x, cur_y))
        cur_y += meta_surf.get_height() + 2

        # Fidelity badge
        fid = step.fidelity
        if fid.status == "exact":
            fid_color = theme.success
        elif fid.status in ("approximate", "unverified"):
            fid_color = theme.warning
        else:
            fid_color = theme.error

        err_str = f" (err: {fid.max_abs_err:.1e})" if fid.max_abs_err is not None else ""
        fid_text = f"Fidelity: {fid.status.upper()}{err_str}"
        if fid.note:
            fid_text += f" - {fid.note}"
        fid_surf = fonts.render_text(
            fid_text,
            text_font_px,
            fid_color,
            bold=True,
            family=theme.font_family,
            max_width=inner_w,
        )
        surface.blit(fid_surf, (inner_x, cur_y))
        cur_y += fid_surf.get_height() + 2

        # Normalizer scale
        norm_surf = fonts.render_text(
            f"Normalizer: {normalizer_name} (scale: {normalizer_scale:.2f})",
            text_font_px,
            theme.text_secondary,
            family=theme.font_family,
            max_width=inner_w,
        )
        surface.blit(norm_surf, (inner_x, cur_y))
        cur_y += norm_surf.get_height() + max(3, scale.px(5, minimum=3))

        pygame.draw.line(
            surface, theme.border, (inner_x, cur_y), (self.rect.right - pad_x, cur_y), width=1
        )
        cur_y += max(4, scale.px(6, minimum=4))

        # Tensor snapshots list
        t_hdr = fonts.render_text(
            f"Tensors ({len(step.tensors)}):",
            text_font_px,
            theme.text_primary,
            bold=True,
            family=theme.font_family,
        )
        surface.blit(t_hdr, (inner_x, cur_y))
        cur_y += t_hdr.get_height() + 3

        gap_t = max(2, scale.px(4, minimum=2))
        for t in step.tensors:
            if cur_y + 35 > self.rect.bottom - pad_y:
                # Render overflow indicator
                more_surf = fonts.render_text(
                    f"... and {len(step.tensors) - step.tensors.index(t)} more",
                    code_font_px,
                    theme.text_secondary,
                    family=theme.font_family,
                )
                surface.blit(more_surf, (inner_x, cur_y))
                break

            # Line 1: role & name & shape
            shape_str = f"{t.full_shape} -> {t.data.shape} ({t.reduction})"
            l1 = fonts.render_text(
                f"• [{t.role}] {t.name}: {shape_str}",
                code_font_px,
                theme.text_primary,
                family=theme.font_family,
                max_width=inner_w,
            )
            surface.blit(l1, (inner_x, cur_y))
            cur_y += l1.get_height() + 1

            # Line 2: stats
            st = t.stats
            st_str = f"  min={st.min:+.2f} max={st.max:+.2f} std={st.std:.2f} p99={st.abs_p99:.2f}"
            l2 = fonts.render_text(
                st_str,
                code_font_px,
                theme.text_secondary,
                family=theme.font_family,
                max_width=inner_w,
            )
            surface.blit(l2, (inner_x, cur_y))
            cur_y += l2.get_height() + gap_t

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Handle click inside nerd panel; return True if consumed (e.g. clicking close)."""
        return bool(self.rect.collidepoint(pos))
