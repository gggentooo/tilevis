"""App class owning window lifecycle, main loop, and clock.

Single responsibility: manage frame updates, event dispatch, and idle throttling.
Allowed dependencies: tilevis.core, tilevis.narration, tilevis.render, pygame, stdlib.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pygame

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel
from tilevis.core.trace import PassRecord, Step, Trace
from tilevis.narration.captions import Caption, CaptionProvider
from tilevis.render.colormap import COLORMAPS
from tilevis.render.commands import Command
from tilevis.render.input_map import InputMap
from tilevis.render.layout import UiScale
from tilevis.render.normalization import get_normalizer
from tilevis.render.platform import detect_display_info, enable_high_dpi
from tilevis.render.player import StepPlayer
from tilevis.render.playlist import resolve_step
from tilevis.render.scenes.base import Scene, SceneContext, Tooltip
from tilevis.render.scenes.registry import create_scene
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme
from tilevis.render.themes.builtin import THEMES, get_theme
from tilevis.render.widgets.caption_box import CaptionBox
from tilevis.render.widgets.help_overlay import HelpOverlayWidget
from tilevis.render.widgets.hud import HudWidget
from tilevis.render.widgets.nerd_panel import NerdPanelWidget
from tilevis.render.widgets.tooltip import TooltipWidget
from tilevis.render.window import WindowManager

logger = logging.getLogger("tilevis.render.app")

DOUBLE_ESC_WINDOW_SEC = 2.0


class App:
    """Main interactive trace player coordinating navigation, scenes, widgets, and display."""

    def __init__(
        self,
        trace: Trace,
        settings: PresenterSettings | None = None,
        *,
        headless: bool = False,
        viewport_size: tuple[int, int] | None = None,
    ) -> None:
        self.trace = trace
        self.settings = settings or PresenterSettings()
        self.headless = headless

        enable_high_dpi()

        if not pygame.font.get_init():
            pygame.font.init()

        display_idx = self.settings.window_display
        disp_info = detect_display_info(display_idx)
        win_size = viewport_size or disp_info.default_windowed

        self.window = WindowManager(
            mode=self.settings.window_mode,
            display=display_idx,
            windowed_size=win_size,
        )
        if headless:
            size = viewport_size or (1920, 1080)
            self.window.windowed_size = size
            self.window.size = (max(1, size[0]), max(1, size[1]))
            self.window.logical_size = self.window.size
            self.window.drawable_size = self.window.size
            self.window.dpi_scale = (1.0, 1.0)
            self.surface: pygame.Surface = pygame.Surface(self.window.size)
        else:
            self.surface = self.window.create(headless=False)
            if viewport_size is not None:
                self.surface = self.window.handle_resize(viewport_size[0], viewport_size[1])

        w, h = self.surface.get_size()
        self.scale = UiScale.from_viewport(w, h)
        self.theme: Theme = get_theme(self.settings.theme)
        self.fonts = FontManager(self.theme.font_family)
        self.captions = CaptionProvider(self.settings.locale)
        self.input_map = InputMap(self.settings.keys)
        self.player = StepPlayer(self.trace, self.settings)

        self.caption_box = CaptionBox()
        self.hud = HudWidget()
        self.tooltip_widget = TooltipWidget()
        self.nerd_panel = NerdPanelWidget()
        self.help_overlay = HelpOverlayWidget()

        self.should_quit: bool = False
        self.exit_code: int = 0
        self._esc_timer: float | None = None
        self._overlay_open: bool = False
        self._help_open: bool = False
        self._active_tooltip: Tooltip | None = None
        self._autoplay: bool = False
        self._autoplay_timer: float = 0.0
        self._autoplay_interval: float = 1.6
        self._needs_redraw: bool = True

        self._current_step: Step | None = None
        self._current_pass: PassRecord | None = None
        self._current_caption: Caption = Caption("Ready", "", "")
        self.scene: Scene = create_scene("GENERIC")
        self._enter_current_step(animate=True)

    def _build_scene_context(self, pass_record: PassRecord) -> SceneContext:
        """Assemble immutable SceneContext for the active step."""
        return SceneContext(
            theme=self.theme,
            colormaps=COLORMAPS,
            normalizer=get_normalizer(self.settings.normalizer),
            fonts=self.fonts,
            scale=self.scale,
            trace=self.trace,
            pass_record=pass_record,
            detail_level=self.settings.detail_level,
            nerd_mode=self.settings.nerd_mode,
            animation_speed=self.settings.animation_speed,
        )

    def _enter_current_step(self, *, animate: bool) -> None:
        """Instantiate and enter Scene for the current StepPlayer cursor."""
        ref = self.player.current
        if ref is None:
            return
        step, pass_rec = resolve_step(self.trace, ref)
        self._current_step = step
        self._current_pass = pass_rec
        self._current_caption = self.captions.caption(step, self.trace, pass_rec)

        scene = create_scene(step.kind)
        scene.enter(step, self._build_scene_context(pass_rec), animate=animate)
        self.scene = scene
        self._needs_redraw = True

    def resize(self, width: int, height: int) -> None:
        """Resize viewport cleanly while preserving playback cursor and state."""
        w = max(1, int(width))
        h = max(1, int(height))
        if self.headless:
            self.window.size = (w, h)
            self.window.logical_size = (w, h)
            self.window.drawable_size = (w, h)
            self.window.dpi_scale = (1.0, 1.0)
            self.surface = pygame.Surface((w, h))
        else:
            self.surface = self.window.handle_resize(w, h, headless=False)
        sw, sh = self.surface.get_size()
        self.scale = UiScale.from_viewport(sw, sh)
        if self._current_step is not None and self._current_pass is not None:
            tl = getattr(self.scene, "_timeline", None)
            was_animating = self.scene.is_animating
            progress_val = tl.progress if tl is not None else 1.0
            self._enter_current_step(animate=was_animating)
            if was_animating and hasattr(self.scene, "_timeline"):
                self.scene._timeline.seek(progress_val)
        self._needs_redraw = True

    def _cycle_detail_level(self) -> None:
        """Rotate STORY (1) -> STANDARD (2) -> DEEP (3) and re-anchor StepPlayer."""
        order = (int(DetailLevel.STORY), int(DetailLevel.STANDARD), int(DetailLevel.DEEP))
        cur = self.settings.detail_level
        next_level = order[(order.index(cur) + 1) % len(order)] if cur in order else order[1]
        self.settings.detail_level = next_level
        self.player.update_settings(self.settings)
        self._enter_current_step(animate=False)

    def _cycle_theme(self) -> None:
        """Rotate between registered presentation themes and invalidate cached surfaces."""
        names = THEMES.keys()
        if not names:
            return
        cur = self.theme.name
        idx = (names.index(cur) + 1) % len(names) if cur in names else 0
        next_theme = THEMES.get(names[idx])
        self.theme = next_theme
        self.settings.theme = next_theme.name
        self.fonts = FontManager(self.theme.font_family)
        self._enter_current_step(animate=False)

    def _toggle_locale(self) -> None:
        """Toggle caption locale (en <-> ko) and re-render current step."""
        next_locale = "ko" if self.settings.locale == "en" else "en"
        self.settings.locale = next_locale
        self.captions = CaptionProvider(next_locale)
        self.fonts.clear_cache()
        self._enter_current_step(animate=False)

    def dispatch_command(self, cmd: Command) -> None:
        """Execute a high-level presenter Command."""
        if cmd == Command.QUIT:
            self.should_quit = True
            return

        if cmd == Command.TOGGLE_HELP:
            self._help_open = not self._help_open
            self._needs_redraw = True
            return

        if cmd == Command.ESCAPE:
            if self._help_open:
                self._help_open = False
                self._needs_redraw = True
                return
            if self._esc_timer is not None and self._esc_timer > 0.0:
                self.should_quit = True
                return
            self._esc_timer = DOUBLE_ESC_WINDOW_SEC
            self._needs_redraw = True
            return

        # Any non-Escape command clears an armed Esc confirmation
        if self._esc_timer is not None:
            self._esc_timer = None
            self._needs_redraw = True

        if cmd == Command.NEXT:
            if self.scene.is_animating and self.settings.advance_skips_animation:
                self.scene.finish()
                self._needs_redraw = True
            elif self.player.next():
                self._enter_current_step(animate=True)
            return

        if cmd == Command.PREV:
            if self.player.prev():
                self._enter_current_step(animate=False)
            return

        if cmd == Command.NEXT_STAGE:
            if self.player.next_stage():
                self._enter_current_step(animate=True)
            return

        if cmd == Command.PREV_STAGE:
            if self.player.prev_stage():
                self._enter_current_step(animate=False)
            return

        if cmd == Command.FIRST:
            if self.player.first():
                self._enter_current_step(animate=True)
            return

        if cmd == Command.LAST:
            if self.player.last():
                self._enter_current_step(animate=True)
            return

        if cmd == Command.NEXT_PASS:
            if self.player.next_pass():
                self._enter_current_step(animate=True)
            return

        if cmd == Command.CYCLE_DETAIL:
            self._cycle_detail_level()
            return

        if cmd == Command.TOGGLE_NERD:
            self.settings.nerd_mode = not self.settings.nerd_mode
            self._needs_redraw = True
            return

        if cmd == Command.CYCLE_THEME:
            self._cycle_theme()
            return

        if cmd == Command.TOGGLE_LOCALE:
            self._toggle_locale()
            return

        if cmd == Command.SPEED_UP:
            self.settings.animation_speed = min(4.0, round(self.settings.animation_speed * 1.25, 2))
            self._needs_redraw = True
            return

        if cmd == Command.SPEED_DOWN:
            self.settings.animation_speed = max(0.25, round(self.settings.animation_speed * 0.8, 2))
            self._needs_redraw = True
            return

        if cmd == Command.TOGGLE_AUTOPLAY:
            self._autoplay = not self._autoplay
            self._autoplay_timer = 0.0
            self._needs_redraw = True
            return

        if cmd == Command.TOGGLE_FULLSCREEN:
            if not self.headless:
                self.surface = self.window.toggle_fullscreen(headless=False)
                sw, sh = self.surface.get_size()
                self.scale = UiScale.from_viewport(sw, sh)
                if self._current_step is not None and self._current_pass is not None:
                    tl = getattr(self.scene, "_timeline", None)
                    was_animating = self.scene.is_animating
                    progress_val = tl.progress if tl is not None else 1.0
                    self._enter_current_step(animate=was_animating)
                    if was_animating and hasattr(self.scene, "_timeline"):
                        self.scene._timeline.seek(progress_val)
            self._needs_redraw = True
            return

        if self.scene.handle_command(cmd):
            self._needs_redraw = True

    def tick(self, dt: float, events: Sequence[pygame.event.Event]) -> None:
        """Pure time step: process events, dispatch commands, and update animations."""
        dt_clean = max(0.0, float(dt))

        # Decay double-Esc confirmation timer prior to event processing
        if self._esc_timer is not None:
            self._esc_timer -= dt_clean
            if self._esc_timer <= 0.0:
                self._esc_timer = None
                self._needs_redraw = True

        for ev in events:
            if ev.type == pygame.QUIT:
                self.should_quit = True
                return

            if ev.type == pygame.VIDEORESIZE:
                w = int(getattr(ev, "w", self.window.size[0]))
                h = int(getattr(ev, "h", self.window.size[1]))
                self.resize(w, h)
                continue

            if hasattr(pygame, "WINDOWSIZECHANGED") and ev.type == pygame.WINDOWSIZECHANGED:
                win_w = getattr(ev, "x", None) or getattr(ev, "w", None)
                win_h = getattr(ev, "y", None) or getattr(ev, "h", None)
                if win_w is None or win_h is None:
                    win_w, win_h = self.window.logical_size
                if win_w is not None and win_h is not None:
                    self.resize(int(win_w), int(win_h))
                continue

            if hasattr(pygame, "WINDOWFOCUSLOST") and ev.type == pygame.WINDOWFOCUSLOST:
                self.window.focused = False
                continue

            if hasattr(pygame, "WINDOWFOCUSGAINED") and ev.type == pygame.WINDOWFOCUSGAINED:
                self.window.focused = True
                if pygame.display.get_init():
                    pygame.key.set_mods(0)
                self._needs_redraw = True
                continue

            if ev.type == pygame.MOUSEMOTION:
                raw_pos = getattr(ev, "pos", (0, 0))
                pos = self.window.map_point(raw_pos)
                if self._help_open:
                    self._active_tooltip = None
                else:
                    self.hud.buttons.update_hover(pos)
                    self.hud.minimap.update_hover(pos)
                    self._active_tooltip = self.scene.hit_test(pos)
                self._needs_redraw = True
                continue

            if ev.type == pygame.MOUSEBUTTONDOWN:
                raw_pos = getattr(ev, "pos", (0, 0))
                pos = self.window.map_point(raw_pos)
                # 1. Help overlay open -> click closes overlay
                if self._help_open:
                    self._help_open = False
                    self._needs_redraw = True
                    continue

                # 2. Nerd panel close button
                if self.settings.nerd_mode and self.nerd_panel.close_btn_rect.collidepoint(pos):
                    self.settings.nerd_mode = False
                    self._needs_redraw = True
                    continue

                # 3. Nerd panel body click consumed
                if self.settings.nerd_mode and self.nerd_panel.rect.collidepoint(pos):
                    continue

                # 4. Interactive buttons click
                btn_cmd = self.hud.buttons.handle_click(pos)
                if btn_cmd is not None:
                    self.dispatch_command(btn_cmd)
                    continue

                # 5. Minimap segment click
                if self.hud.minimap.handle_click(pos, self.player):
                    self._enter_current_step(animate=True)
                    continue

                # 6. Default stage click
                cmd = self.input_map.translate_event(ev)
                if cmd is not None:
                    self.dispatch_command(cmd)
                    if self.should_quit:
                        return
                continue

            if ev.type == pygame.KEYDOWN:
                key = int(getattr(ev, "key", 0))
                if self._help_open:
                    if key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_F1):
                        self._help_open = False
                        self._needs_redraw = True
                    continue

            cmd = self.input_map.translate_event(ev)
            if cmd is not None:
                self.dispatch_command(cmd)
                if self.should_quit:
                    return

        if self.scene.is_animating:
            self.scene.update(dt_clean)
            self._needs_redraw = True
        elif self._autoplay:
            self._autoplay_timer += dt_clean
            if self._autoplay_timer >= self._autoplay_interval:
                self._autoplay_timer = 0.0
                if self.player.next():
                    self._enter_current_step(animate=True)
                else:
                    self._autoplay = False
                    self._needs_redraw = True

    def draw(self, surface: pygame.Surface | None = None) -> None:
        """Render background, stage scene, HUD header, caption box, and footer."""
        target = surface if surface is not None else self.surface
        sw, sh = target.get_size()
        if (sw, sh) != (self.scale.viewport_w, self.scale.viewport_h):
            self.scale = UiScale.from_viewport(sw, sh)

        target.fill(self.theme.background)
        self.hud.draw_header(
            target,
            self.scale.header,
            self.player,
            theme=self.theme,
            fonts=self.fonts,
            scale=self.scale,
            nerd_mode=self.settings.nerd_mode,
            esc_armed=self._esc_timer is not None,
        )
        self.scene.draw(target, self.scale.stage)
        self.caption_box.draw(
            target,
            self.scale.caption,
            self._current_caption,
            theme=self.theme,
            fonts=self.fonts,
            scale=self.scale,
            nerd_mode=self.settings.nerd_mode,
        )

        step = self._current_step
        cmap_name = (
            self.theme.sequential_cmap
            if (step and step.kind in ("ATTENTION_PATTERN", "NEXT_TOKEN"))
            else self.theme.diverging_cmap
        )
        colormap = COLORMAPS.get(cmap_name)
        scale_val = 1.0
        if step and step.tensors:
            scale_val = max(0.01, float(step.tensors[0].stats.abs_p99))

        self.hud.draw_footer(
            target,
            self.scale.footer,
            self.player,
            self.settings,
            colormap=colormap,
            scale_val=scale_val,
            theme=self.theme,
            fonts=self.fonts,
            scale=self.scale,
        )

        if self.settings.nerd_mode and self._current_step is not None:
            self.nerd_panel.draw(
                target,
                self.scale.stage,
                self._current_step,
                theme=self.theme,
                fonts=self.fonts,
                scale=self.scale,
                normalizer_name=self.settings.normalizer,
                normalizer_scale=scale_val,
            )

        if self._active_tooltip is not None and not self._help_open:
            self.tooltip_widget.draw(
                target,
                self._active_tooltip,
                sw,
                sh,
                theme=self.theme,
                fonts=self.fonts,
                scale=self.scale,
            )

        if self._help_open:
            self.help_overlay.draw(
                target,
                sw,
                sh,
                self.input_map,
                theme=self.theme,
                fonts=self.fonts,
                scale=self.scale,
            )

    def run(self) -> int:
        """Execute main presentation event loop with 60 FPS cap and idle CPU throttling."""
        fps = max(1, min(60, int(self.settings.window_fps)))
        clock = pygame.time.Clock()

        self.draw()
        if not self.headless:
            self.window.present()
        self._needs_redraw = False

        try:
            while not self.should_quit:
                if self.headless or not pygame.display.get_init():
                    events: list[pygame.event.Event] = []
                else:
                    is_idle = (
                        not self.scene.is_animating
                        and self._esc_timer is None
                        and not self._autoplay
                        and not self._needs_redraw
                    )
                    if is_idle:
                        first_ev = pygame.event.wait(250)
                        events = (
                            [first_ev, *pygame.event.get()]
                            if first_ev.type != pygame.NOEVENT
                            else []
                        )
                    else:
                        events = pygame.event.get()

                dt = min(clock.tick(fps) / 1000.0, 0.1)
                had_events = bool(events)
                self.tick(dt, events)

                if self._needs_redraw or self.scene.is_animating or had_events:
                    self.draw()
                    if not self.headless:
                        self.window.present()
                    self._needs_redraw = False

                if self.headless:
                    break
        finally:
            if not self.headless:
                self.window.destroy()

        return self.exit_code
