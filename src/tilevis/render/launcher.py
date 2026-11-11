"""Interactive GUI trace launcher for selecting and running presentation traces.

Single responsibility: discover .trace.npz files and present a High-DPI pygame-ce GUI
with an interactive dropdown selector, metadata preview, and presenter settings toggles.
Allowed dependencies: tilevis.core, tilevis.render, pygame, numpy, stdlib.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from tilevis.core.config import PresenterSettings
from tilevis.render.app import App
from tilevis.render.layout import UiScale
from tilevis.render.platform import enable_high_dpi
from tilevis.render.scenes.common import draw_panel, draw_pill
from tilevis.render.text import FontManager
from tilevis.render.themes.base import Theme
from tilevis.render.themes.builtin import get_theme
from tilevis.render.window import WindowManager

logger = logging.getLogger("tilevis.render.launcher")

_TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "title": "LLM Tile Visualizer",
        "subtitle": "Interactive Trace Player & Presentation Launcher",
        "select_header": "SELECT TRACE ARCHIVE",
        "no_traces": "No .trace.npz files found in traces/ directory",
        "metadata_header": "TRACE METADATA & MODEL INFO",
        "empty_notice": "Place your recorded *.trace.npz files into the traces/ directory.",
        "settings_header": "PLAYBACK SETTINGS",
        "mode": "Mode",
        "theme": "Theme",
        "detail": "Detail",
        "lang": "Lang",
        "refresh_btn": "↻ Refresh Traces",
        "launch_btn": "Launch Presentation ▶",
        "exit_btn": "Exit ✕",
        "user_trace": "User Trace",
        "sample": "Sample",
        "layers": "layers",
        "dim": "dim",
        "heads": "heads",
        "params": "params",
        "passes": "passes",
        "steps": "steps",
    },
    "ko": {
        "title": "LLM 타일 시각화 도구",
        "subtitle": "대화형 트레이스 재생기 및 프레젠테이션 런처",
        "select_header": "트레이스 아카이브 선택",
        "no_traces": "traces/ 디렉토리에서 .trace.npz 파일을 찾을 수 없습니다",
        "metadata_header": "트레이스 메타데이터 및 모델 정보",
        "empty_notice": "녹화된 *.trace.npz 파일을 traces/ 폴더에 넣어주세요.",
        "settings_header": "재생 환경 설정",
        "mode": "화면",
        "theme": "테마",
        "detail": "상세도",
        "lang": "언어",
        "refresh_btn": "↻ 새로고침",
        "launch_btn": "발표 시작 ▶",
        "exit_btn": "종료 ✕",
        "user_trace": "사용자 모델",
        "sample": "샘플",
        "layers": "레이어",
        "dim": "임베딩 차원",
        "heads": "어텐션 헤드",
        "params": "파라미터",
        "passes": "패스",
        "steps": "스텝",
    },
    "ja": {
        "title": "LLM タイル可視化ツール",
        "subtitle": "インタラクティブ トレース プレイヤー & ランチャー",
        "select_header": "トレース アーカイブの選択",
        "no_traces": "traces/ ディレクトリに .trace.npz ファイルが見つかりません",
        "metadata_header": "トレース メタデータ & モデル情報",
        "empty_notice": "録画した *.trace.npz ファイルを traces/ フォルダに配置してください。",
        "settings_header": "プレゼンター環境設定",
        "mode": "画面",
        "theme": "テーマ",
        "detail": "詳細度",
        "lang": "言語",
        "refresh_btn": "↻ 更新",
        "launch_btn": "プレゼンテーション開始 ▶",
        "exit_btn": "終了 ✕",
        "user_trace": "モデル",
        "sample": "サンプル",
        "layers": "層",
        "dim": "次元",
        "heads": "ヘッド",
        "params": "パラメータ",
        "passes": "パス",
        "steps": "ステップ",
    },
}


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """Discovered trace file with display name and location metadata."""

    path: Path
    display_name: str
    category: str
    size_kb: float


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    """Summary metadata extracted from a trace archive manifest."""

    model_name: str
    n_layers: int
    emb_dim: int
    n_heads: int | None
    n_params: int
    n_passes: int
    total_steps: int
    prompt_preview: str
    error: str | None = None


def discover_traces(traces_dir: Path | None = None) -> list[TraceEntry]:
    """Discover all valid .trace.npz archives from candidate paths in priority order."""
    candidates: list[Path] = []
    if traces_dir is not None:
        candidates.append(traces_dir)
    else:
        # 1. Portable USB executable-adjacent traces
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / "traces")
            if (
                sys.platform == "darwin"
                and exe_dir.name == "MacOS"
                and exe_dir.parent.name == "Contents"
            ):
                candidates.append(exe_dir.parent.parent.parent / "traces")
                candidates.append(exe_dir.parent.parent / "traces")

        # 2. Project root / CWD traces
        candidates.append(Path("traces"))

        # 3. PyInstaller bundled _MEIPASS
        if hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / "traces")

        # 4. Package directory fallback
        pkg_dir = Path(__file__).resolve().parent.parent.parent.parent / "traces"
        candidates.append(pkg_dir)

    seen_paths: set[Path] = set()
    user_entries: list[TraceEntry] = []
    sample_entries: list[TraceEntry] = []

    for c in candidates:
        if not c.is_dir():
            continue
        try:
            for p in sorted(c.glob("**/*.trace.npz")):
                resolved = p.resolve()
                if resolved in seen_paths or not resolved.is_file():
                    continue
                seen_paths.add(resolved)
                is_sample = "samples" in resolved.parts
                cat = "Sample" if is_sample else "User Trace"
                size_kb = resolved.stat().st_size / 1024.0
                entry = TraceEntry(
                    path=resolved,
                    display_name=resolved.name,
                    category=cat,
                    size_kb=size_kb,
                )
                if is_sample:
                    sample_entries.append(entry)
                else:
                    user_entries.append(entry)
        except Exception as exc:
            logger.debug("Error scanning directory %s for traces: %s", c, exc)

    # Sort User Traces alphabetically first, followed by Sample Traces alphabetically
    user_entries.sort(key=lambda e: e.display_name.lower())
    sample_entries.sort(key=lambda e: e.display_name.lower())
    return [*user_entries, *sample_entries]


def read_trace_metadata(path: Path) -> TraceMetadata:
    """Read summary metadata from trace archive manifest without unpacking weights."""
    if not path.is_file():
        return TraceMetadata("missing", 0, 0, None, 0, 0, 0, "", error="File does not exist")
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "__manifest__" not in npz:
                return TraceMetadata(
                    "invalid", 0, 0, None, 0, 0, 0, "", error="Missing __manifest__ in archive"
                )
            raw_bytes = npz["__manifest__"].tobytes()
            data: dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
            model = data.get("model", {})
            passes = data.get("passes", [])
            total_steps = sum(len(p.get("steps", [])) for p in passes)
            prompt = data.get("prompt", "")
            if not prompt and passes:
                prompt = passes[0].get("text_before", "")
            preview = prompt.replace("\n", " ")
            if len(preview) > 65:
                preview = preview[:62] + "…"
            return TraceMetadata(
                model_name=str(model.get("name", "gpt_model")),
                n_layers=int(model.get("n_layers", 0)),
                emb_dim=int(model.get("emb_dim", 0)),
                n_heads=model.get("n_heads"),
                n_params=int(model.get("n_params", 0)),
                n_passes=len(passes),
                total_steps=total_steps,
                prompt_preview=preview,
            )
    except Exception as exc:
        return TraceMetadata("corrupted", 0, 0, None, 0, 0, 0, "", error=str(exc))


class LauncherApp:
    """High-DPI pygame-ce graphical launcher with trace dropdown selector and preview."""

    DROPDOWN_MAX_VISIBLE = 8

    def __init__(
        self,
        settings: PresenterSettings | None = None,
        traces_dir: Path | None = None,
        *,
        headless: bool = False,
        viewport_size: tuple[int, int] | None = None,
    ) -> None:
        self.settings = settings or PresenterSettings()
        self.traces_dir = traces_dir
        self.headless = headless

        enable_high_dpi()
        if not pygame.font.get_init():
            pygame.font.init()

        win_size = viewport_size or (1180, 700)
        self.window = WindowManager(
            mode="windowed",
            display=self.settings.window_display,
            windowed_size=win_size,
            title="LLM Tile Visualizer — Presentation Launcher",
        )

        if headless:
            self.window.windowed_size = win_size
            self.window.size = win_size
            self.window.logical_size = win_size
            self.window.drawable_size = win_size
            self.window.dpi_scale = (1.0, 1.0)
            self.surface: pygame.Surface = pygame.Surface(self.window.size)
        else:
            self.surface = self.window.create(headless=False)

        sw, sh = self.surface.get_size()
        self.scale = UiScale.from_viewport(sw, sh)
        self.theme: Theme = get_theme(self.settings.theme)
        self.fonts = FontManager(self.theme.font_family)

        self.traces: list[TraceEntry] = []
        self.selected_index: int = 0
        self._dropdown_open: bool = False
        self._dropdown_scroll: int = 0
        self._metadata_cache: dict[Path, TraceMetadata] = {}

        self.should_quit: bool = False
        self.launch_requested: bool = False
        self.exit_code: int = 0
        self._hover_pos: tuple[int, int] = (0, 0)

        # Hit rectangles populated during draw()
        self._dropdown_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[tuple[int, pygame.Rect]] = []
        self._launch_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._refresh_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._mode_pill_rect = pygame.Rect(0, 0, 0, 0)
        self._theme_pill_rect = pygame.Rect(0, 0, 0, 0)
        self._detail_pill_rect = pygame.Rect(0, 0, 0, 0)
        self._lang_pill_rect = pygame.Rect(0, 0, 0, 0)

        self.refresh_traces()

    def refresh_traces(self) -> None:
        """Scan candidate directories for traces and restore or clamp active selection."""
        cur_path = self.selected_trace.path if self.selected_trace is not None else None
        self.traces = discover_traces(self.traces_dir)
        self._dropdown_scroll = 0

        # Try to re-select previously selected path
        if cur_path is not None:
            for idx, entry in enumerate(self.traces):
                if entry.path == cur_path:
                    self.selected_index = idx
                    return
        self.selected_index = 0 if self.traces else -1

    @property
    def selected_trace(self) -> TraceEntry | None:
        """Return the currently selected TraceEntry, or None if trace list is empty."""
        if 0 <= self.selected_index < len(self.traces):
            return self.traces[self.selected_index]
        return None

    def get_metadata(self, entry: TraceEntry) -> TraceMetadata:
        """Fetch and cache summary metadata for a trace archive."""
        if entry.path not in self._metadata_cache:
            self._metadata_cache[entry.path] = read_trace_metadata(entry.path)
        return self._metadata_cache[entry.path]

    def _tr(self, key: str) -> str:
        """Look up localized UI text string for active presenter locale."""
        loc = self.settings.locale if self.settings.locale in _TEXTS else "en"
        return _TEXTS[loc].get(key, _TEXTS["en"].get(key, key))

    def _ensure_selected_visible(self) -> None:
        """Scroll dropdown window so currently selected item is in visible range."""
        if not self.traces:
            return
        if self.selected_index < self._dropdown_scroll:
            self._dropdown_scroll = self.selected_index
        elif self.selected_index >= self._dropdown_scroll + self.DROPDOWN_MAX_VISIBLE:
            self._dropdown_scroll = self.selected_index - self.DROPDOWN_MAX_VISIBLE + 1

    def _cycle_locale(self) -> None:
        """Cycle caption and launcher locale between EN, KO, and JA."""
        order = ("en", "ko", "ja")
        cur = self.settings.locale
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "en"
        self.settings.locale = nxt
        self.fonts.clear_cache()

    def _cycle_detail(self) -> None:
        """Cycle detail level between Story (1), Standard (2), and Deep (3)."""
        nxt = (self.settings.detail_level % 3) + 1
        self.settings.detail_level = nxt
        if nxt == 1:
            self.settings.later_pass_level = 1

    def _cycle_theme(self) -> None:
        """Toggle launcher and presentation theme between dark and light."""
        nxt = "light" if self.theme.name == "dark" else "dark"
        self.settings.theme = nxt
        self.theme = get_theme(nxt)
        self.fonts = FontManager(self.theme.font_family)

    def _toggle_window_mode(self) -> None:
        """Toggle presentation window mode between borderless and windowed."""
        self.settings.window_mode = (
            "windowed" if self.settings.window_mode == "borderless" else "borderless"
        )

    def launch_presentation(self) -> int:
        """Load selected trace and hand off execution to App."""
        sel = self.selected_trace
        if sel is None:
            return 1
        meta = self.get_metadata(sel)
        if meta.error:
            logger.warning("Cannot launch corrupted trace %s: %s", sel.display_name, meta.error)
            return 1

        try:
            from tilevis.core.serialization import load_trace

            trace = load_trace(sel.path)
            app = App(trace, settings=self.settings, headless=self.headless)
            self.launch_requested = True
            code = app.run()

            # Restore launcher window surface if presentation exited cleanly
            if not self.headless and not self.should_quit:
                self.surface = self.window.create(headless=False)
            return code
        except Exception as exc:
            logger.error("Failed to launch presentation: %s", exc)
            return 1

    def tick(self, dt: float, events: list[pygame.event.Event]) -> None:
        """Process input events, update hover states, and dispatch launcher actions."""
        _ = dt
        for ev in events:
            if ev.type == pygame.QUIT:
                self.should_quit = True
                return

            if ev.type == pygame.MOUSEMOTION:
                raw_pos = getattr(ev, "pos", (0, 0))
                self._hover_pos = self.window.map_point(raw_pos)
                continue

            if ev.type == pygame.MOUSEWHEEL and self._dropdown_open:
                # Scroll dropdown list with mouse wheel
                self._dropdown_scroll = max(
                    0,
                    min(
                        max(0, len(self.traces) - self.DROPDOWN_MAX_VISIBLE),
                        self._dropdown_scroll - ev.y,
                    ),
                )
                continue

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                raw_pos = getattr(ev, "pos", (0, 0))
                pos = self.window.map_point(raw_pos)

                if self._dropdown_open:
                    # Check click on any visible dropdown item
                    clicked_item = False
                    for idx, rect in self._dropdown_item_rects:
                        if rect.collidepoint(pos):
                            self.selected_index = idx
                            self._dropdown_open = False
                            clicked_item = True
                            break
                    if clicked_item:
                        continue

                    # Click on dropdown button itself toggles it closed
                    if self._dropdown_btn_rect.collidepoint(pos):
                        self._dropdown_open = False
                        continue

                    # Clicking anywhere outside closes dropdown
                    self._dropdown_open = False
                    continue

                # Dropdown is closed: check interactive controls
                if self._dropdown_btn_rect.collidepoint(pos):
                    if self.traces:
                        self._dropdown_open = True
                        self._ensure_selected_visible()
                    continue

                if self._launch_btn_rect.collidepoint(pos):
                    self.launch_presentation()
                    continue

                if self._refresh_btn_rect.collidepoint(pos):
                    self.refresh_traces()
                    continue

                if self._mode_pill_rect.collidepoint(pos):
                    self._toggle_window_mode()
                    continue

                if self._theme_pill_rect.collidepoint(pos):
                    self._cycle_theme()
                    continue

                if self._detail_pill_rect.collidepoint(pos):
                    self._cycle_detail()
                    continue

                if self._lang_pill_rect.collidepoint(pos):
                    self._cycle_locale()
                    continue

            if ev.type == pygame.KEYDOWN:
                key = getattr(ev, "key", 0)

                if key == pygame.K_ESCAPE:
                    if self._dropdown_open:
                        self._dropdown_open = False
                    else:
                        self.should_quit = True
                    continue

                if key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self._dropdown_open:
                        self._dropdown_open = False
                    else:
                        self.launch_presentation()
                    continue

                if key == pygame.K_DOWN:
                    if self.traces:
                        self.selected_index = (self.selected_index + 1) % len(self.traces)
                        self._ensure_selected_visible()
                    continue

                if key == pygame.K_UP:
                    if self.traces:
                        self.selected_index = (self.selected_index - 1) % len(self.traces)
                        self._ensure_selected_visible()
                    continue

                if key == pygame.K_r:
                    self.refresh_traces()
                    continue

                if key == pygame.K_t:
                    self._cycle_theme()
                    continue

                if key == pygame.K_d:
                    self._cycle_detail()
                    continue

                if key == pygame.K_l:
                    self._cycle_locale()
                    continue

                if key == pygame.K_w:
                    self._toggle_window_mode()
                    continue

    def draw(self, surface: pygame.Surface | None = None) -> None:
        """Render the launcher viewport with High-DPI layout and crisp geometry."""
        target = surface if surface is not None else self.surface
        sw, sh = target.get_size()
        if (sw, sh) != (self.scale.viewport_w, self.scale.viewport_h):
            self.scale = UiScale.from_viewport(sw, sh)

        target.fill(self.theme.background)

        # 1. Header Section
        pad = max(16, self.scale.px(28))
        top_y = max(12, self.scale.px(20))
        avail_w = sw - 2 * pad

        title_px = max(18, self.scale.font_px(32))
        sub_px = max(12, self.scale.font_px(16))

        title_surf = self.fonts.render_text(
            self._tr("title"),
            title_px,
            self.theme.accent,
            bold=True,
            family=self.theme.font_family,
        )
        target.blit(title_surf, (pad, top_y))

        sub_surf = self.fonts.render_text(
            self._tr("subtitle"),
            sub_px,
            self.theme.text_secondary,
            family=self.theme.font_family,
        )
        target.blit(sub_surf, (pad, top_y + title_surf.get_height() + 4))

        # Top-right quick language selector badge
        lang_txt = f"{self._tr('lang')}: {self.settings.locale.upper()}"
        lang_w = max(80, self.scale.px(100))
        lang_h = max(26, self.scale.px(32))
        self._lang_pill_rect = pygame.Rect(sw - pad - lang_w, top_y + 4, lang_w, lang_h)
        draw_pill(
            target,
            self._lang_pill_rect,
            lang_txt,
            self.fonts,
            max(10, self.scale.font_px(14)),
            self.theme.text_primary,
            self.theme.panel,
            border=self.theme.border,
        )

        content_y = (
            top_y + title_surf.get_height() + sub_surf.get_height() + max(14, self.scale.px(20))
        )

        # 2. Main Content Card
        card_h = sh - content_y - max(16, self.scale.px(24))
        main_card_rect = pygame.Rect(pad, content_y, avail_w, card_h)
        draw_panel(target, main_card_rect, self.theme, self.scale, alt=False)

        card_pad = max(14, self.scale.px(24))
        cur_y = main_card_rect.y + card_pad
        inner_w = main_card_rect.width - 2 * card_pad

        # 3. Trace Selection Dropdown Section
        lbl_surf = self.fonts.render_text(
            self._tr("select_header"),
            max(11, self.scale.font_px(15)),
            self.theme.accent,
            bold=True,
            family=self.theme.font_family,
        )
        target.blit(lbl_surf, (main_card_rect.x + card_pad, cur_y))
        cur_y += lbl_surf.get_height() + max(6, self.scale.px(8))

        btn_h = max(36, self.scale.px(44))
        self._dropdown_btn_rect = pygame.Rect(main_card_rect.x + card_pad, cur_y, inner_w, btn_h)

        is_btn_hover = self._dropdown_btn_rect.collidepoint(self._hover_pos)
        draw_panel(
            target,
            self._dropdown_btn_rect,
            self.theme,
            self.scale,
            alt=True,
            highlighted=(self._dropdown_open or is_btn_hover),
        )

        # Selected trace label + category pill inside dropdown button
        sel = self.selected_trace
        if sel is not None:
            cat_label = (
                self._tr("user_trace") if sel.category == "User Trace" else self._tr("sample")
            )
            pill_w = max(60, self.scale.px(80))
            pill_h = btn_h - 12
            pill_rect = pygame.Rect(
                self._dropdown_btn_rect.x + 8, self._dropdown_btn_rect.y + 6, pill_w, pill_h
            )
            pill_bg = (
                self.theme.accent
                if sel.category == "User Trace"
                else (self.theme.panel if self.theme.name == "dark" else self.theme.panel_alt)
            )
            pill_fg = (255, 255, 255) if sel.category == "User Trace" else self.theme.text_secondary
            draw_pill(
                target,
                pill_rect,
                cat_label,
                self.fonts,
                max(9, self.scale.font_px(13)),
                pill_fg,
                pill_bg,
                border=self.theme.border,
            )

            name_x = pill_rect.right + 12
            name_surf = self.fonts.render_text(
                sel.display_name,
                max(12, self.scale.font_px(18)),
                self.theme.text_primary,
                bold=True,
                family=self.theme.font_family,
                max_width=inner_w - pill_w - 70,
            )
            btn_cy = self._dropdown_btn_rect.y + (btn_h - name_surf.get_height()) // 2
            target.blit(name_surf, (name_x, btn_cy))

            # Size badge
            size_txt = f"{sel.size_kb / 1024.0:.1f} MB"
            size_surf = self.fonts.render_text(
                size_txt,
                max(10, self.scale.font_px(14)),
                self.theme.text_secondary,
                family=self.theme.font_family,
            )
            sz_x = self._dropdown_btn_rect.right - 48 - size_surf.get_width()
            sz_y = self._dropdown_btn_rect.y + (btn_h - size_surf.get_height()) // 2
            target.blit(size_surf, (sz_x, sz_y))
        else:
            empty_txt_surf = self.fonts.render_text(
                self._tr("no_traces"),
                max(11, self.scale.font_px(15)),
                self.theme.text_secondary,
                family=self.theme.font_family,
            )
            e_x = self._dropdown_btn_rect.x + 16
            e_y = self._dropdown_btn_rect.y + (btn_h - empty_txt_surf.get_height()) // 2
            target.blit(empty_txt_surf, (e_x, e_y))

        # Arrow indicator
        arrow_char = "▲" if self._dropdown_open else "▼"
        arrow_surf = self.fonts.render_text(
            arrow_char,
            max(11, self.scale.font_px(15)),
            self.theme.accent if self._dropdown_open else self.theme.text_secondary,
            bold=True,
            family=self.theme.font_family,
        )
        arr_x = self._dropdown_btn_rect.right - 28
        arr_y = self._dropdown_btn_rect.y + (btn_h - arrow_surf.get_height()) // 2
        target.blit(arrow_surf, (arr_x, arr_y))

        cur_y += btn_h + max(14, self.scale.px(18))

        # 4. Metadata Preview Card
        meta_h = max(140, self.scale.px(180))
        meta_rect = pygame.Rect(main_card_rect.x + card_pad, cur_y, inner_w, meta_h)
        draw_panel(target, meta_rect, self.theme, self.scale, alt=True)

        meta_hdr_surf = self.fonts.render_text(
            self._tr("metadata_header"),
            max(11, self.scale.font_px(14)),
            self.theme.text_secondary,
            bold=True,
            family=self.theme.font_family,
        )
        target.blit(meta_hdr_surf, (meta_rect.x + 16, meta_rect.y + 12))

        if sel is not None:
            meta = self.get_metadata(sel)
            if meta.error:
                err_surf = self.fonts.render_text(
                    f"Warning: {meta.error}",
                    max(11, self.scale.font_px(15)),
                    (220, 80, 80),
                    bold=True,
                    family=self.theme.font_family,
                )
                target.blit(err_surf, (meta_rect.x + 16, meta_rect.y + 40))
            else:
                m_name_surf = self.fonts.render_text(
                    meta.model_name,
                    max(14, self.scale.font_px(22)),
                    self.theme.accent,
                    bold=True,
                    family=self.theme.font_family,
                )
                target.blit(m_name_surf, (meta_rect.x + 16, meta_rect.y + 36))

                # Architecture row
                h_txt = (
                    f"{meta.n_heads} {self._tr('heads')}"
                    if meta.n_heads
                    else f"? {self._tr('heads')}"
                )
                arch_txt = (
                    f"{meta.n_layers} {self._tr('layers')}  ·  "
                    f"{meta.emb_dim} {self._tr('dim')}  ·  "
                    f"{h_txt}  ·  {meta.n_params:,} {self._tr('params')}"
                )
                arch_surf = self.fonts.render_text(
                    arch_txt,
                    max(11, self.scale.font_px(16)),
                    self.theme.text_primary,
                    family=self.theme.font_family,
                )
                target.blit(arch_surf, (meta_rect.x + 16, meta_rect.y + 66))

                # Passes & steps row
                struct_txt = (
                    f"{meta.n_passes} {self._tr('passes')}  ·  "
                    f"{meta.total_steps} {self._tr('steps')}"
                )
                struct_surf = self.fonts.render_text(
                    struct_txt,
                    max(11, self.scale.font_px(15)),
                    self.theme.text_secondary,
                    family=self.theme.font_family,
                )
                target.blit(struct_surf, (meta_rect.x + 16, meta_rect.y + 94))

                # Prompt preview
                if meta.prompt_preview:
                    prompt_txt = f'Prompt: "{meta.prompt_preview}"'
                    prompt_surf = self.fonts.render_text(
                        prompt_txt,
                        max(10, self.scale.font_px(14)),
                        self.theme.text_secondary,
                        family=self.theme.font_family,
                        max_width=inner_w - 32,
                    )
                    target.blit(prompt_surf, (meta_rect.x + 16, meta_rect.y + 122))
        else:
            notice_surf = self.fonts.render_text(
                self._tr("empty_notice"),
                max(11, self.scale.font_px(15)),
                self.theme.text_secondary,
                family=self.theme.font_family,
            )
            target.blit(notice_surf, (meta_rect.x + 16, meta_rect.y + 44))

        cur_y += meta_h + max(14, self.scale.px(18))

        # 5. Presenter Configuration Pills Row
        cfg_hdr_surf = self.fonts.render_text(
            self._tr("settings_header"),
            max(11, self.scale.font_px(14)),
            self.theme.accent,
            bold=True,
            family=self.theme.font_family,
        )
        target.blit(cfg_hdr_surf, (main_card_rect.x + card_pad, cur_y))
        cur_y += cfg_hdr_surf.get_height() + max(6, self.scale.px(8))

        pill_row_h = max(30, self.scale.px(36))
        pill_gap = max(8, self.scale.px(12))
        pill_w = max(110, (inner_w - 3 * pill_gap) // 4)

        # 1) Mode pill
        mode_label = f"{self._tr('mode')}: {self.settings.window_mode.capitalize()}"
        self._mode_pill_rect = pygame.Rect(main_card_rect.x + card_pad, cur_y, pill_w, pill_row_h)
        draw_pill(
            target,
            self._mode_pill_rect,
            mode_label,
            self.fonts,
            max(10, self.scale.font_px(14)),
            self.theme.text_primary,
            self.theme.panel,
            border=self.theme.border,
        )

        # 2) Theme pill
        theme_label = f"{self._tr('theme')}: {self.settings.theme.capitalize()}"
        self._theme_pill_rect = pygame.Rect(
            self._mode_pill_rect.right + pill_gap, cur_y, pill_w, pill_row_h
        )
        draw_pill(
            target,
            self._theme_pill_rect,
            theme_label,
            self.fonts,
            max(10, self.scale.font_px(14)),
            self.theme.text_primary,
            self.theme.panel,
            border=self.theme.border,
        )

        # 3) Detail pill
        detail_names = {1: "Story", 2: "Standard", 3: "Deep"}
        lvl_name = detail_names.get(self.settings.detail_level, "Standard")
        detail_label = f"{self._tr('detail')}: {lvl_name}"
        self._detail_pill_rect = pygame.Rect(
            self._theme_pill_rect.right + pill_gap, cur_y, pill_w, pill_row_h
        )
        draw_pill(
            target,
            self._detail_pill_rect,
            detail_label,
            self.fonts,
            max(10, self.scale.font_px(14)),
            self.theme.text_primary,
            self.theme.panel,
            border=self.theme.border,
        )

        # 4) Language pill
        lang_label = f"{self._tr('lang')}: {self.settings.locale.upper()}"
        self._lang_pill_rect = pygame.Rect(
            self._detail_pill_rect.right + pill_gap, cur_y, pill_w, pill_row_h
        )
        draw_pill(
            target,
            self._lang_pill_rect,
            lang_label,
            self.fonts,
            max(10, self.scale.font_px(14)),
            self.theme.text_primary,
            self.theme.panel,
            border=self.theme.border,
        )

        cur_y += pill_row_h + max(18, self.scale.px(24))

        # 6. Action Buttons Bar
        btn_action_h = max(38, self.scale.px(48))
        refresh_w = max(130, self.scale.px(160))
        self._refresh_btn_rect = pygame.Rect(
            main_card_rect.x + card_pad, cur_y, refresh_w, btn_action_h
        )

        is_ref_hover = self._refresh_btn_rect.collidepoint(self._hover_pos)
        draw_pill(
            target,
            self._refresh_btn_rect,
            self._tr("refresh_btn"),
            self.fonts,
            max(11, self.scale.font_px(15)),
            self.theme.text_primary,
            self.theme.panel_alt if is_ref_hover else self.theme.panel,
            border=self.theme.accent if is_ref_hover else self.theme.border,
        )

        launch_w = max(190, self.scale.px(240))
        self._launch_btn_rect = pygame.Rect(
            main_card_rect.right - card_pad - launch_w, cur_y, launch_w, btn_action_h
        )

        can_launch = sel is not None and not self.get_metadata(sel).error
        is_launch_hover = self._launch_btn_rect.collidepoint(self._hover_pos)
        launch_bg = self.theme.accent if can_launch else (100, 100, 100)
        draw_pill(
            target,
            self._launch_btn_rect,
            self._tr("launch_btn"),
            self.fonts,
            max(12, self.scale.font_px(16)),
            (255, 255, 255),
            launch_bg,
            border=(255, 255, 255) if (is_launch_hover and can_launch) else None,
        )

        # 7. Dropdown Menu List Overlay (Drawn last on top of everything)
        self._dropdown_item_rects = []
        if self._dropdown_open and self.traces:
            item_h = max(30, self.scale.px(38))
            visible_count = min(self.DROPDOWN_MAX_VISIBLE, len(self.traces))
            menu_h = visible_count * item_h + 8
            menu_rect = pygame.Rect(
                self._dropdown_btn_rect.x,
                self._dropdown_btn_rect.bottom + 2,
                self._dropdown_btn_rect.width,
                menu_h,
            )

            draw_panel(
                target,
                menu_rect,
                self.theme,
                self.scale,
                alt=True,
                border_color=self.theme.accent,
            )

            start_idx = self._dropdown_scroll
            end_idx = min(len(self.traces), start_idx + visible_count)

            row_y = menu_rect.y + 4
            for idx in range(start_idx, end_idx):
                entry = self.traces[idx]
                item_rect = pygame.Rect(menu_rect.x + 4, row_y, menu_rect.width - 8, item_h)
                self._dropdown_item_rects.append((idx, item_rect))

                is_selected = idx == self.selected_index
                is_hovered = item_rect.collidepoint(self._hover_pos)

                if is_selected or is_hovered:
                    bg_color = self.theme.panel if is_hovered else self.theme.panel_alt
                    pygame.draw.rect(target, bg_color, item_rect, border_radius=4)
                    if is_selected:
                        pygame.draw.rect(
                            target, self.theme.accent, item_rect, width=1, border_radius=4
                        )

                # Category badge
                cat_pill_rect = pygame.Rect(
                    item_rect.x + 8, item_rect.y + (item_h - 20) // 2, 54, 20
                )
                cat_txt = "User" if entry.category == "User Trace" else "Sample"
                draw_pill(
                    target,
                    cat_pill_rect,
                    cat_txt,
                    self.fonts,
                    max(8, self.scale.font_px(11)),
                    (
                        (255, 255, 255)
                        if entry.category == "User Trace"
                        else self.theme.text_secondary
                    ),
                    self.theme.accent if entry.category == "User Trace" else self.theme.panel,
                )

                # Filename
                item_name_surf = self.fonts.render_text(
                    entry.display_name,
                    max(11, self.scale.font_px(15)),
                    self.theme.accent if is_selected else self.theme.text_primary,
                    bold=is_selected,
                    family=self.theme.font_family,
                    max_width=item_rect.width - 160,
                )
                name_y = item_rect.y + (item_h - item_name_surf.get_height()) // 2
                target.blit(item_name_surf, (cat_pill_rect.right + 10, name_y))

                # Size
                sz_txt = f"{entry.size_kb / 1024.0:.1f} MB"
                sz_surf = self.fonts.render_text(
                    sz_txt,
                    max(10, self.scale.font_px(13)),
                    self.theme.text_secondary,
                    family=self.theme.font_family,
                )
                sz_x = item_rect.right - 12 - sz_surf.get_width()
                sz_y = item_rect.y + (item_h - sz_surf.get_height()) // 2
                target.blit(sz_surf, (sz_x, sz_y))

                row_y += item_h

    def run(self) -> int:
        """Main event loop for the launcher GUI."""
        fps = 60
        clock = pygame.time.Clock()

        self.draw()
        if not self.headless:
            self.window.present()

        try:
            while not self.should_quit:
                if self.headless:
                    events = []
                else:
                    first_ev = pygame.event.wait(100)
                    events = (
                        [first_ev, *pygame.event.get()] if first_ev.type != pygame.NOEVENT else []
                    )

                dt = min(clock.tick(fps) / 1000.0, 0.1)
                self.tick(dt, events)
                self.draw()
                if not self.headless:
                    self.window.present()

                if self.headless:
                    break
        finally:
            if not self.headless:
                self.window.destroy()

        return self.exit_code
