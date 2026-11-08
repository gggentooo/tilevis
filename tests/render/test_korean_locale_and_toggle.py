"""Tests for Korean locale support and interactive language toggle button."""

from __future__ import annotations

import pygame
from tests.fixtures.fake_traces import make_fake_trace

from tilevis.core.config import PresenterSettings
from tilevis.core.kinds import DetailLevel, Semantic
from tilevis.narration.captions import CaptionProvider
from tilevis.render.app import App
from tilevis.render.commands import Command
from tilevis.render.input_map import InputMap
from tilevis.render.layout import UiScale
from tilevis.render.player import StepPlayer
from tilevis.render.text import FontManager
from tilevis.render.widgets.buttons import ButtonsWidget


def test_korean_caption_provider_translations() -> None:
    """Korean caption provider translates titles and bodies accurately."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=2)
    provider = CaptionProvider("ko")
    pass_rec = trace.passes[0]

    # 1. Prompt step
    cap_prompt = provider.caption(pass_rec.steps[0], trace, pass_rec)
    assert "시작 텍스트" in cap_prompt.title
    assert "모델이 지금까지 입력된 텍스트를 읽고" in cap_prompt.body

    # 2. Embed step
    embed_step = next(s for s in pass_rec.steps if s.semantic == Semantic.EMBED.value)
    cap_embed = provider.caption(embed_step, trace, pass_rec)
    assert "단어 의미와 위치 정보 결합" in cap_embed.title
    assert "16개의 의미 타일" in cap_embed.body

    # 3. Next token step
    next_step = next(s for s in pass_rec.steps if s.semantic == Semantic.NEXT_TOKEN.value)
    cap_next = provider.caption(next_step, trace, pass_rec)
    assert "예측된 다음 단어는" in cap_next.title


def test_font_manager_korean_rendering_and_wrapping() -> None:
    """FontManager renders, truncates, and wraps Korean text safely."""
    fm = FontManager()

    # Render single-line text
    surf = fm.render_text("한국어 레이어 제목", 18, (255, 255, 255))
    assert surf.get_width() > 0
    assert surf.get_height() > 0

    # Truncate Korean text
    truncated = fm.truncate("길고 긴 한국어 설명 텍스트입니다", max_width=60, size_px=16)
    assert truncated.endswith("...")
    assert len(truncated) < len("길고 긴 한국어 설명 텍스트입니다")

    # Wrap multi-word Korean text
    lines = fm.wrap_text(
        "거대 언어 모델이 타일 기반으로 한 단계씩 순방향 연산을 수행합니다.",
        max_width=180,
        size_px=14,
    )
    assert len(lines) >= 2
    assert all(len(line) > 0 for line in lines)


def test_input_map_maps_l_to_toggle_locale() -> None:
    """InputMap maps the 'L' key to Command.TOGGLE_LOCALE."""
    input_map = InputMap()
    event_l = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l, mod=0)
    cmd = input_map.translate_event(event_l)
    assert cmd == Command.TOGGLE_LOCALE


def test_buttons_widget_contains_and_clicks_toggle_locale() -> None:
    """ButtonsWidget presents a language toggle button that responds to clicks."""
    trace = make_fake_trace()
    settings = PresenterSettings(locale="en")
    player = StepPlayer(trace, settings)
    scale = UiScale.from_viewport(1920, 1080)

    widget = ButtonsWidget()
    rect = pygame.Rect(1200, 1020, 600, 40)
    buttons = widget.layout(rect, player, settings, scale=scale)

    lang_btn = next((b for b in buttons if b.cmd == Command.TOGGLE_LOCALE), None)
    assert lang_btn is not None
    assert lang_btn.label == "Lang: EN"
    assert not lang_btn.is_active

    # Click test returns Command.TOGGLE_LOCALE
    cmd = widget.handle_click(lang_btn.rect.center)
    assert cmd == Command.TOGGLE_LOCALE

    # Switching to Korean changes button label and active highlight
    settings_ko = PresenterSettings(locale="ko")
    buttons_ko = widget.layout(rect, player, settings_ko, scale=scale)
    lang_btn_ko = next((b for b in buttons_ko if b.cmd == Command.TOGGLE_LOCALE), None)
    assert lang_btn_ko is not None
    assert lang_btn_ko.label == "Lang: KO"
    assert lang_btn_ko.is_active


def test_app_toggle_locale_command_switches_captions() -> None:
    """App.dispatch_command(Command.TOGGLE_LOCALE) toggles locale and updates captions."""
    trace = make_fake_trace()
    app = App(trace, PresenterSettings(locale="en"), headless=True)
    assert app.settings.locale == "en"
    assert app._current_caption is not None
    initial_title = app._current_caption.title

    # 1. Toggle EN -> KO
    app.dispatch_command(Command.TOGGLE_LOCALE)
    assert app.settings.locale == "ko"
    assert app._current_caption is not None
    ko_title = app._current_caption.title
    assert ko_title != initial_title
    assert "거대 언어 모델" in ko_title or "시작 텍스트" in ko_title

    # 2. Toggle KO -> EN
    app.dispatch_command(Command.TOGGLE_LOCALE)
    assert app.settings.locale == "en"
    assert app._current_caption is not None
    assert app._current_caption.title == initial_title


def test_korean_locale_headless_playback_draw_sweep() -> None:
    """Full step-through playback with Korean locale draws without any runtime error."""
    trace = make_fake_trace(n_layers=2, emb_dim=16, n_tokens=4, n_passes=1, n_heads=2)
    settings = PresenterSettings(locale="ko", detail_level=int(DetailLevel.STANDARD))
    app = App(trace, settings, headless=True, viewport_size=(1920, 1080))

    steps_counted = 0
    while True:
        app.draw()
        steps_counted += 1
        if not app.player.can_next:
            break
        app.scene.finish()
        app.dispatch_command(Command.NEXT)

    assert steps_counted > 0
    assert app.settings.locale == "ko"
