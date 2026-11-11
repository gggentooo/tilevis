"""FontManager for loading bundled OFL fonts with size caching.

Single responsibility: manage font glyph surfaces safely without relying on system fonts.
Allowed dependencies: pygame, stdlib.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from importlib import resources
from pathlib import Path

import pygame

from tilevis.core.errors import RenderError

logger = logging.getLogger("tilevis.render.text")

_DEFAULT_FONT_FILES: dict[tuple[str, bool], str] = {
    ("notosans", False): "NotoSans-Regular.ttf",
    ("notosans", True): "NotoSans-Bold.ttf",
}

_GLYPH_FALLBACKS: dict[str, str] = {
    "↵": "\\n",
    "→": "->",
    "←": "<-",
    "▸": ">",
    "▾": "v",
    "\u25c0": "<",
    "\u25b6": ">",
    "\u2212": "-",
    "\u2248": "~",
}


def _has_korean(text: str) -> bool:
    """Return True if text contains Hangul syllables, jamo, or compatibility jamo."""
    return any(
        "\uac00" <= ch <= "\ud7a3" or "\u1100" <= ch <= "\u11ff" or "\u3130" <= ch <= "\u318f"
        for ch in text
    )


def _has_japanese(text: str) -> bool:
    """Return True if text contains kana, CJK ideographs, or CJK/fullwidth punctuation."""
    return any(
        "\u3040" <= ch <= "\u30ff"  # hiragana, katakana
        or "\u31f0" <= ch <= "\u31ff"  # katakana phonetic extensions
        or "\u3400" <= ch <= "\u4dbf"  # CJK ext A
        or "\u4e00" <= ch <= "\u9fff"  # CJK unified ideographs
        or "\u3000" <= ch <= "\u303f"  # CJK punctuation (、。)
        or "\uff00" <= ch <= "\uffef"  # halfwidth/fullwidth forms
        for ch in text
    )


def _script_of(text: str) -> str | None:
    """Pick which non-Latin font a text needs: 'korean', 'japanese', or None."""
    if _has_korean(text):
        return "korean"
    if _has_japanese(text):
        return "japanese"
    return None


class FontManager:
    """Loads bundled OFL TrueType fonts via importlib.resources and caches Font instances."""

    def __init__(self, default_family: str = "NotoSans") -> None:
        self.default_family = default_family
        self._font_bytes_cache: dict[tuple[str, bool], bytes] = {}
        self._font_cache: dict[tuple[str, bool, int], pygame.font.Font] = {}
        self._glyph_ok_cache: dict[tuple[str, bool, str], bool] = {}
        self._korean_font_path: dict[bool, str | None] = {}
        self._japanese_font_path: dict[bool, str | None] = {}

    def _ensure_font_init(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

    def _find_korean_font(self, bold: bool) -> str | None:
        """Find an available Korean font path on the host system or bundled assets."""
        if bold in self._korean_font_path:
            return self._korean_font_path[bold]

        # 1. Check bundled assets first
        bundled_names = (
            ("NotoSansKR-Bold.ttf", "NotoSansKR-Regular.ttf")
            if bold
            else ("NotoSansKR-Regular.ttf", "NotoSansKR-Bold.ttf")
        )
        for name in bundled_names:
            try:
                res = resources.files("tilevis.assets.fonts").joinpath(name)
                if res.is_file():
                    path_str = str(res)
                    self._korean_font_path[bold] = path_str
                    return path_str
            except Exception:
                pass

        # 2. Check platform-specific common fonts
        candidates: list[Path] = []
        if sys.platform == "darwin":
            candidates = [
                Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
                Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            ]
        elif sys.platform == "win32":
            windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
            fonts_dir = windir / "Fonts"
            if bold:
                candidates = [
                    fonts_dir / "malgunbd.ttf",
                    fonts_dir / "malgun.ttf",
                    fonts_dir / "gulim.ttc",
                ]
            else:
                candidates = [
                    fonts_dir / "malgun.ttf",
                    fonts_dir / "malgunbd.ttf",
                    fonts_dir / "gulim.ttc",
                ]
        else:  # Linux / Unix
            candidates = [
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            ]

        for cand in candidates:
            if cand.is_file():
                cand_str = str(cand)
                self._korean_font_path[bold] = cand_str
                return cand_str

        self._korean_font_path[bold] = None
        return None

    def _find_japanese_font(self, bold: bool) -> str | None:
        """Find a host font that covers kana and kanji."""
        if bold in self._japanese_font_path:
            return self._japanese_font_path[bold]

        candidates: list[Path] = []
        if sys.platform == "darwin":
            candidates = [
                Path(
                    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
                    if bold
                    else "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
                ),
                Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            ]
        elif sys.platform == "win32":
            fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
            names = (
                ["YuGothB.ttc", "meiryob.ttc", "YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"]
                if bold
                else ["YuGothM.ttc", "YuGothR.ttc", "meiryo.ttc", "msgothic.ttc"]
            )
            candidates = [fonts_dir / n for n in names]
        else:  # Linux / Unix
            candidates = [
                Path(
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
                    if bold
                    else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
                ),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            ]

        for cand in candidates:
            if cand.is_file():
                self._japanese_font_path[bold] = str(cand)
                return str(cand)

        self._japanese_font_path[bold] = None
        return None

    def _load_bytes(self, family: str, bold: bool) -> bytes:
        norm_family = family.strip().lower().replace(" ", "")
        key = (norm_family, bold)
        if key in self._font_bytes_cache:
            return self._font_bytes_cache[key]

        filename = _DEFAULT_FONT_FILES.get(key)
        if filename is None:
            fallback_key = ("notosans", bold)
            filename = _DEFAULT_FONT_FILES[fallback_key]
            if norm_family != "notosans":
                logger.warning("Font family %r not bundled; falling back to NotoSans", family)

        try:
            font_resource = resources.files("tilevis.assets.fonts").joinpath(filename)
            raw = font_resource.read_bytes()
        except Exception as exc:
            raise RenderError(
                f"Failed to load bundled font asset {filename!r} from tilevis.assets.fonts: {exc}"
            ) from exc

        self._font_bytes_cache[key] = raw
        return raw

    def get(
        self,
        size_px: int,
        *,
        bold: bool = False,
        family: str | None = None,
        is_korean: bool = False,
        script: str | None = None,
    ) -> pygame.font.Font:
        """Return cached pygame.font.Font for the requested family, weight, and pixel size."""
        self._ensure_font_init()
        fam = family or self.default_family
        norm_fam = fam.strip().lower().replace(" ", "")
        px = max(8, int(size_px))
        if is_korean:
            script = "korean"
        cache_key = (script or norm_fam, bold, px)

        cached = self._font_cache.get(cache_key)
        if cached is not None:
            return cached

        if script in ("korean", "japanese"):
            kr_path = (
                self._find_korean_font(bold)
                if script == "korean"
                else self._find_japanese_font(bold)
            )
            if kr_path is not None:
                try:
                    font = pygame.font.Font(kr_path, px)
                    self._font_cache[cache_key] = font
                    return font
                except Exception as exc:
                    logger.warning("Failed to load Korean font from %r: %s", kr_path, exc)

        raw_bytes = self._load_bytes(fam, bold)
        font = pygame.font.Font(io.BytesIO(raw_bytes), px)
        self._font_cache[cache_key] = font
        return font

    def clear_cache(self) -> None:
        """Clear cached Font objects (e.g. if pygame display re-initializes)."""
        self._font_cache.clear()
        self._glyph_ok_cache.clear()
        self._korean_font_path.clear()
        self._japanese_font_path.clear()

    def _prepare_text(
        self,
        text: str,
        *,
        bold: bool = False,
        family: str | None = None,
    ) -> str:
        """Replace known symbols unsupported by the active font with ASCII equivalents."""
        if not text or text.isascii():
            return text
        if not any(ch in _GLYPH_FALLBACKS for ch in text):
            return text

        fam = (family or self.default_family).strip().lower().replace(" ", "")
        out_chars: list[str] = []
        for ch in text:
            repl = _GLYPH_FALLBACKS.get(ch)
            if repl is None:
                out_chars.append(ch)
                continue
            cache_key = (fam, bold, ch)
            has_glyph = self._glyph_ok_cache.get(cache_key)
            if has_glyph is None:
                ref_font = self.get(20, bold=bold, family=family)
                notdef_surf = ref_font.render("\ufffe", True, (255, 255, 255), (0, 0, 0))
                ch_surf = ref_font.render(ch, True, (255, 255, 255), (0, 0, 0))
                has_glyph = not (
                    ch_surf.get_size() == notdef_surf.get_size()
                    and pygame.image.tobytes(ch_surf, "RGBA")
                    == pygame.image.tobytes(notdef_surf, "RGBA")
                )
                self._glyph_ok_cache[cache_key] = has_glyph
            out_chars.append(ch if has_glyph else repl)
        return "".join(out_chars)

    def render_text(
        self,
        text: str,
        size_px: int,
        color: tuple[int, int, int],
        *,
        bold: bool = False,
        family: str | None = None,
        max_width: int | None = None,
    ) -> pygame.Surface:
        """Render single-line text, optionally truncating with ellipsis to fit max_width."""
        font = self.get(size_px, bold=bold, family=family, script=_script_of(text))
        display_text = self._prepare_text(text, bold=bold, family=family)
        if max_width is not None and max_width > 0 and font.size(display_text)[0] > max_width:
            display_text = self.truncate(display_text, max_width, size_px, bold=bold, family=family)
        return font.render(display_text, True, color)

    def truncate(
        self,
        text: str,
        max_width: int,
        size_px: int,
        *,
        bold: bool = False,
        family: str | None = None,
    ) -> str:
        """Truncate text with '...' if its rendered pixel width exceeds max_width."""
        font = self.get(size_px, bold=bold, family=family, script=_script_of(text))
        clean_text = self._prepare_text(text, bold=bold, family=family)
        if max_width <= 0:
            return ""
        if font.size(clean_text)[0] <= max_width:
            return clean_text
        ellipsis = "..."
        if font.size(ellipsis)[0] > max_width:
            return ""
        lo, hi = 0, len(clean_text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.size(clean_text[:mid] + ellipsis)[0] <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return clean_text[:lo] + ellipsis

    def wrap_text(
        self,
        text: str,
        max_width: int,
        size_px: int,
        *,
        bold: bool = False,
        family: str | None = None,
        max_lines: int | None = None,
    ) -> list[str]:
        """Word-wrap text into lines fitting within max_width pixels."""
        if max_width <= 0 or not text:
            return []
        font = self.get(size_px, bold=bold, family=family, script=_script_of(text))
        prepared = self._prepare_text(text, bold=bold, family=family)
        lines: list[str] = []

        for paragraph in prepared.splitlines():
            words = paragraph.split(" ")
            current = ""
            for word in words:
                candidate = f"{current} {word}" if current else word
                if font.size(candidate)[0] <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    if font.size(word)[0] <= max_width:
                        current = word
                    else:
                        lines.append(
                            self.truncate(word, max_width, size_px, bold=bold, family=family)
                        )
                        current = ""
            if current or not paragraph:
                lines.append(current)

        if max_lines is not None and max_lines > 0 and len(lines) > max_lines:
            kept = lines[:max_lines]
            kept[-1] = self.truncate(
                kept[-1] + " ...", max_width, size_px, bold=bold, family=family
            )
            return kept

        return lines
