"""FontManager for loading bundled OFL fonts with size caching.

Single responsibility: manage font glyph surfaces safely without relying on system fonts.
Allowed dependencies: pygame, stdlib.
"""

from __future__ import annotations

import io
import logging
from importlib import resources

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


class FontManager:
    """Loads bundled OFL TrueType fonts via importlib.resources and caches Font instances."""

    def __init__(self, default_family: str = "NotoSans") -> None:
        self.default_family = default_family
        self._font_bytes_cache: dict[tuple[str, bool], bytes] = {}
        self._font_cache: dict[tuple[str, bool, int], pygame.font.Font] = {}
        self._glyph_ok_cache: dict[tuple[str, bool, str], bool] = {}

    def _ensure_font_init(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

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
    ) -> pygame.font.Font:
        """Return cached pygame.font.Font for the requested family, weight, and pixel size."""
        self._ensure_font_init()
        fam = family or self.default_family
        norm_fam = fam.strip().lower().replace(" ", "")
        px = max(8, int(size_px))
        cache_key = (norm_fam, bold, px)

        cached = self._font_cache.get(cache_key)
        if cached is not None:
            return cached

        raw_bytes = self._load_bytes(fam, bold)
        font = pygame.font.Font(io.BytesIO(raw_bytes), px)
        self._font_cache[cache_key] = font
        return font

    def clear_cache(self) -> None:
        """Clear cached Font objects (e.g. if pygame display re-initializes)."""
        self._font_cache.clear()

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
        font = self.get(size_px, bold=bold, family=family)
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
        font = self.get(size_px, bold=bold, family=family)
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
        font = self.get(size_px, bold=bold, family=family)
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
