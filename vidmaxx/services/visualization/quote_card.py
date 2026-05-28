"""
Generates full-screen quote card images (PNG) for high-exaggeration sentences.

The PNG is saved to assets_dir/{sentence_id}.png and treated as a static image
by the renderer — Ken Burns is applied automatically at render time.

Usage: call generate_quote_cards() from s03_assets.py before stock fetching.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import structlog

log = structlog.get_logger(__name__)

# Card dimensions
_W, _H = 1920, 1080
_ACCENT_COLOR = (230, 57, 70)   # #E63946 — matches viz palette
_TEXT_COLOR   = (235, 235, 235)
_DIM_COLOR    = (136, 136, 136)

# Font search order — first match wins
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def generate_quote_card(sentence_id: str, text: str, assets_dir: Path) -> Path:
    """Render a quote card PNG for one sentence. Returns the output path."""
    out = assets_dir / f"{sentence_id}.png"
    if out.exists():
        log.debug("quote_card_cache_hit", id=sentence_id)
        return out

    img  = Image.new("RGB", (_W, _H), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)

    # Accent line — top and bottom thirds
    accent_y_top = _H // 3 - 2
    accent_y_bot = 2 * _H // 3 + 2
    line_x0, line_x1 = 160, _W - 160

    draw.rectangle([line_x0, accent_y_top, line_x1, accent_y_top + 3], fill=_ACCENT_COLOR)
    draw.rectangle([line_x0, accent_y_bot, line_x1, accent_y_bot + 3], fill=_ACCENT_COLOR)

    # Wrap text to ~36 chars per line (looks good at font size 80)
    wrapped = textwrap.wrap(text.strip(), width=36)
    font_large = _load_font(80)
    font_small = _load_font(32)

    # Calculate total text block height
    line_spacing = 100
    total_h = len(wrapped) * line_spacing - (line_spacing - 80)
    y_start = (_H - total_h) // 2

    for i, line in enumerate(wrapped):
        y = y_start + i * line_spacing
        bbox = draw.textbbox((0, 0), line, font=font_large)
        tw = bbox[2] - bbox[0]
        x  = (_W - tw) // 2
        draw.text((x, y), line, font=font_large, fill=_TEXT_COLOR)

    # Subtle label below text block
    label = "—"
    y_label = y_start + len(wrapped) * line_spacing + 24
    bbox = draw.textbbox((0, 0), label, font=font_small)
    x_label = (_W - (bbox[2] - bbox[0])) // 2
    draw.text((x_label, y_label), label, font=font_small, fill=_DIM_COLOR)

    img.save(str(out), format="PNG")
    log.info("quote_card_generated", id=sentence_id, out=str(out))
    return out


def select_quote_card_sentences(
    sentences: list,
    threshold: float = 0.75,
    max_per_chapter: int = 2,
) -> set[str]:
    """Return sentence IDs that should become quote cards.

    Picks the top `max_per_chapter` highest-exaggeration sentences per chapter
    that exceed `threshold`. Returns a set of sentence IDs.
    """
    from collections import defaultdict

    by_chapter: dict[str, list] = defaultdict(list)
    for s in sentences:
        if s.exaggeration >= threshold:
            ch = s.id.split("_s")[0]
            by_chapter[ch].append(s)

    selected = set()
    for ch_sentences in by_chapter.values():
        ch_sentences.sort(key=lambda s: s.exaggeration, reverse=True)
        for s in ch_sentences[:max_per_chapter]:
            selected.add(s.id)

    return selected
