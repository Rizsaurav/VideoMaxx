"""
ASS subtitle generator — groups words into ~5-word chunks per Dialogue event.

Times in CaptionWord are absolute (full-video seconds). Pass chapter_start_sec
so write_ass converts them to chapter-relative. Emphasis words render yellow
via ASS inline colour override within the chunk.
"""

from pathlib import Path

from vidmaxx.models.timeline import CaptionSegment, CaptionWord

# ASS colours: &HAABBGGRR (AA=00 → fully opaque)
_WHITE        = "&H00FFFFFF"
_YELLOW       = "&H0000FFFF"
_BLACK_OUTLINE = "&H00000000"
_SEMI_BLACK_BG = "&H80000000"

WORDS_PER_LINE = 5   # tweak here to go wider/narrower

_HEADER = f"""\
[Script Info]
Title: VidMaxx Captions
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,68,{_WHITE},{_WHITE},{_BLACK_OUTLINE},{_SEMI_BLACK_BG},0,0,0,0,100,100,0,0,1,4,1,2,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(sec: float) -> str:
    """Seconds → ASS timestamp h:mm:ss.cc"""
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = sec % 60
    cs  = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def _chunk_words(words: list[CaptionWord], size: int) -> list[list[CaptionWord]]:
    return [words[i:i + size] for i in range(0, len(words), size)]


def _render_chunk(chunk: list[CaptionWord]) -> str:
    """Build the ASS text for one chunk, colouring emphasis words yellow."""
    parts = []
    for word in chunk:
        # Strip any stray ASS control chars from the raw word
        clean = word.word.replace("{", "").replace("}", "").replace("\n", " ").strip()
        if not clean:
            continue
        if word.is_emphasis:
            parts.append(f"{{\\c{_YELLOW}&}}{clean}{{\\r}}")
        else:
            parts.append(clean)
    return " ".join(parts)


def write_ass(
    segments: list[CaptionSegment],
    out_path: Path,
    chapter_start_sec: float = 0.0,
) -> None:
    lines = [_HEADER]

    for seg in segments:
        if not seg.words:
            continue

        for chunk in _chunk_words(seg.words, WORDS_PER_LINE):
            start = chunk[0].start_sec - chapter_start_sec
            end   = chunk[-1].end_sec  - chapter_start_sec

            if end <= 0.0 or end <= start:
                continue

            text = _render_chunk(chunk)
            if not text:
                continue

            lines.append(
                f"Dialogue: 0,{_ts(max(0.0, start))},{_ts(end)},Default,,0,0,0,,{text}\n"
            )

    out_path.write_text("".join(lines), encoding="utf-8")
