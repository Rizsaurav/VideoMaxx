"""
ASS subtitle generator — one Dialogue event per word.

Times in CaptionWord are absolute (full-video seconds).  Pass
chapter_start_sec so write_ass converts them to chapter-relative before
writing.  Per-word events let each word appear/disappear independently.
Emphasis words render yellow + bold; regular words render white.
"""

from pathlib import Path

from vidmaxx.models.timeline import CaptionSegment

# ASS colour values are &HAABBGGRR (alpha=00 → fully opaque).
# White:  R=FF G=FF B=FF  → BBGGRR=FFFFFF → &H00FFFFFF
# Yellow: R=FF G=FF B=00  → BBGGRR=00FFFF → &H0000FFFF
_WHITE = "&H00FFFFFF"
_YELLOW = "&H0000FFFF"
_BLACK_OUTLINE = "&H00000000"
_SEMI_BLACK_BG = "&H80000000"

_HEADER = f"""\
[Script Info]
Title: VidMaxx Captions
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Normal,Arial,72,{_WHITE},{_WHITE},{_BLACK_OUTLINE},{_SEMI_BLACK_BG},0,0,0,0,100,100,0,0,1,4,1,2,80,80,60,1
Style: Emphasis,Arial,72,{_YELLOW},{_YELLOW},{_BLACK_OUTLINE},{_SEMI_BLACK_BG},-1,0,0,0,100,100,0,0,1,4,1,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(sec: float) -> str:
    """Seconds → ASS timestamp h:mm:ss.cc"""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def write_ass(
    segments: list[CaptionSegment],
    out_path: Path,
    chapter_start_sec: float = 0.0,
) -> None:
    """Write an ASS subtitle file for the given caption segments.

    Each word becomes one Dialogue event.  Times are shifted by
    chapter_start_sec so the file is chapter-relative (starts at 0).
    """
    lines = [_HEADER]
    for seg in segments:
        for word in seg.words:
            start = word.start_sec - chapter_start_sec
            end = word.end_sec - chapter_start_sec
            if end <= start or end <= 0.0:
                continue
            style = "Emphasis" if word.is_emphasis else "Normal"
            # Strip ASS control characters so the parser doesn't break.
            text = word.word.replace("{", "").replace("}", "").replace("\n", " ")
            lines.append(
                f"Dialogue: 0,{_ts(start)},{_ts(end)},{style},,0,0,0,,{text}\n"
            )
    out_path.write_text("".join(lines), encoding="utf-8")
