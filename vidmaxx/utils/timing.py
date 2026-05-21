"""
Pause timing utilities.

All pause logic is centralised here so the timeline compiler and any
future callers stay in sync with the constants in config/constants.py.
"""

from vidmaxx.config.constants import (
    PAUSE_MS_COMMA,
    PAUSE_MS_EXCLAMATION,
    PAUSE_MS_PARAGRAPH_BREAK,
    PAUSE_MS_PERIOD,
    PAUSE_MS_QUESTION,
)


def pause_ms_for_punctuation(text: str, is_paragraph_end: bool = False) -> int:
    """
    Return the pause_after_ms value for a sentence based on its terminal
    punctuation. Paragraph-end flag overrides punctuation when set.
    """
    if is_paragraph_end:
        return PAUSE_MS_PARAGRAPH_BREAK
    stripped = text.rstrip()
    if not stripped:
        return PAUSE_MS_PERIOD
    last = stripped[-1]
    return {
        ".": PAUSE_MS_PERIOD,
        "?": PAUSE_MS_QUESTION,
        "!": PAUSE_MS_EXCLAMATION,
        ",": PAUSE_MS_COMMA,
    }.get(last, PAUSE_MS_PERIOD)


def sentence_start_times(durations_sec: list[float]) -> list[float]:
    """
    Given a list of per-sentence durations (WAV file lengths including padding),
    return the absolute start time of each sentence in the concatenated narration.

    len(result) == len(durations_sec)
    """
    starts: list[float] = []
    t = 0.0
    for d in durations_sec:
        starts.append(t)
        t += d
    return starts


def find_sentence_index(word_start_sec: float, sentence_starts: list[float]) -> int:
    """
    Binary-search sentence_starts to find which sentence a word belongs to.
    Returns the 0-based index of the owning sentence.
    """
    lo, hi = 0, len(sentence_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sentence_starts[mid] <= word_start_sec:
            lo = mid
        else:
            hi = mid - 1
    return lo
