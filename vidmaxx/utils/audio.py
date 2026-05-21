"""
Audio utilities — pure functions, no pipeline state.
"""

from pathlib import Path

import soundfile as sf


def wav_duration_sec(path: Path) -> float:
    """Return exact duration of a WAV file without loading the audio data."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def wav_durations_sec(paths: list[Path]) -> list[float]:
    """Return durations for a list of WAV paths, in order."""
    return [wav_duration_sec(p) for p in paths]
