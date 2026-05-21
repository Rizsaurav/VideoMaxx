"""
Vocal post-processing — light compression only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def apply_studio_chain(wav_path: Path) -> Path:
    try:
        from pedalboard import Compressor, Pedalboard
    except ImportError as exc:
        raise ImportError("Run: pip install pedalboard") from exc

    data, sr = sf.read(str(wav_path), dtype="float32")
    mono = data[:, 0] if data.ndim == 2 else data.flatten()

    board = Pedalboard([
        Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=10.0, release_ms=100.0),
    ])

    processed = board(mono[np.newaxis, :], sr)
    sf.write(str(wav_path), processed[0], sr)
    return wav_path
