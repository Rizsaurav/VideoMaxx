"""
Kokoro TTS service via mlx-audio.

Lifecycle:
  - Created once at Stage 4 start. Model and pipeline loaded once.
  - generate() calls the KokoroPipeline — no model reload per sentence.
  - unload() frees the model from memory.

Voice:
  - Uses Kokoro built-in voices (e.g. "af_heart", "af_bella").
  - No reference audio required.

Output:
  Each generate() call writes audio/{sentence_id}.wav to out_dir.
  The WAV includes silence padding (pause_before + pause_after) baked in.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model

from vidmaxx.models.sentence import Sentence

log = structlog.get_logger(__name__)

_MODEL_REPO = "mlx-community/Kokoro-82M-bf16"
_SAMPLE_RATE = 24_000


def _apply_emphasis(text: str, emphasis_words: list[str]) -> str:
    for word in emphasis_words:
        text = re.sub(
            rf"\b{re.escape(word)}\b",
            word.upper(),
            text,
            flags=re.IGNORECASE,
        )
    return text


def _silence(ms: int, sample_rate: int) -> np.ndarray:
    n_samples = max(0, int(sample_rate * ms / 1000))
    return np.zeros(n_samples, dtype=np.float32)


class TTSService:
    def __init__(
        self,
        voice: str = "af_heart",
    ) -> None:
        self._voice = voice
        self._model = None
        self._pipeline = None
        self._sample_rate = _SAMPLE_RATE
        self._load()

    def _load(self) -> None:
        log.info("tts_loading", model=_MODEL_REPO, voice=self._voice)
        self._model = load_model(_MODEL_REPO)
        self._pipeline = KokoroPipeline(
            lang_code="a",
            model=self._model,
            repo_id=_MODEL_REPO,
        )
        log.info("tts_ready_kokoro", voice=self._voice)

    def unload(self) -> None:
        self._model = None
        self._pipeline = None
        log.info("tts_unloaded")

    def __enter__(self) -> "TTSService":
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    def generate(
        self,
        sentence: Sentence,
        out_dir: Path,
    ) -> Path:
        if self._model is None:
            raise RuntimeError("TTSService has been unloaded")

        text = _apply_emphasis(sentence.text, sentence.emphasis_words)
        speed = {"slow": 0.85, "fast": 1.15, "medium": 1.0}.get(sentence.pace, 1.0)

        log.debug("tts_generate", id=sentence.id, chars=len(text), speed=speed)

        chunks: list[np.ndarray] = []
        for result in self._pipeline(text, voice=self._voice, speed=speed):
            if result.audio is not None:
                chunks.append(np.array(result.audio, dtype=np.float32).squeeze())

        if not chunks:
            data = np.zeros(int(_SAMPLE_RATE * 0.5), dtype=np.float32)
        else:
            data = np.concatenate(chunks)

        pre = _silence(sentence.pause_before_ms, _SAMPLE_RATE)
        post = _silence(sentence.pause_after_ms, _SAMPLE_RATE)
        padded = np.concatenate([pre, data, post])

        out_path = out_dir / f"{sentence.id}.wav"
        sf.write(str(out_path), padded, _SAMPLE_RATE)

        log.debug("tts_saved", path=str(out_path), duration_sec=round(len(padded) / _SAMPLE_RATE, 2))
        return out_path
