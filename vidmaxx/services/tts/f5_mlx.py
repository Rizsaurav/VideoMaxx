"""
F5-TTS MLX TTS service.

Lifecycle:
  - Created once at Stage 4 start. Model and reference audio loaded once.
  - generate() calls f5tts.sample() directly — no model reload per sentence.
  - unload() frees the model from memory.

Voice cloning:
  - Requires style_pack/voice_ref.wav — 12-15s clean speech at 24kHz.
  - Requires style_pack/voice_ref_transcript.txt — exact transcript of the
    reference audio. F5-TTS uses this to learn the speaker's voice pattern.

Output:
  Each generate() call writes audio/{sentence_id}.wav to out_dir.
  The WAV includes silence padding (pause_before + pause_after) baked in.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vidmaxx.models.sentence import Sentence

log = structlog.get_logger(__name__)

_MODEL_NAME = "lucasnewman/f5-tts-mlx"
_SAMPLE_RATE = 24_000
_TARGET_RMS = 0.1


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
        voice_ref_path: Path,
        voice_ref_transcript: str,
        device: str = "mlx",
    ) -> None:
        self._model = None
        self._ref_audio = None   # preprocessed mlx array, reused every call
        self._ref_text = voice_ref_transcript
        self._ref_audio_path = voice_ref_path
        self._sample_rate = _SAMPLE_RATE
        self._load()

    def _load(self) -> None:
        try:
            import mlx.core as mx
            from f5_tts_mlx import F5TTS
            from f5_tts_mlx.generate import convert_char_to_pinyin
        except ImportError as exc:
            raise ImportError(
                "f5-tts-mlx is not installed. Run: pip install f5-tts-mlx"
            ) from exc

        log.info("tts_loading", model=_MODEL_NAME)
        self._model = F5TTS.from_pretrained(_MODEL_NAME)

        # Lock HF to offline mode — model is now cached locally
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Preprocess reference audio once — reused for every sentence
        audio, sr = sf.read(str(self._ref_audio_path), dtype="float32")
        if sr != _SAMPLE_RATE:
            raise ValueError(
                f"voice_ref.wav must be 24kHz, got {sr}Hz. "
                "Resample with: ffmpeg -i voice_ref.wav -ar 24000 voice_ref_24k.wav"
            )
        audio_mx = mx.array(audio)
        rms = mx.sqrt(mx.mean(mx.square(audio_mx)))
        if rms < _TARGET_RMS:
            audio_mx = audio_mx * _TARGET_RMS / rms
        self._ref_audio = mx.expand_dims(audio_mx, axis=0)
        self._ref_audio_len = audio_mx.shape[0]
        # Mel frames for the reference audio — used as a floor for duration param
        self._ref_frames = int(self._ref_audio_len / 256) + 50

        # Store helpers
        self._mx = mx
        self._convert_char_to_pinyin = convert_char_to_pinyin

        log.info("tts_ready_f5_mlx", ref_duration_sec=round(self._ref_audio_len / _SAMPLE_RATE, 1))

    def unload(self) -> None:
        self._model = None
        self._ref_audio = None
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

        mx = self._mx
        text = _apply_emphasis(sentence.text, sentence.emphasis_words)
        speed = {"slow": 0.85, "fast": 1.15, "medium": 1.0}.get(sentence.pace, 1.0)

        log.debug("tts_generate", id=sentence.id, chars=len(text), speed=speed)

        # Prepend ref transcript — F5-TTS concatenates ref + generation internally
        full_text = self._convert_char_to_pinyin([self._ref_text + " " + text])

        duration = max(self._ref_frames + len(text) * 3, self._ref_frames + 100)
        wave, _ = self._model.sample(
            self._ref_audio,
            text=full_text,
            duration=duration,
            steps=8,
            method="rk4",
            speed=speed,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

        # Strip the reference audio portion from the output
        wave = wave[self._ref_audio_len:]
        mx.eval(wave)

        data = np.array(wave, dtype=np.float32)
        pre = _silence(sentence.pause_before_ms, _SAMPLE_RATE)
        post = _silence(sentence.pause_after_ms, _SAMPLE_RATE)
        padded = np.concatenate([pre, data, post])

        out_path = out_dir / f"{sentence.id}.wav"
        sf.write(str(out_path), padded, _SAMPLE_RATE)

        log.debug("tts_saved", path=str(out_path), duration_sec=round(len(padded) / _SAMPLE_RATE, 2))
        return out_path
