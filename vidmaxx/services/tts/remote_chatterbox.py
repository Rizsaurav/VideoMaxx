"""
Remote Chatterbox TTS — calls a self-hosted Chatterbox instance over HTTP.

The server exposes an OpenAI-compatible /v1/audio/speech endpoint and returns
raw WAV audio. Intended as a fallback when ElevenLabs credits are exhausted.

Server is expected at CHATTERBOX_URL (e.g. an ngrok tunnel to a GPU machine).

POST /v1/audio/speech
  Body:    {"text": "<sentence text>"}
  Returns: audio/wav

Per-sentence generation, same interface as the Kokoro TTSService.
Silence padding (pause_before_ms, pause_after_ms) is baked in at write time.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import structlog

from vidmaxx.models.sentence import Sentence

log = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(60.0)   # Chatterbox can be slow on first call (GPU warmup)


def _silence(ms: int, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(sample_rate * ms / 1000)), dtype=np.float32)


class RemoteChatterboxService:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_TIMEOUT, follow_redirects=True)
        log.info("remote_chatterbox_ready", url=self._base_url)

    def unload(self) -> None:
        self._client.close()
        log.info("remote_chatterbox_closed")

    def __enter__(self) -> "RemoteChatterboxService":
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    def generate(self, sentence: Sentence, out_dir: Path) -> Path:
        out_path = out_dir / f"{sentence.id}.wav"
        if out_path.exists():
            return out_path

        log.debug("remote_chatterbox_generate", id=sentence.id, chars=len(sentence.text))

        response = self._client.post(
            f"{self._base_url}/v1/audio/speech",
            json={"text": sentence.text},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio = audio.flatten()

        pre  = _silence(sentence.pause_before_ms, sample_rate)
        post = _silence(sentence.pause_after_ms,  sample_rate)
        padded = np.concatenate([pre, audio, post])

        sf.write(str(out_path), padded, sample_rate)
        log.debug(
            "remote_chatterbox_saved",
            path=str(out_path),
            duration_sec=round(len(padded) / sample_rate, 2),
        )
        return out_path

    def generate_chapter(
        self,
        sentences: list[Sentence],
        chapter_id: str,
        out_dir: Path,
    ) -> list[Path]:
        """Chapter-level wrapper — calls generate() per sentence sequentially."""
        paths = []
        for s in sentences:
            paths.append(self.generate(s, out_dir))
        log.info("remote_chatterbox_chapter_done", chapter=chapter_id, sentences=len(sentences))
        return paths
