"""
ElevenLabs TTS service — chapter-level generation for natural prosodic flow.

One API call per chapter (9 total). ElevenLabs sees the full chapter text and
delivers it with natural rhythm, pausing, and sentence-to-sentence flow.
The resulting audio is split proportionally by character count into per-sentence
WAV files so the alignment service gets the individual durations it needs.

Emotion tags (eleven_v3) are injected per sentence within the chapter text.
Emphasis words are uppercased so v3 stresses them naturally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vidmaxx.models.sentence import Sentence

log = structlog.get_logger(__name__)

_SAMPLE_RATE = 24_000
_MODEL_ID = "eleven_v3"
_OUTPUT_FORMAT = "pcm_24000"  # raw 16-bit signed PCM at 24kHz

_REGISTER_EMOTION: dict[str, str] = {
    "strategic_hook":       "suspense",
    "mechanism_flow":       "neutral",
    "incongruity_drop":     "confused",
    "dependency_link":      "neutral",
    "human_anchor":         "friendly",
    "causal_collapse":      "worry",
    "institutional_facade": "serious",
    "shadow_truth":         "serious",
    "synthesis":            "reflective",
    "clinical_shock":       "suspense",
    "false_normal":         "friendly",
    "infrastructure":       "neutral",
    "systemic_flow":        "neutral",
    "personal_consequence": "worry",
    "twist":                "suspense",
    "exit":                 "reflective",
}


class ElevenLabsTTSService:
    def __init__(self, api_key: str, voice_id: str = "", voice_name: str = "George") -> None:
        from elevenlabs.client import ElevenLabs

        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id or self._resolve_voice_id(voice_name)
        log.info("elevenlabs_tts_ready", voice_id=self._voice_id)

    def _resolve_voice_id(self, name: str) -> str:
        voices = self._client.voices.get_all()
        for v in voices.voices:
            if name.lower() in v.name.lower():
                log.info("elevenlabs_voice_resolved", name=v.name, voice_id=v.voice_id)
                return v.voice_id
        available = [v.name for v in voices.voices[:10]]
        raise ValueError(
            f"ElevenLabs voice '{name}' not found. "
            f"First 10 available: {available}. "
            "Set ELEVENLABS_VOICE_ID in .env to skip name search."
        )

    def unload(self) -> None:
        pass

    def __enter__(self) -> "ElevenLabsTTSService":
        return self

    def __exit__(self, *_) -> None:
        pass

    def generate_chapter(self, sentences: list[Sentence], chapter_id: str, out_dir: Path) -> list[Path]:
        """
        Generate audio for a full chapter in one API call, then split into
        per-sentence WAV files proportionally by character count.

        Returns the list of per-sentence WAV paths in sentence order.
        """
        out_paths = [out_dir / f"{s.id}.wav" for s in sentences]

        # Cache hit — all sentence WAVs already exist
        if all(p.exists() for p in out_paths):
            log.debug("elevenlabs_chapter_cache_hit", chapter=chapter_id)
            return out_paths

        # Build chapter text: one emotion-tagged segment per sentence
        parts = []
        for s in sentences:
            emotion = _REGISTER_EMOTION.get(s.emotional_register or "", "neutral")
            text = s.text
            for word in s.emphasis_words:
                text = text.replace(word, word.upper())
            parts.append(f'<emotion name="{emotion}">{text}</emotion>')

        chapter_text = " ".join(parts)
        char_counts = [len(s.text) for s in sentences]
        total_chars = max(sum(char_counts), 1)

        log.info(
            "elevenlabs_chapter_generate",
            chapter=chapter_id,
            sentences=len(sentences),
            chars=total_chars,
        )

        try:
            audio_bytes = b"".join(
                self._client.text_to_speech.convert(
                    text=chapter_text,
                    voice_id=self._voice_id,
                    model_id=_MODEL_ID,
                    output_format=_OUTPUT_FORMAT,
                )
            )
        except Exception as exc:
            log.error("elevenlabs_chapter_error", chapter=chapter_id, error=str(exc))
            raise

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        total_samples = len(audio)

        # Split proportionally by character count
        cursor = 0
        for i, (s, path, chars) in enumerate(zip(sentences, out_paths, char_counts)):
            if i == len(sentences) - 1:
                chunk = audio[cursor:]  # last sentence gets the remainder
            else:
                n = int((chars / total_chars) * total_samples)
                chunk = audio[cursor:cursor + n]
                cursor += n

            sf.write(str(path), chunk, _SAMPLE_RATE)
            log.debug("elevenlabs_sentence_saved", id=s.id, duration_sec=round(len(chunk) / _SAMPLE_RATE, 2))

        return out_paths
