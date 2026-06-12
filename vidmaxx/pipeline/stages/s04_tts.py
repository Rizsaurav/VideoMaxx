"""
Stage 4 — TTS

Reads:   paths.script                (script.json)
Writes:  paths.audio_dir/{sentence_id}.wav   (one per sentence, split from chapter audio)
         paths.narration_wav                  (full concatenated narration)
Advances: TTS → ALIGNMENT

Sentences are grouped by chapter. One ElevenLabs call per chapter (9 total)
so the model sees full chapter text and delivers natural prosodic flow.
The chapter audio is split proportionally into per-sentence WAVs for alignment.
"""

from __future__ import annotations

import asyncio
from itertools import groupby
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Script
from vidmaxx.models.sentence import Sentence
from vidmaxx.services.tts.audio_fx import apply_studio_chain
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)


def _make_tts_service(settings: Settings):
    if settings.tts_backend == "chatterbox":
        if not settings.chatterbox_url:
            raise ValueError("CHATTERBOX_URL must be set when TTS_BACKEND=chatterbox")
        from vidmaxx.services.tts.remote_chatterbox import RemoteChatterboxService
        return RemoteChatterboxService(settings.chatterbox_url)
    if settings.tts_backend == "elevenlabs":
        from vidmaxx.services.tts.elevenlabs_tts import ElevenLabsTTSService
        return ElevenLabsTTSService(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            voice_name=settings.elevenlabs_voice_name,
        )
    from vidmaxx.services.tts.mlx_audio_tts import TTSService
    return TTSService()


def _make_chatterbox_fallback(settings: Settings):
    if not settings.chatterbox_url:
        raise RuntimeError(
            "ElevenLabs quota exhausted and CHATTERBOX_URL is not set. "
            "Add CHATTERBOX_URL to .env to enable the remote Chatterbox fallback."
        )
    from vidmaxx.services.tts.remote_chatterbox import RemoteChatterboxService
    log.warning("tts_elevenlabs_quota_fallback_to_chatterbox", url=settings.chatterbox_url)
    return RemoteChatterboxService(settings.chatterbox_url)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("quota", "credit", "429", "insufficient", "billing", "limit exceeded"))


def _chapter_key(s: Sentence) -> str:
    return s.id.split("_s")[0]  # "ch03_s12" → "ch03"


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
) -> Path:
    paths = state_mgr.paths(slug)
    script = Script.model_validate_json(paths.script.read_text())
    sentences = script.sentences
    log.info("stage_tts_start", slug=slug, sentences=len(sentences), backend=settings.tts_backend)

    with state_mgr.run_stage(slug, PipelineStage.TTS) as paths:
        paths.audio_dir.mkdir(parents=True, exist_ok=True)

        if paths.narration_wav.exists():
            log.info("stage_tts_skip_already_complete", slug=slug)
            return paths.narration_wav

        sentence_paths = await _generate_all(sentences, paths.audio_dir, settings)

        narration_path = await _concatenate(sentence_paths, paths.narration_wav)
        await asyncio.to_thread(apply_studio_chain, narration_path)
        log.info("stage_tts_done", slug=slug, narration=str(narration_path))

    return narration_path


async def _generate_all(sentences: list[Sentence], audio_dir: Path, settings: Settings) -> list[Path]:
    tts = await asyncio.to_thread(_make_tts_service, settings)
    fallback: object = None  # set to RemoteChatterboxService on first quota error

    chapters = [
        (ch_id, list(ch_sentences))
        for ch_id, ch_sentences in groupby(sentences, key=_chapter_key)
    ]

    all_paths: list[Path] = []
    try:
        for i, (chapter_id, ch_sentences) in enumerate(chapters):
            active = fallback or tts
            use_chapter_api = settings.tts_backend == "elevenlabs" and fallback is None

            try:
                if use_chapter_api:
                    paths = await asyncio.to_thread(
                        active.generate_chapter, ch_sentences, chapter_id, audio_dir
                    )
                else:
                    paths = []
                    for s in ch_sentences:
                        p = await asyncio.to_thread(active.generate, s, audio_dir)
                        paths.append(p)
            except Exception as exc:
                if _is_quota_error(exc) and fallback is None:
                    fallback = await asyncio.to_thread(_make_chatterbox_fallback, settings)
                    # Retry this chapter with the fallback
                    paths = await asyncio.to_thread(
                        fallback.generate_chapter, ch_sentences, chapter_id, audio_dir
                    )
                else:
                    raise

            all_paths.extend(paths)
            log.info("tts_chapter_done", chapter=chapter_id, index=i + 1, total=len(chapters))
    finally:
        await asyncio.to_thread(tts.unload)
        if fallback is not None:
            await asyncio.to_thread(fallback.unload)

    return all_paths


async def _concatenate(sentence_paths: list[Path], out_path: Path) -> Path:
    return await asyncio.to_thread(_concat_sync, sentence_paths, out_path)


def _concat_sync(sentence_paths: list[Path], out_path: Path) -> Path:
    chunks: list[np.ndarray] = []
    sample_rate: int | None = None

    log.debug("concat_inputs", count=len(sentence_paths))
    for p in sentence_paths:
        data, sr = sf.read(str(p), dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        # Normalize to mono — guard against stereo files from old runs
        if data.ndim == 2:
            data = data[:, 0]
        chunks.append(data)

    full = np.concatenate(chunks, axis=0)
    sf.write(str(out_path), full, sample_rate)
    log.info(
        "tts_concat_done",
        sentences=len(sentence_paths),
        duration_sec=round(len(full) / sample_rate, 1),
    )
    return out_path
