"""
Stage 5 — Alignment

Reads:   paths.narration_wav   (full concatenated narration)
         paths.script           (script.json, for sentence list + emphasis words)
         paths.audio_dir        (individual sentence WAVs, for exact durations)
Writes:  paths.alignment_json   (list of CaptionSegment as JSON)
Advances: ALIGNMENT → TIMELINE

AlignmentService is blocking (no async API in whisperx) — run in a thread
so the event loop stays responsive. Model is unloaded on exit.
"""

import asyncio
import json

import structlog

from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Script
from vidmaxx.models.timeline import CaptionSegment
from vidmaxx.services.alignment.whisperx import AlignmentService
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
) -> list[CaptionSegment]:
    paths = state_mgr.paths(slug)
    script = Script.model_validate_json(paths.script.read_text())

    if not paths.narration_wav.exists():
        raise FileNotFoundError(
            f"Narration WAV not found at {paths.narration_wav}. "
            "Run Stage 4 (TTS) first."
        )

    log.info("stage_alignment_start", slug=slug, sentences=len(script.sentences))

    with state_mgr.run_stage(slug, PipelineStage.ALIGNMENT) as paths:
        if paths.alignment_json.exists():
            log.info("stage_alignment_skip_already_complete", slug=slug)
            segments = [
                CaptionSegment.model_validate(item)
                for item in json.loads(paths.alignment_json.read_text())
            ]
            return segments

        segments = await asyncio.to_thread(
            _run_sync,
            paths.narration_wav,
            script,
            paths.audio_dir,
            settings.whisperx_model,
            settings.tts_device,
        )

        _write_alignment(segments, paths.alignment_json)
        log.info(
            "stage_alignment_done",
            slug=slug,
            segments=len(segments),
            words=sum(len(s.words) for s in segments),
        )

    return segments


def _run_sync(narration_path, script, audio_dir, model_name, device) -> list[CaptionSegment]:
    with AlignmentService(
        device=device,
        model_name=model_name,
    ) as svc:
        return svc.run(narration_path, script, audio_dir)


def _write_alignment(segments: list[CaptionSegment], path) -> None:
    data = [seg.model_dump() for seg in segments]
    path.write_text(json.dumps(data, indent=2))
