"""
Stage 7 — Render

Reads:   paths.timeline_json     (timeline.json)
         paths.narration_wav      (full concatenated narration)
         paths.assets_dir/        (per-sentence asset files)
         paths.audio_dir/         (per-sentence WAVs + full_narration.wav)
Writes:  paths.out_dir/chapters/  (per-chapter MP4s)
         paths.out_dir/captions/  (per-chapter .ass files)
         paths.final_video        (out/video.mp4 — concat of all chapters)
Advances: RENDER → SHORTS

Chapter renders run in parallel via ProcessPoolExecutor (RENDER_MAX_WORKERS).
The concat step uses the FFmpeg concat demuxer (stream-copy, no re-encode).
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import structlog

from vidmaxx.config.constants import (
    PATTERN_INTERRUPT_INDICES,
    RENDER_MAX_WORKERS,
    SCENE_LABELS,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.timeline import Timeline
from vidmaxx.render.captions import write_ass
from vidmaxx.render.chapter_renderer import render_chapter
from vidmaxx.render.concat import concat_chapters
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

_MIN_DURATION_SEC = 900   # 15 min
_MAX_DURATION_SEC = 1200  # 20 min


def _validate_timeline(timeline: Timeline) -> None:
    """Raise ValueError with a specific message if any pre-render check fails."""
    errors: list[str] = []

    # Check 1: video opens on the first master beat
    if timeline.chapters:
        first = timeline.chapters[0]
        if first.index != 0:
            errors.append(
                f"First chapter index is {first.index}, expected 0 ({SCENE_LABELS[0]}). "
                "Video does not open on the first master beat."
            )

    # Check 2: if the structure declares audio pattern-interrupt chapters,
    # ensure they exist. The sample-derived master beat map has none; its
    # ruptures are rhetorical, not full-chapter music-duck events.
    if PATTERN_INTERRUPT_INDICES:
        pi_chapters = [ch for ch in timeline.chapters if ch.index in PATTERN_INTERRUPT_INDICES]
        if len(pi_chapters) < len(PATTERN_INTERRUPT_INDICES):
            errors.append(
                f"Only {len(pi_chapters)} pattern interrupt chapter(s) found "
                f"(expected chapters at indices {sorted(PATTERN_INTERRUPT_INDICES)}). "
                "Check that s02 produced all 9 scenes."
            )

    # Check 3: total duration in 15-20 minute window
    dur = timeline.total_duration_sec
    if not (_MIN_DURATION_SEC <= dur <= _MAX_DURATION_SEC):
        errors.append(
            f"Total duration {dur:.0f}s is outside the 15-20 min window "
            f"({_MIN_DURATION_SEC}-{_MAX_DURATION_SEC}s). "
            "Check TTS output or script length."
        )

    if errors:
        raise ValueError(
            "Pre-render validation failed — halting before FFmpeg:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
) -> Path:
    paths = state_mgr.paths(slug)
    timeline = Timeline.model_validate_json(paths.timeline_json.read_text())

    log.info(
        "stage_render_start",
        slug=slug,
        chapters=len(timeline.chapters),
        duration_sec=round(timeline.total_duration_sec, 1),
    )

    with state_mgr.run_stage(slug, PipelineStage.RENDER) as paths:
        _validate_timeline(timeline)
        log.info("stage_render_validation_passed", slug=slug)

        chapters_dir = paths.out_dir / "chapters"
        captions_dir = paths.out_dir / "captions"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        # Write ASS caption files (fast — pure Python, no I/O bottleneck).
        ass_paths: dict[int, Path] = {}
        for chapter in timeline.chapters:
            ass_path = captions_dir / f"ch{chapter.index:02d}.ass"
            write_ass(chapter.caption_segments, ass_path, chapter.start_sec)
            ass_paths[chapter.index] = ass_path

        # Render chapters in parallel via ProcessPoolExecutor.
        chapter_mp4s: dict[int, Path] = {}
        futures = {}

        with ProcessPoolExecutor(max_workers=RENDER_MAX_WORKERS) as pool:
            for chapter in timeline.chapters:
                out_mp4 = chapters_dir / f"ch{chapter.index:02d}.mp4"
                future = pool.submit(
                    render_chapter,
                    chapter.model_dump(),
                    str(paths.narration_wav),
                    str(out_mp4),
                    str(ass_paths[chapter.index]),
                )
                futures[future] = chapter.index

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_path = future.result()
                    chapter_mp4s[idx] = Path(result_path)
                    log.info("render_chapter_complete", chapter=idx)
                except Exception as exc:
                    raise RuntimeError(
                        f"Chapter {idx} render failed: {exc}"
                    ) from exc

        # Concat in chapter order.
        ordered = [chapter_mp4s[i] for i in sorted(chapter_mp4s)]
        final = await asyncio.to_thread(concat_chapters, ordered, paths.final_video)

        log.info(
            "stage_render_done",
            slug=slug,
            video=str(final),
            size_mb=round(final.stat().st_size / 1e6, 1),
        )

    return paths.final_video
