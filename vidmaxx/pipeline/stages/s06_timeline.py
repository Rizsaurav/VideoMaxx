"""
Stage 6 — Timeline compile

Reads:   paths.script               (script.json)
         paths.alignment_json       (alignment.json)
         paths.audio_dir/           (per-sentence WAVs for exact durations)
         paths.assets_dir/          (per-sentence asset files)
         paths.vlm_manifest_json    (vlm_manifest.json — optional, from s03b)
         paths.narration_wav
         settings.music_dir/        (style_pack music beds)
         settings.sfx_dir/          (style_pack SFX)
Writes:  paths.timeline_json        (timeline.json)
Advances: TIMELINE → RENDER

Pure Python — no external services, no async I/O. Completes in < 1s.

Design notes:
  - Sentence durations come from actual WAV files (soundfile.info), not
    script estimates. Everything downstream is exact.
  - If vlm_manifest.json exists (s03b ran), asset assignments and per-sentence
    mood override the file-system defaults. Mood drives motion effect selection
    instead of the generic energy-keyed cycle.
  - Without the manifest, behaviour is identical to pre-s03b.
  - SFX and music are omitted gracefully when style_pack files don't exist yet.
  - caption_segments pass through directly from alignment.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import structlog

from vidmaxx.config.constants import (
    MUSIC_DEFAULT_VOLUME,
    MUSIC_DUCK_VOLUME,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.asset import AssetType
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Chapter, Script
from vidmaxx.models.sentence import Sentence
from vidmaxx.models.timeline import (
    ChapterSlice,
    CaptionSegment,
    MotionEffect,
    MusicBed,
    MusicClip,
    NarrationClip,
    SFXClip,
    Timeline,
    VideoClip,
)
from vidmaxx.state.project_state import ProjectStateManager
from vidmaxx.utils.audio import wav_durations_sec
from vidmaxx.utils.timing import sentence_start_times

log = structlog.get_logger(__name__)

# Motion effect sequences cycled per sentence, indexed by chapter energy
_MOTION_CYCLE: dict[str, list[MotionEffect]] = {
    "low":    [MotionEffect.KEN_BURNS_IN, MotionEffect.KEN_BURNS_OUT],
    "medium": [MotionEffect.KEN_BURNS_IN, MotionEffect.PAN_RIGHT,
               MotionEffect.KEN_BURNS_OUT, MotionEffect.PAN_LEFT],
    "high":   [MotionEffect.PAN_LEFT, MotionEffect.PAN_RIGHT,
               MotionEffect.KEN_BURNS_IN, MotionEffect.PAN_RIGHT],
}

# Chapter energy → music bed (first/last chapter override applied separately)
_BED_FOR_ENERGY: dict[str, MusicBed] = {
    "low":    MusicBed.CHILL,
    "medium": MusicBed.BUILD,
    "high":   MusicBed.TENSE,
}

# VLM mood → motion effect.  Used when vlm_manifest.json exists so each clip
# gets a motion that matches what Gemini actually sees, not a generic energy cycle.
_MOOD_MOTION: dict[str, MotionEffect] = {
    "clinical":      MotionEffect.KEN_BURNS_IN,   # slow reveal — cold, measured
    "ominous":       MotionEffect.KEN_BURNS_IN,   # creeping zoom — dread builds
    "institutional": MotionEffect.KEN_BURNS_IN,   # steady, formal
    "alarming":      MotionEffect.PAN_RIGHT,       # lateral movement — urgency
    "warm":          MotionEffect.KEN_BURNS_OUT,  # receding — humanising
    "neutral":       MotionEffect.KEN_BURNS_IN,   # safe default
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
) -> Timeline:
    paths = state_mgr.paths(slug)
    script = Script.model_validate_json(paths.script.read_text())
    alignment = _load_alignment(paths.alignment_json)

    log.info("stage_timeline_start", slug=slug, sentences=len(script.sentences))

    with state_mgr.run_stage(slug, PipelineStage.TIMELINE) as paths:
        timeline = _compile(script, alignment, paths, settings)
        paths.timeline_json.write_text(
            timeline.model_dump_json(indent=2)
        )
        log.info(
            "stage_timeline_done",
            slug=slug,
            chapters=len(timeline.chapters),
            duration_sec=round(timeline.total_duration_sec, 1),
        )

    return timeline


# ---------------------------------------------------------------------------
# Duration fallback — derive from alignment when WAVs are unavailable
# ---------------------------------------------------------------------------

def _durations_from_alignment(
    sentences: list[Sentence],
    alignment: list[CaptionSegment],
) -> list[float]:
    seg_map = {s.sentence_id: s for s in alignment}
    durations: list[float] = []
    for s in sentences:
        seg = seg_map.get(s.id)
        if seg and seg.words:
            pause_before = s.pause_before_ms / 1000.0
            pause_after = s.pause_after_ms / 1000.0
            speech_dur = seg.words[-1].end_sec - seg.words[0].start_sec
            durations.append(max(speech_dur + pause_before + pause_after, 0.1))
        else:
            durations.append(3.0)  # fallback for sentences with no alignment data
    return durations


# ---------------------------------------------------------------------------
# Core compiler
# ---------------------------------------------------------------------------

def _compile(
    script: Script,
    alignment: list[CaptionSegment],
    paths,
    settings: Settings,
) -> Timeline:
    sentences = script.sentences

    # Step 1 — exact sentence durations and absolute start times
    # Prefer WAV files (most accurate); fall back to alignment.json when audio
    # dir has been deleted (safe because alignment was derived from those WAVs).
    wav_paths = [paths.audio_dir / f"{s.id}.wav" for s in sentences]
    if all(p.exists() for p in wav_paths):
        durations = wav_durations_sec(wav_paths)
    else:
        log.warning("timeline_audio_missing_using_alignment")
        durations = _durations_from_alignment(sentences, alignment)
    starts = sentence_start_times(durations)
    total_duration = sum(durations)

    # Step 2 — fast lookup maps
    caption_map: dict[str, CaptionSegment] = {seg.sentence_id: seg for seg in alignment}
    asset_map: dict[str, tuple[Path, AssetType]] = _build_asset_map(
        sentences, paths.assets_dir
    )
    mood_map: dict[str, str] = _load_mood_map(paths.vlm_manifest_json, asset_map)
    sfx_files = _scan_sfx(settings.sfx_dir)

    # Step 3 — index sentences by their sentence_id for O(1) access
    sentence_map = {s.id: s for s in sentences}
    sentence_starts = {s.id: starts[i] for i, s in enumerate(sentences)}
    sentence_durations = {s.id: durations[i] for i, s in enumerate(sentences)}

    # Step 4 — build one ChapterSlice per chapter
    chapter_slices: list[ChapterSlice] = []
    n_chapters = len(script.chapters)

    for chapter in script.chapters:
        if not chapter.sentences:
            continue

        ch_start = sentence_starts[chapter.sentences[0].id]
        ch_end = (
            sentence_starts[chapter.sentences[-1].id]
            + sentence_durations[chapter.sentences[-1].id]
        )

        video_clips = _build_video_clips(
            chapter, sentence_starts, sentence_durations, asset_map, mood_map
        )
        narration_clips = _build_narration_clips(
            chapter, sentence_starts, sentence_durations, paths.audio_dir
        )
        music_clips = _build_music_clips(
            chapter, ch_start, ch_end - ch_start,
            chapter.index, n_chapters, settings.music_dir
        )
        sfx_clips = _build_sfx_clips(ch_start, chapter.index, sfx_files)
        caption_segs = [
            caption_map[s.id]
            for s in chapter.sentences
            if s.id in caption_map
        ]

        # Drive music_duck from emotional_register on sentences, not chapter index.
        # Pattern interrupt scenes set register "pattern_interrupt_1" or "_2" on
        # their sentences; any chapter containing those sentences gets silence.
        _PI_REGISTERS = frozenset({"pattern_interrupt_1", "pattern_interrupt_2"})
        has_pattern_interrupt = any(
            getattr(s, "emotional_register", "") in _PI_REGISTERS
            for s in chapter.sentences
        )

        chapter_slices.append(ChapterSlice(
            index=chapter.index,
            start_sec=ch_start,
            end_sec=ch_end,
            video_clips=video_clips,
            narration_clips=narration_clips,
            music_clips=music_clips,
            sfx_clips=sfx_clips,
            caption_segments=caption_segs,
            music_duck=has_pattern_interrupt,
        ))

    return Timeline(
        topic=script.topic,
        slug=script.slug,
        total_duration_sec=total_duration,
        chapters=chapter_slices,
        narration_path=paths.narration_wav,
    )


# ---------------------------------------------------------------------------
# Video clips
# ---------------------------------------------------------------------------

def _build_video_clips(
    chapter: Chapter,
    sentence_starts: dict[str, float],
    sentence_durations: dict[str, float],
    asset_map: dict[str, tuple[Path, AssetType]],
    mood_map: dict[str, str] | None = None,
) -> list[VideoClip]:
    motion_cycle = itertools.cycle(_MOTION_CYCLE.get(chapter.energy, _MOTION_CYCLE["medium"]))
    clips: list[VideoClip] = []

    for sentence in chapter.sentences:
        if sentence.id not in asset_map:
            log.warning("timeline_no_asset", sentence_id=sentence.id)
            continue
        asset_path, _ = asset_map[sentence.id]

        if mood_map and sentence.id in mood_map:
            motion = _MOOD_MOTION.get(mood_map[sentence.id], next(motion_cycle))
        else:
            motion = next(motion_cycle)

        clips.append(VideoClip(
            sentence_id=sentence.id,
            asset_path=asset_path,
            start_sec=sentence_starts[sentence.id],
            duration_sec=sentence_durations[sentence.id],
            motion=motion,
            source_start_sec=0.0,
        ))

    return clips


# ---------------------------------------------------------------------------
# Narration clips
# ---------------------------------------------------------------------------

def _build_narration_clips(
    chapter: Chapter,
    sentence_starts: dict[str, float],
    sentence_durations: dict[str, float],
    audio_dir: Path,
) -> list[NarrationClip]:
    return [
        NarrationClip(
            sentence_id=s.id,
            audio_path=audio_dir / f"{s.id}.wav",
            start_sec=sentence_starts[s.id],
            duration_sec=sentence_durations[s.id],
        )
        for s in chapter.sentences
    ]


# ---------------------------------------------------------------------------
# Music clips
# ---------------------------------------------------------------------------

def _music_bed(chapter_index: int, n_chapters: int, energy: str) -> MusicBed:
    if chapter_index == 0:
        return MusicBed.INTRO
    if chapter_index == n_chapters - 1:
        return MusicBed.OUTRO
    return _BED_FOR_ENERGY.get(energy, MusicBed.BUILD)


def _build_music_clips(
    chapter: Chapter,
    start_sec: float,
    duration_sec: float,
    chapter_index: int,
    n_chapters: int,
    music_dir: Path,
) -> list[MusicClip]:
    bed = _music_bed(chapter_index, n_chapters, chapter.energy)
    music_path = music_dir / f"{bed.value}.wav"
    if not music_path.exists():
        log.debug("timeline_music_missing", bed=bed.value, path=str(music_path))
        return []
    return [MusicClip(
        bed=bed,
        audio_path=music_path,
        start_sec=start_sec,
        duration_sec=duration_sec,
        volume=MUSIC_DEFAULT_VOLUME,
        duck_volume=MUSIC_DUCK_VOLUME,
    )]


# ---------------------------------------------------------------------------
# SFX clips
# ---------------------------------------------------------------------------

def _scan_sfx(sfx_dir: Path) -> list[Path]:
    if not sfx_dir.exists():
        return []
    return sorted(sfx_dir.glob("*.wav"))


def _pick_sfx(sfx_files: list[Path], preferred_prefix: str) -> Path | None:
    """Pick the first SFX whose filename starts with preferred_prefix, else any."""
    for f in sfx_files:
        if f.stem.startswith(preferred_prefix):
            return f
    return sfx_files[0] if sfx_files else None


def _build_sfx_clips(
    chapter_start_sec: float,
    chapter_index: int,
    sfx_files: list[Path],
) -> list[SFXClip]:
    if not sfx_files or chapter_index == 0:
        # No SFX at the very start of the video
        return []
    sfx = _pick_sfx(sfx_files, "whoosh") or _pick_sfx(sfx_files, "transition")
    if sfx is None:
        return []
    # Place 0.1s before the chapter cut so it feels like it leads the transition
    return [SFXClip(
        name=sfx.stem,
        audio_path=sfx,
        start_sec=max(0.0, chapter_start_sec - 0.1),
    )]


# ---------------------------------------------------------------------------
# VLM manifest loader
# ---------------------------------------------------------------------------

def _load_mood_map(
    manifest_path: Path,
    asset_map: dict[str, tuple[Path, AssetType]],
) -> dict[str, str]:
    """Return sentence_id → mood from vlm_manifest.json.

    Also overwrites asset_map entries in-place where the manifest reassigned
    an asset to a different sentence.  Falls back silently if the manifest
    is absent or malformed — no manifest means no VLM pass ran.
    """
    if not manifest_path.exists():
        return {}

    try:
        entries = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("timeline_vlm_manifest_unreadable", path=str(manifest_path))
        return {}

    mood_map: dict[str, str] = {}
    for entry in entries:
        sid        = entry.get("sentence_id")
        asset_path = entry.get("asset_path")
        mood       = entry.get("mood", "neutral")
        if not sid:
            continue
        mood_map[sid] = mood
        if asset_path:
            p = Path(asset_path)
            if p.exists():
                ext   = p.suffix.lower()
                atype = AssetType.VIDEO if ext in _VIDEO_EXTS else AssetType.IMAGE
                asset_map[sid] = (p, atype)

    reassigned = sum(1 for e in entries if e.get("reassigned"))
    log.info(
        "timeline_vlm_manifest_loaded",
        entries=len(entries),
        reassigned=reassigned,
    )
    return mood_map


# ---------------------------------------------------------------------------
# Asset map
# ---------------------------------------------------------------------------

_VIDEO_EXTS = {".mp4", ".mov", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _build_asset_map(
    sentences: list[Sentence],
    assets_dir: Path,
) -> dict[str, tuple[Path, AssetType]]:
    # Pass 1: find all sentences that have a downloaded asset
    owned: dict[str, tuple[Path, AssetType]] = {}
    for s in sentences:
        matches = [
            f for f in assets_dir.glob(f"{s.id}.*")
            if f.suffix.lower() in _VIDEO_EXTS | _IMAGE_EXTS
        ]
        if matches:
            path = matches[0]
            atype = AssetType.VIDEO if path.suffix.lower() in _VIDEO_EXTS else AssetType.IMAGE
            owned[s.id] = (path, atype)

    if not owned:
        return {}

    # Pass 2: for sentences without an asset, cycle through the pool so
    # no single clip dominates the video.
    pool = list(owned.values())
    pool_cycle = itertools.cycle(pool)
    result: dict[str, tuple[Path, AssetType]] = {}
    for s in sentences:
        if s.id in owned:
            result[s.id] = owned[s.id]
        else:
            result[s.id] = next(pool_cycle)

    return result


# ---------------------------------------------------------------------------
# Alignment loader
# ---------------------------------------------------------------------------

def _load_alignment(path: Path) -> list[CaptionSegment]:
    if not path.exists():
        log.warning("timeline_no_alignment", path=str(path))
        return []
    data = json.loads(path.read_text())
    return [CaptionSegment.model_validate(item) for item in data]
