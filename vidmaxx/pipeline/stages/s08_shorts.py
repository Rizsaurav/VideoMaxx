"""
Stage 8 — Shorts (two-step: SHORTS_SELECT → SHORTS_RENDER)

SHORTS_SELECT (run_select)
--------------------------
Reads:   paths.timeline_json    (timeline.json — chapter boundaries + music beds)
         paths.script           (script.json — chapter energy labels)
         paths.alignment_json   (alignment.json — word-level timestamps)
Writes:  paths.shorts_dir/candidates.json
Advances: SHORTS_SELECT → SHORTS_RENDER
Prints top 5 scored candidates to stdout. Pipeline pauses here — the user
must run `vidmaxx shorts <slug> --pick 1,3` to continue.

SHORTS_RENDER (run_render)
--------------------------
Reads:   paths.shorts_dir/candidates.json
         paths.final_video
         paths.alignment_json
Writes:  paths.shorts_dir/short_N.mp4
         paths.shorts_dir/captions/short_N.ass
Advances: SHORTS_RENDER → PUBLISH

Scoring
-------
Three components combined into a weighted total:
  emphasis_density  — emphasis words per second in the window            (weight 0.40)
  chapter_energy    — 1.0 high / 0.5 medium / 0.2 low                  (weight 0.30)
  coherence         — starts at sentence boundary + no leading connector  (weight 0.30)

Emphasis density is normalized across all candidates before weighting.

Render spec
-----------
  Source: out/video.mp4 (1920×1080, already rendered)
  Crop:   center 608×1080 column → 9:16 ratio
  Scale:  1080×1920
  Captions: ASS, 96 pt, PlayRes 1080×1920, word-level (from alignment.json)
  Hook overlay: drawtext for first 1.5 s, text = first ~60 chars of narration
  Encoder: h264_videotoolbox at 6 Mbps
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from vidmaxx.config.constants import (
    AUDIO_BITRATE,
    AUDIO_ENCODER,
    RENDER_MAX_WORKERS,
    SHORTS_COUNT,
    SHORTS_HEIGHT,
    SHORTS_MAX_DURATION_SEC,
    SHORTS_MIN_DURATION_SEC,
    SHORTS_WIDTH,
    VIDEO_ENCODER,
    VIDEO_FPS,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Script
from vidmaxx.models.timeline import CaptionSegment, MusicBed, Timeline
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

# 9:16 center crop from 1920×1080.
# 9/16 × 1080 = 607.5 → 608 (nearest even).  x_offset = (1920 − 608) / 2 = 656.
_CROP_W = 608
_CROP_H = 1080
_CROP_X = (1920 - _CROP_W) // 2

_HOOK_DURATION_SEC = 1.5

_TOP_N = 5  # candidates to print for user review

_CONNECTORS = frozenset({
    "and", "but", "so", "yet", "or", "nor",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "additionally", "also", "meanwhile", "then", "because", "since",
    "although", "though", "while", "whereas", "nevertheless",
})

_ENERGY_SCORE: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.2,
}

_BED_ENERGY: dict[MusicBed, str] = {
    MusicBed.TENSE: "high",
    MusicBed.BUILD: "high",
    MusicBed.INTRO: "medium",
    MusicBed.OUTRO: "medium",
    MusicBed.CHILL: "low",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    rank: int
    start_sec: float
    end_sec: float
    duration_sec: float
    chapter_index: int
    chapter_energy: str
    emphasis_density: float    # raw: emphasis words per second
    energy_score: float        # 0.2 / 0.5 / 1.0
    coherence_score: float     # 0.0 – 1.0
    total_score: float         # weighted composite (post-normalization)
    narration_text: str        # first ~300 chars of window narration


def _candidates_path(paths) -> Path:
    return paths.shorts_dir / "candidates.json"


def _save_candidates(candidates: list[Candidate], paths) -> None:
    paths.shorts_dir.mkdir(parents=True, exist_ok=True)
    _candidates_path(paths).write_text(
        json.dumps([asdict(c) for c in candidates], indent=2),
        encoding="utf-8",
    )


def _load_candidates(paths) -> list[Candidate]:
    data = json.loads(_candidates_path(paths).read_text())
    return [Candidate(**d) for d in data]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _chapter_energy_for_window(
    mid: float,
    timeline: Timeline,
    script: Script,
) -> tuple[int, str]:
    """Return (chapter_index, energy_str) for the chapter containing *mid*."""
    for ch_slice in timeline.chapters:
        if ch_slice.start_sec <= mid < ch_slice.end_sec:
            # Look up energy from script (ChapterSlice doesn't store it).
            for ch in script.chapters:
                if ch.index == ch_slice.index:
                    return ch.index, ch.energy
            # Fallback: derive from music bed if script lookup missed.
            if ch_slice.music_clips:
                bed = ch_slice.music_clips[0].bed
                return ch_slice.index, _BED_ENERGY.get(bed, "medium")
            return ch_slice.index, "medium"
    # Window midpoint past last chapter — use last chapter.
    last = timeline.chapters[-1]
    return last.index, "medium"


def _coherence_score(
    start: float,
    end: float,
    sentence_starts: list[float],
    all_words: list[tuple[float, float, bool, str]],
) -> float:
    """Score how self-contained this window feels as a standalone clip.

    Components (all additive, capped at 1.0):
      0.6 — window starts within 0.15 s of a sentence boundary
      0.2 — first word is not a discourse connector
      0.2 — last word in window ends with terminal punctuation
    """
    score = 0.0

    # Sentence-boundary alignment.
    if any(abs(start - ss) <= 0.15 for ss in sentence_starts):
        score += 0.6

    # First-word connector check.
    first_words = [w for w in all_words if w[0] >= start and w[0] < start + 1.5]
    if first_words:
        first_txt = first_words[0][3].strip("\"'").lower().rstrip(".,!?;:")
        if first_txt not in _CONNECTORS:
            score += 0.2

    # Terminal punctuation on the last word of the window.
    last_words = [w for w in all_words if w[0] < end and w[1] <= end + 0.5]
    if last_words:
        last_txt = last_words[-1][3]
        if last_txt and last_txt[-1] in ".!?":
            score += 0.2

    return min(score, 1.0)


def _narration_preview(
    start: float,
    end: float,
    segments: list[CaptionSegment],
    max_chars: int = 300,
) -> str:
    """Reconstruct narration text for words within [start, end]."""
    words: list[str] = []
    prev_sentence: str | None = None
    for seg in segments:
        for w in seg.words:
            if w.start_sec >= start and w.end_sec <= end + 0.5:
                words.append(w.word)
    text = " ".join(words)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _score_all_candidates(
    segments: list[CaptionSegment],
    timeline: Timeline,
    script: Script,
    n: int,
) -> list[Candidate]:
    """Score windows starting within the first master beat only; return top *n*.

    The Short is always cut from Beat 1. This is non-negotiable — the strategic
    hook is the long-form video's promise. Scoring windows from the full video
    picks random high-energy moments.
    """
    # Flatten alignment data.
    all_words: list[tuple[float, float, bool, str]] = []
    sentence_starts: list[float] = []

    # Determine Beat 1 end time from emotional_register. Support both the new
    # master-beat register and legacy clinical_shock scripts.
    cold_fact_ids = {
        s.id
        for ch in script.chapters
        for s in ch.sentences
        if getattr(s, "emotional_register", "") in {"strategic_hook", "clinical_shock"}
    }

    # Build full word list first so we can find the latest end time of cold fact words.
    for seg in segments:
        if seg.words:
            for w in seg.words:
                all_words.append((w.start_sec, w.end_sec, w.is_emphasis, w.word))

    # Scene 1 ends at the latest word end_sec belonging to a clinical_shock sentence.
    scene1_end: float | None = None
    if cold_fact_ids:
        cold_ends = [
            w.end_sec
            for seg in segments
            if seg.sentence_id in cold_fact_ids
            for w in seg.words
        ]
        if cold_ends:
            scene1_end = max(cold_ends)

    # Fallback: use timeline chapter 0 boundary if register data isn't available.
    if scene1_end is None:
        for ch in timeline.chapters:
            if ch.index == 0:
                scene1_end = ch.end_sec
                break

    for seg in segments:
        if seg.words:
            start = seg.words[0].start_sec
            if scene1_end is None or start < scene1_end:
                sentence_starts.append(start)

    if scene1_end is not None:
        log.info(
            "shorts_scene1_locked",
            scene1_end_sec=round(scene1_end, 1),
            candidate_starts=len(sentence_starts),
            cold_fact_ids=len(cold_fact_ids),
        )

    if not sentence_starts:
        return []

    total_dur = timeline.total_duration_sec
    midpoint = (SHORTS_MIN_DURATION_SEC + SHORTS_MAX_DURATION_SEC) / 2
    duration_tiers = [SHORTS_MIN_DURATION_SEC, midpoint, SHORTS_MAX_DURATION_SEC]

    # Gather raw scores for all candidate windows.
    raw: list[tuple[float, float, float, float, float, int, str]] = []
    # (start, end, emph_density, energy_score, coherence_score, chapter_idx, chapter_energy)

    for start in sentence_starts:
        for dur in duration_tiers:
            end = start + dur
            if end > total_dur:
                continue

            dur_sec = end - start
            emph_count = sum(
                1 for ws, we, em, _ in all_words if em and ws >= start and ws < end
            )
            emph_density = emph_count / dur_sec

            mid = (start + end) / 2
            ch_idx, ch_energy = _chapter_energy_for_window(mid, timeline, script)
            en_score = _ENERGY_SCORE.get(ch_energy, 0.5)

            coh_score = _coherence_score(start, end, sentence_starts, all_words)

            raw.append((start, end, emph_density, en_score, coh_score, ch_idx, ch_energy))

    if not raw:
        return []

    # Normalize emphasis density to [0, 1] across all candidates.
    max_density = max(r[2] for r in raw) or 1.0

    def _total(emph, energy, coh) -> float:
        return 0.40 * (emph / max_density) + 0.30 * energy + 0.30 * coh

    scored = sorted(
        raw,
        key=lambda r: _total(r[2], r[3], r[4]),
        reverse=True,
    )

    # Greedy selection: highest score first, reject overlapping windows.
    selected: list[tuple] = []
    for row in scored:
        start, end = row[0], row[1]
        overlaps = any(
            start < sel[1] and end > sel[0]
            for sel in selected
        )
        if not overlaps:
            selected.append(row)
        if len(selected) >= n:
            break

    candidates = []
    for rank, (start, end, ed, en, coh, ch_idx, ch_en) in enumerate(selected, start=1):
        total = _total(ed, en, coh)
        text = _narration_preview(start, end, segments)
        candidates.append(Candidate(
            rank=rank,
            start_sec=round(start, 3),
            end_sec=round(end, 3),
            duration_sec=round(end - start, 1),
            chapter_index=ch_idx,
            chapter_energy=ch_en,
            emphasis_density=round(ed, 4),
            energy_score=round(en, 2),
            coherence_score=round(coh, 2),
            total_score=round(total, 4),
            narration_text=text,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def _print_candidates(candidates: list[Candidate], slug: str) -> None:
    print(f"\nShort candidates for '{slug}' — choose up to {SHORTS_COUNT}:\n")
    for c in candidates:
        print(
            f"  #{c.rank}  {_fmt_time(c.start_sec)} – {_fmt_time(c.end_sec)}"
            f"  ({c.duration_sec:.0f}s)  score: {c.total_score:.3f}"
            f"  ch{c.chapter_index} [{c.chapter_energy}]"
        )
        print(
            f"      emph: {c.emphasis_density:.3f}/s"
            f"  energy: {c.energy_score:.2f}"
            f"  coherence: {c.coherence_score:.2f}"
        )
        preview = c.narration_text[:200].replace("\n", " ")
        print(f"      \"{preview}\"")
        print()

    pick_example = ",".join(str(c.rank) for c in candidates[:SHORTS_COUNT])
    print(
        f"Pick your shorts:\n"
        f"  vidmaxx shorts {slug} --pick {pick_example}\n"
    )


# ---------------------------------------------------------------------------
# Mobile ASS captions (1080×1920, 96 pt)
# ---------------------------------------------------------------------------

_SHORTS_ASS_HEADER = (
    "[Script Info]\n"
    "Title: VidMaxx Shorts Captions\n"
    "ScriptType: v4.00+\n"
    f"PlayResX: {SHORTS_WIDTH}\n"
    f"PlayResY: {SHORTS_HEIGHT}\n"
    "WrapStyle: 1\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Normal,Arial,96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
    "0,0,0,0,100,100,0,0,1,4,1,2,60,60,120,1\n"
    "Style: Emphasis,Arial,96,&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,4,1,2,60,60,120,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def _ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def _write_shorts_ass(
    segments: list[CaptionSegment],
    window_start: float,
    window_end: float,
    out_path: Path,
) -> None:
    lines = [_SHORTS_ASS_HEADER]
    for seg in segments:
        for word in seg.words:
            if word.end_sec <= window_start or word.start_sec >= window_end:
                continue
            start = max(word.start_sec - window_start, 0.0)
            end = max(word.end_sec - window_start, 0.0)
            if end <= start:
                continue
            style = "Emphasis" if word.is_emphasis else "Normal"
            text = word.word.replace("{", "").replace("}", "").replace("\n", " ")
            lines.append(
                f"Dialogue: 0,{_ts(start)},{_ts(end)},{style},,0,0,0,,{text}\n"
            )
    out_path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# FFmpeg render — synchronous for ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _render_short_sync(
    video_path_str: str,
    window_start: float,
    window_dur: float,
    ass_path_str: str,
    hook_text: str,
    out_path_str: str,
    video_encoder: str = VIDEO_ENCODER,
) -> str:
    safe_ass = (
        ass_path_str
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
    )
    safe_hook = (
        hook_text
        .replace("'", "")
        .replace(":", " ")
        .replace("\\", "")
        .replace("[", "")
        .replace("]", "")
        [:72]
    )

    filter_complex = (
        f"[0:v]"
        f"crop={_CROP_W}:{_CROP_H}:{_CROP_X}:0,"
        f"scale={SHORTS_WIDTH}:{SHORTS_HEIGHT},"
        f"ass='{safe_ass}',"
        f"drawtext=text='{safe_hook}'"
        f":fontsize=64:fontcolor=white"
        f":x=(w-text_w)/2:y=h*0.15"
        f":enable='lte(t\\,{_HOOK_DURATION_SEC})'"
        f":box=1:boxcolor=black@0.5:boxborderw=8"
        f"[vout];"
        f"[0:a]atrim=duration={window_dur:.4f},asetpts=PTS-STARTPTS[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{window_start:.4f}",
        "-i", video_path_str,
        "-t", f"{window_dur:.4f}",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", video_encoder,
        "-b:v", "6M",
        "-c:a", AUDIO_ENCODER,
        "-b:a", AUDIO_BITRATE,
        "-r", str(VIDEO_FPS),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path_str,
    ]

    log.info(
        "render_short_start",
        out=out_path_str,
        start=round(window_start, 1),
        dur=round(window_dur, 1),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg short render failed:\n{result.stderr[-2000:]}"
        )
    log.info("render_short_done", out=out_path_str)
    return out_path_str


# ---------------------------------------------------------------------------
# Public stage runners
# ---------------------------------------------------------------------------

async def run_select(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
) -> list[Candidate]:
    """Score candidate windows, print top 5, save candidates.json.

    Advances: SHORTS_SELECT → SHORTS_RENDER.
    Pipeline halts after this — user must run `vidmaxx shorts <slug> --pick`.
    """
    paths = state_mgr.paths(slug)

    timeline = Timeline.model_validate_json(paths.timeline_json.read_text())
    script = Script.model_validate_json(paths.script.read_text())

    segments: list[CaptionSegment] = []
    if paths.alignment_json.exists():
        raw = json.loads(paths.alignment_json.read_text())
        segments = [CaptionSegment.model_validate(s) for s in raw]
    else:
        log.warning("shorts_select_no_alignment", slug=slug)

    log.info("stage_shorts_select_start", slug=slug)

    with state_mgr.run_stage(slug, PipelineStage.SHORTS_SELECT) as paths:
        candidates = _score_all_candidates(segments, timeline, script, n=_TOP_N)
        _save_candidates(candidates, paths)

    _print_candidates(candidates, slug)
    return candidates


async def run_render(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
    picks: list[int],
) -> list[Path]:
    """Render user-chosen candidates (by 1-based rank) as 9:16 MP4s.

    Advances: SHORTS_RENDER → PUBLISH.
    """
    paths = state_mgr.paths(slug)

    all_candidates = _load_candidates(paths)
    valid_ranks = {c.rank for c in all_candidates}
    bad = [p for p in picks if p not in valid_ranks]
    if bad:
        raise ValueError(
            f"Invalid pick(s): {bad}. Valid ranks: {sorted(valid_ranks)}"
        )

    chosen = [c for c in all_candidates if c.rank in picks]

    segments: list[CaptionSegment] = []
    if paths.alignment_json.exists():
        raw = json.loads(paths.alignment_json.read_text())
        segments = [CaptionSegment.model_validate(s) for s in raw]

    log.info("stage_shorts_render_start", slug=slug, picks=picks)

    with state_mgr.run_stage(slug, PipelineStage.SHORTS_RENDER) as paths:
        shorts_dir = paths.shorts_dir
        ass_dir = shorts_dir / "captions"
        ass_dir.mkdir(parents=True, exist_ok=True)

        futures: dict = {}
        with ProcessPoolExecutor(
            max_workers=min(len(chosen), RENDER_MAX_WORKERS)
        ) as pool:
            for i, cand in enumerate(chosen, start=1):
                ass_path = ass_dir / f"short_{i}.ass"
                out_path = shorts_dir / f"short_{i}.mp4"

                _write_shorts_ass(segments, cand.start_sec, cand.end_sec, ass_path)

                # Hook text = first ~60 chars of the narration preview.
                hook_text = cand.narration_text[:60].rstrip(" ,")

                future = pool.submit(
                    _render_short_sync,
                    str(paths.final_video),
                    cand.start_sec,
                    cand.duration_sec,
                    str(ass_path),
                    hook_text,
                    str(out_path),
                )
                futures[future] = (i, out_path, cand.rank)

            short_paths: list[Path] = []
            for future in as_completed(futures):
                i, out_path, rank = futures[future]
                try:
                    future.result()
                    short_paths.append(out_path)
                    log.info(
                        "short_complete",
                        rank=rank,
                        size_mb=round(out_path.stat().st_size / 1e6, 1),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Short (rank {rank}) render failed: {exc}"
                    ) from exc

    short_paths.sort()
    log.info("stage_shorts_render_complete", slug=slug, count=len(short_paths))
    return short_paths
