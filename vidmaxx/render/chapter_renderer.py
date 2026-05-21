"""
Per-chapter FFmpeg renderer — runs synchronously inside ProcessPoolExecutor.

All parameters are primitives (str, dict, float) so the function pickles
cleanly for multiprocessing.  Deserialise ChapterSlice from the dict inside
the function, not at the call site.

SHORT-CLIP LOOP HANDLING
------------------------
Pexels and Pixabay commonly return clips of 10–15 s.  A sentence may last
20–30 s.  Without looping, FFmpeg would produce a short video segment that
breaks the chapter concat.

Fix applied here:
  - Video assets: -stream_loop -1 -ss <source_start> -i <path>
      The loop makes the input stream infinite; filtergraph trim=duration=D
      cuts it at exactly the sentence duration D.
  - Image assets: -loop 1 -t <duration> -i <path>
      The image is repeated for exactly the sentence duration.

DO NOT remove these flags or the trim step in build_clip_filter without
replacing the loop mechanism with something equivalent.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import structlog

from vidmaxx.config.constants import (
    AUDIO_BITRATE,
    AUDIO_ENCODER,
    VIDEO_BITRATE,
    VIDEO_ENCODER,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from vidmaxx.models.asset import AssetType
from vidmaxx.models.timeline import ChapterSlice, MotionEffect
from vidmaxx.render.filtergraph import (
    build_audio_mix,
    build_clip_filter,
    build_subtitle_filter,
    build_video_concat,
)

log = structlog.get_logger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".webm"}


def _pretrim_clip(asset_path: str, start_sec: float, duration_sec: float, tmp_dir: str) -> str:
    """Trim/loop a video to exact duration at target FPS. Returns path to a short temp file."""
    key = hashlib.md5(f"{asset_path}{start_sec:.4f}{duration_sec:.4f}".encode()).hexdigest()[:12]
    out = Path(tmp_dir) / f"clip_{key}.mp4"
    if out.exists():
        return str(out)

    # Step 1: stream copy (fast — no decode/encode). May produce wrong FPS.
    copy_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-ss", f"{start_sec:.4f}",
        "-i", asset_path,
        "-t", f"{duration_sec:.4f}",
        "-c:v", "copy", "-an", str(out),
    ]
    result = subprocess.run(copy_cmd, capture_output=True)

    if result.returncode == 0:
        # Verify output has correct FPS — re-encode if it doesn't
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True,
        )
        fps_str = probe.stdout.strip()  # e.g. "30/1" or "2997/100"
        try:
            num, den = fps_str.split("/")
            fps_ok = abs(int(num) / int(den) - VIDEO_FPS) < 1.0
        except Exception:
            fps_ok = False

        if fps_ok:
            return str(out)
        out.unlink(missing_ok=True)

    # Step 2: re-encode to normalize FPS (needed for non-30fps sources)
    encode_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-ss", f"{start_sec:.4f}",
        "-i", asset_path,
        "-t", f"{duration_sec:.4f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-r", str(VIDEO_FPS), "-an", str(out),
    ]
    subprocess.run(encode_cmd, capture_output=True, check=True)
    return str(out)


def render_chapter(
    chapter_dict: dict,
    narration_path_str: str,
    out_path_str: str,
    ass_path_str: str,
    video_encoder: str = VIDEO_ENCODER,
    video_bitrate: str = VIDEO_BITRATE,
    audio_bitrate: str = AUDIO_BITRATE,
) -> str:
    """Render one chapter to an MP4 file.

    Called inside ProcessPoolExecutor — must stay synchronous and must not
    hold any non-picklable state.  Returns out_path_str on success.
    """
    chapter = ChapterSlice.model_validate(chapter_dict)
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips = chapter.video_clips
    if not clips:
        raise ValueError(f"Chapter {chapter.index} has no video clips — cannot render.")

    # ------------------------------------------------------------------
    # Pre-trim video clips to exact durations
    # ------------------------------------------------------------------
    # Each clip becomes a short finite file so the main render never holds
    # 25+ infinite-loop decode buffers in memory simultaneously.
    tmp_dir = tempfile.mkdtemp(prefix="vidmaxx_ch_")

    # ------------------------------------------------------------------
    # Build FFmpeg input list
    # ------------------------------------------------------------------
    inputs: list[tuple[list[str], str]] = []

    for clip in clips:
        path = Path(clip.asset_path)
        ext = path.suffix.lower()
        is_video = ext in _VIDEO_EXTS

        if is_video:
            trimmed = _pretrim_clip(str(path), clip.source_start_sec, clip.duration_sec, tmp_dir)
            flags = []  # finite file — no loop needed
            inputs.append((flags, trimmed))
        else:
            if clip.motion != MotionEffect.NONE:
                # Single-frame input: zoompan's d parameter creates the full
                # duration. -loop/-t would multiply frames and produce 3-hour output.
                flags = []
            else:
                flags = ["-loop", "1", "-t", f"{clip.duration_sec:.4f}"]
            inputs.append((flags, str(path)))

    narr_idx = len(inputs)
    inputs.append(([], narration_path_str))

    music_idx: int | None = None
    for mc in chapter.music_clips:
        music_idx = len(inputs)
        # Loop music bed so it fills any chapter duration without running out.
        inputs.append((["-stream_loop", "-1"], str(mc.audio_path)))
        break  # at most one music bed per chapter

    sfx_idx: int | None = None
    sfx_delay_ms: float = 0.0
    sfx_vol: float = 0.6
    for sc in chapter.sfx_clips:
        sfx_idx = len(inputs)
        inputs.append(([], str(sc.audio_path)))
        sfx_delay_ms = max(0.0, (sc.start_sec - chapter.start_sec) * 1000)
        sfx_vol = sc.volume
        break  # at most one SFX per chapter

    # ------------------------------------------------------------------
    # Build filter_complex
    # ------------------------------------------------------------------
    filter_parts: list[str] = []

    for i, clip in enumerate(clips):
        ext = Path(clip.asset_path).suffix.lower()
        is_video = ext in _VIDEO_EXTS
        filter_parts.append(build_clip_filter(i, clip, is_video))

    filter_parts.append(build_video_concat(len(clips)))
    filter_parts.append(build_subtitle_filter(ass_path_str))
    filter_parts.append(
        build_audio_mix(
            narr_idx=narr_idx,
            ch_start_sec=chapter.start_sec,
            ch_dur_sec=chapter.duration_sec,
            music_idx=music_idx,
            duck_vol=chapter.music_clips[0].duck_volume if chapter.music_clips else 0.05,
            sfx_idx=sfx_idx,
            sfx_delay_ms=sfx_delay_ms,
            sfx_vol=sfx_vol,
            music_duck=chapter.music_duck,
        )
    )

    filter_complex = ";".join(filter_parts)

    # ------------------------------------------------------------------
    # Assemble the FFmpeg command
    # ------------------------------------------------------------------
    cmd: list[str] = ["ffmpeg", "-y", "-threads", "2"]

    for flags, path in inputs:
        cmd.extend(flags)
        cmd.extend(["-i", path])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", video_encoder,
        "-crf", "23",
        "-preset", "ultrafast",
        "-c:a", AUDIO_ENCODER,
        "-b:a", audio_bitrate,
        "-r", str(VIDEO_FPS),
        "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "1024",
        str(out_path),
    ])

    log.info(
        "render_chapter_start",
        chapter=chapter.index,
        clips=len(clips),
        duration=round(chapter.duration_sec, 1),
        out=str(out_path),
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up pre-trimmed temp clips regardless of outcome
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed for chapter {chapter.index}:\n{result.stderr[-2000:]}"
        )

    log.info("render_chapter_done", chapter=chapter.index)
    return out_path_str
