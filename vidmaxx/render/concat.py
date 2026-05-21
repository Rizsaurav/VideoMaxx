"""
Concatenate per-chapter MP4s into the final video using FFmpeg concat demuxer.

The concat demuxer does a stream-copy (no re-encode), so it's near-instant
regardless of video length.  All chapter MP4s must share the same codec,
resolution, and frame rate — which is guaranteed when chapter_renderer
produces them with the same settings.
"""

import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def concat_chapters(
    chapter_paths: list[Path],
    out_path: Path,
) -> Path:
    """Join sorted chapter MP4s into a single output file.

    chapter_paths must already be in the correct order (sorted by chapter
    index before calling).  out_path parent directory must exist.
    """
    if not chapter_paths:
        raise ValueError("concat_chapters called with empty chapter list")

    if len(chapter_paths) == 1:
        # Trivial case — just copy/rename without invoking FFmpeg.
        import shutil
        shutil.copy2(chapter_paths[0], out_path)
        log.info("concat_trivial_copy", out=str(out_path))
        return out_path

    # Write a temporary concat list file.
    # FFmpeg concat demuxer format:
    #   file '/absolute/path/to/chapter.mp4'
    #   file '/absolute/path/to/next.mp4'
    list_lines = [f"file '{p.resolve()}'\n" for p in chapter_paths]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.writelines(list_lines)
        list_path = fh.name

    log.info(
        "concat_start",
        chapters=len(chapter_paths),
        out=str(out_path),
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(list_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg concat failed:\n{result.stderr[-2000:]}"
        )

    log.info("concat_done", out=str(out_path), size_mb=round(out_path.stat().st_size / 1e6, 1))
    return out_path
