"""
Manim executor — runs generated scene code in an isolated temp dir,
enforces exact duration with FFmpeg, cleans up on exit.

Degrades gracefully if manim is not installed: execute() returns
(False, "manim not installed", None) so callers fall through to
the matplotlib fallback without crashing the pipeline.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_MANIM_AVAILABLE = shutil.which("manim") is not None

if not _MANIM_AVAILABLE:
    log.warning("manim_not_installed", note="Viz agents will fall back to matplotlib")


def _strip_fences(code: str) -> str:
    if "```" not in code:
        return code.strip()
    for part in code.split("```"):
        s = part.lstrip("python").strip()
        if "class MainScene" in s or "from manim import" in s:
            return s
    return code.strip()


def _frame_hash(video_path: Path, t: float, tmp_path: Path, idx: int) -> str | None:
    frame = tmp_path / f"f{idx}.png"
    r = subprocess.run(
        ["ffmpeg", "-ss", str(t), "-i", str(video_path),
         "-vframes", "1", str(frame), "-y", "-loglevel", "error"],
        capture_output=True,
    )
    if r.returncode == 0 and frame.exists():
        return hashlib.md5(frame.read_bytes()).hexdigest()
    return None


class ManimExecutor:

    def execute(
        self,
        code: str,
        output_path: Path,
        duration_sec: float,
        quality: str = "l",
    ) -> tuple[bool, str | None, Path | None]:

        if not _MANIM_AVAILABLE:
            return False, "manim not installed", None

        code = _strip_fences(code)

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            script = tmp / "scene.py"
            script.write_text(code)
            media_dir = tmp / "media"

            result = subprocess.run(
                [
                    "manim", f"-q{quality}",
                    "--media_dir", str(media_dir),
                    "--output_file", "MainScene",
                    str(script),
                    "MainScene",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                error = (result.stdout[-1500:] + "\n" + result.stderr[-1500:]).strip()
                return False, error, None

            mp4s = list(media_dir.rglob("*.mp4"))
            if not mp4s:
                return False, "Manim exit 0 but no mp4 produced", None

            rendered = mp4s[0]
            enforced = tmp / "final.mp4"

            ff = subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(rendered),
                    "-t", str(duration_sec),
                    "-vf", (
                        "fps=30,"
                        "scale=1920:1080:force_original_aspect_ratio=decrease,"
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                    ),
                    "-c:v", "h264_videotoolbox",
                    "-c:a", "aac",
                    str(enforced),
                    "-y", "-loglevel", "error",
                ],
                capture_output=True,
                text=True,
            )

            if ff.returncode != 0:
                return False, f"FFmpeg duration enforcement failed: {ff.stderr}", None

            ok, reason = self._verify_duration(enforced, duration_sec)
            if not ok:
                log.warning("viz_duration_mismatch", expected=duration_sec, reason=reason)
                return False, f"duration_mismatch: {reason}", None

            is_bad, reason = self._check_quality(enforced, duration_sec)
            if is_bad:
                log.warning("viz_quality_check_failed", reason=reason)
                return False, f"quality_check: {reason}", None

            shutil.copy(enforced, output_path)
            return True, None, output_path

    def extract_frame(self, video_path: Path, t: float, out_path: Path) -> bool:
        """Extract single frame at timestamp t (seconds) to out_path as PNG."""
        r = subprocess.run(
            ["ffmpeg", "-ss", str(t), "-i", str(video_path),
             "-vframes", "1", str(out_path), "-y", "-loglevel", "error"],
            capture_output=True,
        )
        return r.returncode == 0 and out_path.exists()

    def _verify_duration(
        self, video_path: Path, expected: float, tolerance: float = 0.15
    ) -> tuple[bool, str]:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not r.stdout.strip():
            log.warning("viz_duration_probe_failed", stderr=r.stderr[:200])
            return True, ""  # can't probe — fail-open

        try:
            actual = float(r.stdout.strip())
        except ValueError:
            return True, ""  # unparseable — fail-open

        if abs(actual - expected) > tolerance:
            return False, f"actual={actual:.2f}s expected={expected:.2f}s delta={abs(actual-expected):.2f}s"
        return True, ""

    def _check_quality(self, video_path: Path, duration_sec: float) -> tuple[bool, str]:
        """
        Samples frames at 15%, 65%, 90% of duration.
        Fails if:
          - all three frames are identical (fully static / frozen at start)
          - 65% == 90% but 15% != 65% (animation finished too early, frozen hold)
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            h15 = _frame_hash(video_path, duration_sec * 0.15, tmp_path, 0)
            h65 = _frame_hash(video_path, duration_sec * 0.65, tmp_path, 1)
            h90 = _frame_hash(video_path, duration_sec * 0.90, tmp_path, 2)

        extracted = [h for h in (h15, h65, h90) if h is not None]
        if len(extracted) < 2:
            log.warning("viz_quality_frame_extract_failed",
                        video=str(video_path), extracted=len(extracted))
            return False, ""  # can't check — fail-open

        if len(set(extracted)) == 1:
            return True, "all sampled frames identical — animation frozen at starting value"

        if h65 and h90 and h65 == h90 and h15 != h65:
            return True, "animation finished too early — final frame frozen for remainder of duration"

        return False, ""
