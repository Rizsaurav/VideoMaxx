"""
FFmpeg filter_complex string builders for per-chapter rendering.

Public API
----------
build_clip_filter(input_idx, clip, is_video) -> str
    Returns one filter chain for a single VideoClip (no trailing semicolon).

build_video_concat(n_clips) -> str
    Returns the concat filter that joins [v0]..[vN-1] → [vraw].

build_subtitle_filter(ass_path) -> str
    Returns the ASS subtitle burn-in filter [vraw] → [vout].

build_audio_mix(...) -> str
    Returns the full audio filter chain → [aout].

LOOP NOTE
---------
Short stock clips (Pexels typically returns 10–15 s clips) are looped at
the FFmpeg input level with  -stream_loop -1  (videos) or  -loop 1
(images).  The filtergraph trims the looped stream to exactly the sentence
duration so the output is always the right length.  Without this, a 10 s
clip against a 20 s sentence would produce a 10 s video clip and break the
concat.  DO NOT remove the trim step or the loop flags from chapter_renderer.
"""

from __future__ import annotations

from vidmaxx.config.constants import (
    KEN_BURNS_SCALE_END,
    KEN_BURNS_SCALE_START,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from vidmaxx.models.timeline import MotionEffect, VideoClip

# Pre-scale dimensions give zoompan enough headroom for up to 10 % zoom.
# 1280 * 1.10 = 1408,  720 * 1.10 = 792  (both even).
_PRE_W = 1408
_PRE_H = 792


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zoompan_expr(motion: MotionEffect, duration_sec: float) -> str:
    """Return a zoompan filter string for the given motion type.

    Output size is always VIDEO_WIDTH × VIDEO_HEIGHT.  The calling chain
    must pre-scale the input to at least _PRE_W × _PRE_H.
    """
    W, H = VIDEO_WIDTH, VIDEO_HEIGHT
    fps = VIDEO_FPS
    total_frames = max(1, int(duration_sec * fps))

    if motion == MotionEffect.KEN_BURNS_IN:
        step = (KEN_BURNS_SCALE_END - KEN_BURNS_SCALE_START) / total_frames
        z = f"min({KEN_BURNS_SCALE_START}+on*{step:.8f},{KEN_BURNS_SCALE_END})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    elif motion == MotionEffect.KEN_BURNS_OUT:
        step = (KEN_BURNS_SCALE_END - KEN_BURNS_SCALE_START) / total_frames
        z = f"max({KEN_BURNS_SCALE_END}-on*{step:.8f},{KEN_BURNS_SCALE_START})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    elif motion == MotionEffect.PAN_RIGHT:
        # Zoom fixed at 1.04; pan the crop window left→right across the frame.
        z_val = 1.04
        z = str(z_val)
        x = f"iw*(1-1/{z_val})*on/{total_frames}"
        y = "ih/2-(ih/zoom/2)"

    elif motion == MotionEffect.PAN_LEFT:
        z_val = 1.04
        z = str(z_val)
        x = f"iw*(1-1/{z_val})*(1-on/{total_frames})"
        y = "ih/2-(ih/zoom/2)"

    else:
        raise ValueError(f"zoompan not applicable for motion={motion!r}")

    return (
        f"zoompan=z='{z}':x='{x}':y='{y}'"
        f":d={total_frames}:s={W}x{H}:fps={fps}"
    )


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_clip_filter(
    input_idx: int,
    clip: VideoClip,
    is_video: bool,
) -> str:
    """Build the filtergraph chain for one VideoClip.

    is_video=True  → pre-trimmed finite video file; trim=duration is a safety
                     net. Ken Burns is suppressed — videos have inherent motion
                     and zoompan on a multi-frame stream multiplies frames by d.
    is_video=False → static image input.
      motion=NONE  → -loop 1 -t D at input level controls duration; fps filter
                     here normalises the output rate.
      motion!=NONE → single-frame input (no -loop/-t); zoompan's d parameter
                     creates exactly duration_sec*fps output frames. Do NOT add
                     an fps filter before zoompan — that would multiply frames.

    Returns a chain string without a trailing semicolon:
        [N:v] ... [vN]
    """
    dur = clip.duration_sec
    # Videos have inherent motion; Ken Burns only applies to still images.
    # zoompan outputs d frames *per input frame*, so feeding a video stream
    # multiplies duration by d — that is what caused the 3-hour chapters.
    motion = clip.motion if not is_video else MotionEffect.NONE
    label_out = f"[v{input_idx}]"

    parts: list[str] = [f"[{input_idx}:v]"]

    if is_video:
        parts.append(f"trim=duration={dur:.4f},setpts=PTS-STARTPTS,")

    if motion == MotionEffect.NONE:
        parts.append(
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
            f"setsar=1,fps={VIDEO_FPS},setpts=PTS-STARTPTS"
        )
    else:
        # Image Ken Burns: single-frame input → zoompan creates all frames.
        # No fps filter before zoompan; zoompan's fps= parameter sets output rate.
        parts.append(
            f"scale={_PRE_W}:{_PRE_H}:force_original_aspect_ratio=increase,"
            f"crop={_PRE_W}:{_PRE_H},setsar=1,"
        )
        parts.append(f"{_zoompan_expr(motion, dur)},")
        parts.append("setpts=PTS-STARTPTS")

    parts.append(label_out)
    return "".join(parts)


def build_video_concat(n_clips: int) -> str:
    """[v0][v1]...[vN-1]concat=n=N:v=1:a=0[vraw]"""
    if n_clips == 1:
        return "[v0]copy[vraw]"
    labels = "".join(f"[v{i}]" for i in range(n_clips))
    return f"{labels}concat=n={n_clips}:v=1:a=0[vraw]"


def build_subtitle_filter(ass_path: str) -> str:
    """[vraw]ass=<path>[vout] — burns captions into the video stream."""
    # Escape single-quotes and backslashes in the path for the filter string.
    safe = ass_path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return f"[vraw]ass='{safe}'[vout]"


def build_audio_mix(
    narr_idx: int,
    ch_start_sec: float,
    ch_dur_sec: float,
    music_idx: int | None = None,
    duck_vol: float = 0.05,
    fade_dur: float = 1.5,
    sfx_idx: int | None = None,
    sfx_delay_ms: float = 0.0,
    sfx_vol: float = 0.6,
    music_duck: bool = False,
) -> str:
    """Build the audio portion of the filter_complex.

    Returns one long filter chain that maps the audio inputs → [aout].

    narr_idx   : FFmpeg input index for the full concatenated narration WAV.
    ch_start_sec / ch_dur_sec : used to trim the narration to this chapter.
    music_idx  : optional; looped music bed (use -stream_loop -1 at input).
    sfx_idx    : optional; SFX clip.
    music_duck : if True (PATTERN INTERRUPT scenes), music volume=0 for the
                 whole chapter with a 0.5s fade-in after — complete silence.
    """
    chains: list[str] = []
    audio_labels: list[str] = []

    # Narration — trim to chapter window.
    chains.append(
        f"[{narr_idx}:a]"
        f"atrim=start={ch_start_sec:.4f}:duration={ch_dur_sec:.4f},"
        f"asetpts=PTS-STARTPTS"
        f"[naudio]"
    )
    audio_labels.append("[naudio]")

    # Music bed — already looped at input; trim + volume + fade.
    if music_idx is not None:
        if music_duck:
            # Pattern interrupt: complete silence — volume=0 throughout.
            # A 0.5s fade-in at the very end so the next chapter's music
            # doesn't snap in. The fade starts at (ch_dur - 0.5s).
            fade_back_start = max(0.0, ch_dur_sec - 0.5)
            chains.append(
                f"[{music_idx}:a]"
                f"atrim=duration={ch_dur_sec:.4f},asetpts=PTS-STARTPTS,"
                f"volume=0,"
                f"afade=t=out:st=0:d={ch_dur_sec:.4f},"
                f"afade=t=in:st={fade_back_start:.4f}:d=0.5"
                f"[maudio]"
            )
        else:
            # Normal chapters: duck to duck_vol with in/out fades.
            fo_start = max(0.0, ch_dur_sec - fade_dur)
            chains.append(
                f"[{music_idx}:a]"
                f"atrim=duration={ch_dur_sec:.4f},asetpts=PTS-STARTPTS,"
                f"volume={duck_vol},"
                f"afade=t=in:st=0:d={fade_dur},"
                f"afade=t=out:st={fo_start:.4f}:d={fade_dur}"
                f"[maudio]"
            )
        audio_labels.append("[maudio]")

    # SFX — delay to chapter-relative offset.
    if sfx_idx is not None:
        delay_ms = int(max(0.0, sfx_delay_ms))
        chains.append(
            f"[{sfx_idx}:a]"
            f"adelay={delay_ms}|{delay_ms},"
            f"volume={sfx_vol}"
            f"[saudio]"
        )
        audio_labels.append("[saudio]")

    # Mix all audio streams.
    n = len(audio_labels)
    if n == 1:
        chains.append(f"{audio_labels[0]}acopy[aout]")
    else:
        label_str = "".join(audio_labels)
        chains.append(
            f"{label_str}amix=inputs={n}:duration=first:normalize=0[aout]"
        )

    return ";".join(chains)
