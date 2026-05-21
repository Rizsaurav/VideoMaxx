"""
Pipeline orchestrator — runs stages sequentially for a project.

Each stage function owns its own run_stage context, which validates the
project is at the expected stage before running and advances it on success.
The orchestrator's job is purely to call them in order, starting from
wherever the project currently sits.

run_log.json is written at the project root. It is initialized on first run
(if not present) and appended per stage: stage, started_at, completed_at,
duration_sec. Each entry also records whether the stage raised an error.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
import typer

from vidmaxx.config.settings import Settings
from vidmaxx.models.project import STAGE_ORDER, PipelineStage
from vidmaxx.pipeline.stages import (
    s01a_research as s01a,
    s01b_verify as s01b,
    s01c_freeze as s01c,
    s02a_architect as s02a,
    s02b_critic as s02b,
    s02c_optimizer as s02c,
    s02d_validate as s02d,
    s03_assets as s03,
    s03b_vlm_select as s03b,
    s04_tts as s04,
    s05_alignment as s05,
    s06_timeline as s06,
    s07_render as s07,
    s08_shorts as s08,
    s09_publish as s09,
)
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

# Stages that terminate automatic progression.
# SHORTS_RENDER pauses for user input (vidmaxx shorts --pick).
_TERMINAL_STAGES = {
    PipelineStage.SHORTS_RENDER,
    PipelineStage.DONE,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_run_log(log_path: Path, entry: dict) -> None:
    if log_path.exists():
        entries = json.loads(log_path.read_text())
    else:
        entries = []
    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2))


def _critic_counts(paths) -> dict:
    """Read critic_report.json after CRITIC stage and return flag counts."""
    try:
        data = json.loads(paths.critic_report.read_text())
        flags = data.get("flags", [])
        return {
            "critical_count": sum(1 for f in flags if f.get("severity") == "CRITICAL"),
            "warning_count": sum(1 for f in flags if f.get("severity") == "WARNING"),
        }
    except Exception:
        return {}


async def run_pipeline(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
    stop_before: PipelineStage | None = None,
) -> None:
    """Run all pending stages for *slug* from its current stage onward.

    Resumes automatically if the project was stopped or failed mid-run —
    just fix the underlying issue and call this again.

    stop_before: if set, halt before entering that stage (exclusive).
    """
    project = state_mgr.load(slug)
    run_log_path = state_mgr.paths(slug).run_log

    if project.stage == PipelineStage.DONE:
        typer.echo(f"Project '{slug}' is already complete.")
        return

    if project.error:
        typer.echo(f"Retrying from stage '{project.stage.value}' (previous error: {project.error})")

    # CREATED is a pre-run sentinel; advance to RESEARCH so the first stage
    # runner finds the project at the stage it expects.
    if project.stage == PipelineStage.CREATED:
        state_mgr.advance(slug)

    # s01c (freeze) is synchronous — wrap in a coroutine for uniform handling
    async def _run_freeze():
        s01c.run(slug, state_mgr, cache, settings)

    # Ordered list of (stage, coroutine-factory) pairs.
    # SHORTS_RENDER is intentionally absent — triggered by the user
    # via `vidmaxx shorts <slug> --pick` after reviewing candidates.
    stage_runners = [
        (PipelineStage.RESEARCH,      lambda: s01a.run(slug, state_mgr, cache, settings)),
        (PipelineStage.VERIFY,        lambda: s01b.run(slug, state_mgr, cache, settings)),
        (PipelineStage.FREEZE,        _run_freeze),
        (PipelineStage.ARCHITECT,     lambda: s02a.run(slug, state_mgr, cache, settings)),
        (PipelineStage.CRITIC,        lambda: s02b.run(slug, state_mgr, cache, settings)),
        (PipelineStage.OPTIMIZER,     lambda: s02c.run(slug, state_mgr, cache, settings)),
        (PipelineStage.VALIDATE,      lambda: s02d.run(slug, state_mgr, cache, settings)),
        (PipelineStage.ASSETS,        lambda: s03.run(slug, state_mgr, cache, settings)),
        (PipelineStage.VLM_SELECT,    lambda: s03b.run(slug, state_mgr, cache, settings)),
        (PipelineStage.TTS,           lambda: s04.run(slug, state_mgr, settings)),
        (PipelineStage.ALIGNMENT,     lambda: s05.run(slug, state_mgr, settings)),
        (PipelineStage.TIMELINE,      lambda: s06.run(slug, state_mgr, settings)),
        (PipelineStage.RENDER,        lambda: s07.run(slug, state_mgr, settings)),
        (PipelineStage.SHORTS_SELECT, lambda: s08.run_select(slug, state_mgr, settings)),
        (PipelineStage.PUBLISH,       lambda: s09.run(slug, state_mgr, settings)),
    ]

    for stage, make_coro in stage_runners:
        if stop_before and stage == stop_before:
            typer.echo(f"Stopping before {stage.value} (--stop-before).")
            break

        project = state_mgr.load(slug)
        if project.stage in _TERMINAL_STAGES:
            break
        if project.stage != stage:
            # Already past this stage — skip.
            continue

        typer.echo(f"  {stage.value}...", nl=False)
        started_at = _now_iso()
        start_ts = datetime.fromisoformat(started_at)
        paths = state_mgr.paths(slug)
        try:
            await make_coro()
            completed_at = _now_iso()
            duration_sec = (datetime.fromisoformat(completed_at) - start_ts).total_seconds()
            entry: dict = {
                "stage": stage.value,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_sec": round(duration_sec, 2),
                "error": None,
            }
            # Attach critic flag counts so future debugging shows whether
            # CRITIC found a lot to fix — a sign the Architect prompt is drifting.
            if stage == PipelineStage.CRITIC:
                entry.update(_critic_counts(paths))
            _append_run_log(run_log_path, entry)
            typer.echo(" done")
        except Exception as exc:
            completed_at = _now_iso()
            duration_sec = (datetime.fromisoformat(completed_at) - start_ts).total_seconds()
            entry = {
                "stage": stage.value,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_sec": round(duration_sec, 2),
                "error": str(exc),
            }
            if stage == PipelineStage.CRITIC:
                entry.update(_critic_counts(paths))
            _append_run_log(run_log_path, entry)
            typer.echo(f" FAILED\n{exc}", err=True)
            raise typer.Exit(1) from exc

    project = state_mgr.load(slug)
    if project.stage == PipelineStage.DONE:
        paths = state_mgr.paths(slug)
        typer.echo(f"\nVideo: {paths.final_video}")
        if paths.shorts_dir.exists():
            shorts = sorted(paths.shorts_dir.glob("short_*.mp4"))
            if shorts:
                typer.echo("Shorts:")
                for s in shorts:
                    typer.echo(f"  {s}")
        if paths.publish_json.exists():
            published = json.loads(paths.publish_json.read_text())
            if published:
                typer.echo("YouTube:")
                for label, url in published.items():
                    typer.echo(f"  {label}: {url}")
    elif project.stage == PipelineStage.SHORTS_RENDER:
        typer.echo(
            "\nShort candidates saved. Review the list above, then render:\n"
            f"  vidmaxx shorts {slug} --pick 1,3"
        )
