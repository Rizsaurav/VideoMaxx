"""
vidmaxx CLI — topic in, YouTube video out.

Commands
--------
  vidmaxx new "the halting problem"   Create a project and run the full pipeline.
  vidmaxx run <slug>                  Resume a stopped or failed project.
  vidmaxx list                        Show all projects and their current stage.
  vidmaxx status <slug>               Show detailed status for one project.
"""

import asyncio
import json
import re
import secrets
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.pipeline.orchestrator import run_pipeline
from vidmaxx.pipeline.stages import s08_shorts as s08
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

app = typer.Typer(
    name="vidmaxx",
    help="Local YouTube automation pipeline — research to rendered video on M2.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slug(topic: str) -> str:
    """'The Halting Problem' → 'the-halting-problem-a3f2'"""
    base = re.sub(r"[^a-z0-9]+", "-", topic.lower().strip()).strip("-")[:48]
    suffix = secrets.token_hex(2)  # 4 hex chars — enough to avoid collisions
    return f"{base}-{suffix}"


def _get_state_mgr(settings: Settings) -> ProjectStateManager:
    return ProjectStateManager(settings.projects_root)


def _get_cache(settings: Settings) -> PipelineCache:
    return PipelineCache(settings.cache_dir)


def _load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        console.print("Make sure a .env file exists with GCP_PROJECT_ID, GOOGLE_API_KEY, PEXELS_API_KEY, etc.")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_BRIEF_REQUIRED_FIELDS = {"topic", "cold_fact", "false_belief", "three_layers", "twist", "dinner_fact"}


@app.command()
def new(
    topic: str = typer.Argument(..., help="Video topic, e.g. 'the halting problem'"),
    slug: Optional[str] = typer.Option(
        None, "--slug", "-s", help="Custom project slug (auto-generated if omitted)"
    ),
    stop_before: Optional[str] = typer.Option(
        None, "--stop-before", help="Halt before this stage, e.g. 'render'"
    ),
    create_only: bool = typer.Option(
        False, "--create-only", help="Create the project folder without running the pipeline"
    ),
    brief: Optional[Path] = typer.Option(
        None, "--brief", "-b",
        help="Path to a brief.json file (see brief_template.json). Strongly recommended.",
    ),
) -> None:
    """Create a project for TOPIC and run the full pipeline."""
    settings = _load_settings()
    state_mgr = _get_state_mgr(settings)
    cache = _get_cache(settings)

    # Validate and load brief if provided
    brief_data: dict | None = None
    if brief is not None:
        if not brief.exists():
            console.print(f"[red]Brief file not found:[/red] {brief}")
            raise typer.Exit(1)
        try:
            brief_data = json.loads(brief.read_text())
        except json.JSONDecodeError as exc:
            console.print(f"[red]Brief is not valid JSON:[/red] {exc}")
            raise typer.Exit(1)
        missing = _BRIEF_REQUIRED_FIELDS - set(brief_data.keys())
        if missing:
            console.print(
                f"[yellow]Warning:[/yellow] Brief is missing fields: {sorted(missing)}. "
                "Run without --brief or fill in the missing fields."
            )
    else:
        console.print(
            "[yellow]No --brief provided.[/yellow] "
            "Running without a brief produces generic content. "
            "See [bold]brief_template.json[/bold] to prepare one."
        )

    project_slug = slug or _make_slug(topic)

    if state_mgr.exists(project_slug):
        console.print(f"[yellow]Project '{project_slug}' already exists.[/yellow]")
        console.print(f"Use [bold]vidmaxx run {project_slug}[/bold] to resume it.")
        raise typer.Exit(1)

    project = state_mgr.create(topic, project_slug)
    console.print(f"Created project [bold]{project_slug}[/bold]  ({project.stage.value})")

    # Copy brief into the project dir so s02 can read it
    if brief_data is not None:
        paths = state_mgr.paths(project_slug)
        paths.brief_json.write_text(json.dumps(brief_data, indent=2))
        console.print(f"Brief saved to project: {paths.brief_json}")

    if create_only:
        console.print(f"Run [bold]vidmaxx run {project_slug}[/bold] when ready.")
        return

    _run(project_slug, state_mgr, cache, settings, stop_before)


@app.command()
def run(
    slug: str = typer.Argument(..., help="Project slug to resume"),
    stop_before: Optional[str] = typer.Option(
        None, "--stop-before", help="Halt before this stage, e.g. 'render'"
    ),
) -> None:
    """Resume a stopped or failed project from its current stage."""
    settings = _load_settings()
    state_mgr = _get_state_mgr(settings)
    cache = _get_cache(settings)

    if not state_mgr.exists(slug):
        console.print(f"[red]No project found with slug '{slug}'.[/red]")
        console.print("Use [bold]vidmaxx list[/bold] to see available projects.")
        raise typer.Exit(1)

    _run(slug, state_mgr, cache, settings, stop_before)


@app.command(name="list")
def list_projects() -> None:
    """List all projects and their current stage."""
    settings = _load_settings()
    state_mgr = _get_state_mgr(settings)

    projects = state_mgr.list_all()
    if not projects:
        console.print("No projects yet. Use [bold]vidmaxx new \"topic\"[/bold] to start one.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Slug")
    table.add_column("Topic")
    table.add_column("Stage")
    table.add_column("Updated")

    stage_colors = {
        PipelineStage.DONE: "green",
        PipelineStage.CREATED: "dim",
        PipelineStage.SHORTS_RENDER: "cyan",  # waiting for user input
    }

    for p in projects:
        default_color = "red" if p.error else ("yellow" if p.stage != PipelineStage.DONE else "green")
        color = stage_colors.get(p.stage, default_color)
        table.add_row(
            p.slug,
            p.topic[:60],
            f"[{color}]{p.stage.value}[/{color}]",
            p.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@app.command()
def shorts(
    slug: str = typer.Argument(..., help="Project slug"),
    pick: str = typer.Option(
        ..., "--pick", "-p",
        help="Comma-separated candidate ranks to render, e.g. 1,3",
    ),
) -> None:
    """Render chosen short candidates for a project.

    Run `vidmaxx run <slug>` first — it stops after scoring and prints the
    candidate list. Then call this with the ranks you want.
    """
    settings = _load_settings()
    state_mgr = _get_state_mgr(settings)

    if not state_mgr.exists(slug):
        console.print(f"[red]No project found with slug '{slug}'.[/red]")
        raise typer.Exit(1)

    project = state_mgr.load(slug)
    if project.stage != PipelineStage.SHORTS_RENDER:
        console.print(
            f"[yellow]Project is at stage '{project.stage.value}', not 'shorts_render'.[/yellow]"
        )
        if project.stage == PipelineStage.SHORTS_SELECT:
            console.print("Run [bold]vidmaxx run " + slug + "[/bold] first to score candidates.")
        raise typer.Exit(1)

    try:
        picks = [int(x.strip()) for x in pick.split(",") if x.strip()]
    except ValueError:
        console.print(f"[red]--pick must be comma-separated integers, e.g. 1,3[/red]")
        raise typer.Exit(1)

    if not picks:
        console.print("[red]No valid indices in --pick.[/red]")
        raise typer.Exit(1)

    console.print(f"\nRendering shorts {picks} for [bold]{slug}[/bold]...\n")
    short_paths = asyncio.run(s08.run_render(slug, state_mgr, settings, picks))

    console.print("\nShorts rendered:")
    for p in short_paths:
        size_mb = round(p.stat().st_size / 1e6, 1)
        console.print(f"  {p}  ({size_mb} MB)")

    console.print(
        f"\nProject is now at 'publish'. Run [bold]vidmaxx run {slug}[/bold] "
        "if you have YouTube configured."
    )


@app.command()
def status(
    slug: str = typer.Argument(..., help="Project slug"),
) -> None:
    """Show detailed status for one project."""
    settings = _load_settings()
    state_mgr = _get_state_mgr(settings)

    if not state_mgr.exists(slug):
        console.print(f"[red]No project found with slug '{slug}'.[/red]")
        raise typer.Exit(1)

    project = state_mgr.load(slug)
    paths = state_mgr.paths(slug)

    console.print(f"[bold]{project.slug}[/bold]")
    console.print(f"  Topic:   {project.topic}")
    console.print(f"  Stage:   {project.stage.value}")
    console.print(f"  Created: {project.created_at.strftime('%Y-%m-%d %H:%M')}")
    console.print(f"  Updated: {project.updated_at.strftime('%Y-%m-%d %H:%M')}")

    if project.error:
        console.print(f"  [red]Error:[/red]   {project.error}")

    # Show which output files exist.
    checks = [
        ("verified_fact_sheet.json", paths.verified_fact_sheet),
        ("script.json",              paths.script),
        ("alignment.json",         paths.alignment_json),
        ("timeline.json",          paths.timeline_json),
        ("video.mp4",              paths.final_video),
        ("shorts/candidates.json", paths.shorts_dir / "candidates.json"),
        ("out/publish.json",       paths.publish_json),
    ]
    console.print("\n  Outputs:")
    for label, path in checks:
        mark = "[green]✓[/green]" if path.exists() else "[dim]·[/dim]"
        console.print(f"    {mark} {label}")

    if paths.shorts_dir.exists():
        shorts = sorted(paths.shorts_dir.glob("short_*.mp4"))
        for s in shorts:
            size_mb = round(s.stat().st_size / 1e6, 1)
            console.print(f"    [green]✓[/green] {s.name} ({size_mb} MB)")


@app.command()
def reset(
    slug: str = typer.Argument(..., help="Project slug"),
    to: str = typer.Option(..., "--to", help="Stage to reset to, e.g. 'script'"),
) -> None:
    """Reset a project to an earlier stage so it reruns from there."""
    settings = _load_settings()
    state_mgr = _get_state_mgr(settings)

    if not state_mgr.exists(slug):
        console.print(f"[red]No project found with slug '{slug}'.[/red]")
        raise typer.Exit(1)

    try:
        target = PipelineStage(to.upper())
    except ValueError:
        valid = [s.value for s in PipelineStage]
        console.print(f"[red]Unknown stage '{to}'.[/red] Valid: {valid}")
        raise typer.Exit(1)

    project = state_mgr.reset(slug, target)
    console.print(f"[bold]{slug}[/bold] reset to [bold]{project.stage.value}[/bold]")


# ---------------------------------------------------------------------------
# Shared async runner
# ---------------------------------------------------------------------------

def _run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
    stop_before_str: Optional[str],
) -> None:
    stop_before: Optional[PipelineStage] = None
    if stop_before_str:
        try:
            stop_before = PipelineStage(stop_before_str.upper())
        except ValueError:
            valid = [s.value for s in PipelineStage]
            console.print(f"[red]Unknown stage '{stop_before_str}'.[/red] Valid: {valid}")
            raise typer.Exit(1)

    project = state_mgr.load(slug)
    console.print(
        f"\nRunning [bold]{slug}[/bold]  "
        f"({project.topic[:60]})  "
        f"from stage: {project.stage.value}\n"
    )

    asyncio.run(run_pipeline(slug, state_mgr, cache, settings, stop_before))


if __name__ == "__main__":
    app()
