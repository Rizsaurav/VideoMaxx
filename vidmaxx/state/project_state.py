"""
Project state machine backed by SQLite (WAL mode) + per-project state.json.

SQLite is the authoritative store for stage + metadata. state.json is a
human-readable snapshot written alongside every SQLite update — useful for
debugging without opening the DB.
"""

import json
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from vidmaxx.models.project import STAGE_ORDER, PipelineStage, Project, ProjectPaths


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    slug        TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    stage       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS factsheet_hashes (
    slug        TEXT PRIMARY KEY,
    hash        TEXT NOT NULL,
    frozen_at   TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Maps removed or renamed stage values from old DB rows to current equivalents.
_STAGE_MIGRATION: dict[str, str] = {
    "SCRIPT": "VALIDATE",   # old monolithic script stage → closest current equivalent
    "FAILED": "VALIDATE",   # FAILED was a synthetic state, not a real stage
}


def _row_to_project(row: sqlite3.Row) -> Project:
    raw = row["stage"].upper()
    stage_str = _STAGE_MIGRATION.get(raw, raw)
    try:
        stage = PipelineStage(stage_str)
    except ValueError:
        stage = PipelineStage.CREATED  # unknown legacy value — reset to start
    return Project(
        slug=row["slug"],
        topic=row["topic"],
        stage=stage,
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        error=row["error"],
    )


class ProjectStateManager:
    def __init__(self, projects_root: Path) -> None:
        self._root = projects_root
        self._db_path = projects_root / "registry.db"
        self._root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _paths(self, slug: str) -> ProjectPaths:
        return ProjectPaths(root=self._root / slug)

    def _write_state_json(self, project: Project) -> None:
        paths = self._paths(project.slug)
        paths.root.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename so a crash mid-write never leaves
        # a partial state.json.
        tmp = paths.root / ".state.json.tmp"
        tmp.write_text(project.model_dump_json(indent=2))
        tmp.rename(paths.state_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, topic: str, slug: str) -> Project:
        now = _now()
        project = Project(
            topic=topic,
            slug=slug,
            stage=PipelineStage.CREATED,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (slug, topic, stage, created_at, updated_at, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (slug, topic, project.stage.value, now, now, None),
            )
        self._write_state_json(project)
        return project

    def load(self, slug: str) -> Project:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE slug = ?", (slug,)
            ).fetchone()
        if row is None:
            raise KeyError(f"No project with slug '{slug}'")
        return _row_to_project(row)

    def advance(self, slug: str) -> Project:
        """Move the project to the next stage in STAGE_ORDER."""
        project = self.load(slug)
        if project.stage == PipelineStage.DONE:
            return project

        current_idx = STAGE_ORDER.index(project.stage)
        next_stage = STAGE_ORDER[current_idx + 1]
        now = _now()

        with self._connect() as conn:
            conn.execute(
                "UPDATE projects SET stage = ?, updated_at = ?, error = NULL WHERE slug = ?",
                (next_stage.value, now, slug),
            )
        project = project.model_copy(
            update={"stage": next_stage, "updated_at": datetime.fromisoformat(now), "error": None}
        )
        self._write_state_json(project)
        return project

    def fail(self, slug: str, error: str) -> Project:
        # Keep stage where it is so the next run retries the same stage.
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE projects SET updated_at = ?, error = ? WHERE slug = ?",
                (now, error, slug),
            )
        project = self.load(slug)
        self._write_state_json(project)
        return project

    def reset(self, slug: str, stage: PipelineStage) -> Project:
        """Force a project to a specific stage, clearing any error."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE projects SET stage = ?, updated_at = ?, error = NULL WHERE slug = ?",
                (stage.value, now, slug),
            )
        project = self.load(slug)
        self._write_state_json(project)
        return project

    def list_all(self) -> list[Project]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_project(r) for r in rows]

    def exists(self, slug: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM projects WHERE slug = ?", (slug,)
            ).fetchone()
        return row is not None

    def paths(self, slug: str) -> ProjectPaths:
        return self._paths(slug)

    def set_factsheet_hash(self, slug: str, hash_str: str) -> None:
        """Record the SHA-256 hash of the frozen VerifiedFactSheet."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO factsheet_hashes (slug, hash, frozen_at)
                VALUES (?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET hash = excluded.hash, frozen_at = excluded.frozen_at
                """,
                (slug, hash_str, _now()),
            )

    def get_factsheet_hash(self, slug: str) -> str | None:
        """Return the stored hash, or None if not yet frozen."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT hash FROM factsheet_hashes WHERE slug = ?", (slug,)
            ).fetchone()
        return row["hash"] if row else None

    @contextmanager
    def run_stage(self, slug: str, stage: PipelineStage):
        """
        Context manager for a pipeline stage. Advances on success, fails on exception.

        Usage:
            with state_manager.run_stage(slug, PipelineStage.RESEARCH):
                ...do research work...
        """
        project = self.load(slug)
        if project.stage != stage:
            raise RuntimeError(
                f"Expected stage {stage!r} but project is at {project.stage!r}"
            )
        try:
            yield self._paths(slug)
            self.advance(slug)
        except Exception as exc:
            self.fail(slug, str(exc))
            raise
