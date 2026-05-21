"""
Stage 2a — Architect + Chapter Writers

Reads:   paths.verified_fact_sheet  (frozen=True, hash verified against SQLite)
                                      fact_sheet.brief carries auto_brief.json fields
Writes:  paths.outline               (outline.json)
         paths.chapter_prose_dir/    ({idx:02d}.txt per chapter)
Advances: ARCHITECT → CRITIC

Pass 1 — Architect (Gemini 2.5 Pro + extended thinking)
  Reads verified factsheet, verifies SHA-256 hash matches SQLite, then maps
  claims onto the nine-scene structure. Cached by factsheet hash + brief hash.

Pass 2 — Chapter writers (Gemini 2.5 Pro, all chapters in parallel)
  Each chapter's prose is written from its outline beats. Saved as individual
  .txt files for the critic to read and annotate sentence-by-sentence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import structlog

from vidmaxx.config.constants import (
    ARCHITECT_THINKING_BUDGET,
    SCENE_DURATIONS_SEC,
    SCENE_LABELS,
    SCRIPT_ARCHITECT_MODEL,
    SCRIPT_CHAPTER_MODEL,
    TARGET_VIDEO_DURATION_SEC,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.factsheet import VerifiedFactSheet
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import ChapterBeat, ChapterOutline, Outline
from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.llm.prompts import architect as architect_prompts
from vidmaxx.services.llm.prompts import chapter_writer as chapter_writer_prompts
from vidmaxx.services.beat_doctrine import (
    architect_doctrine,
    chapter_doctrine,
    doctrine_hash,
    load_master_beat_analysis,
)
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

_ARCHITECT_CACHE_PREFIX = "architect_outline_v4_master_beat"

_SCENE_ENERGY = {
    "The Strategic Hook": "high",
    "The Mechanism Flow": "medium",
    "The Incongruity Drop": "high",
    "The Dependency Link": "medium",
    "The Human Anchor": "medium",
    "The Causal Collapse": "high",
    "The Institutional Facade": "medium",
    "The Shadow-Truth": "high",
    "The Synthesis": "medium",
}

_MIN_BEATS = {
    "The Strategic Hook": 4,
    "The Mechanism Flow": 5,
    "The Incongruity Drop": 4,
    "The Dependency Link": 5,
    "The Human Anchor": 5,
    "The Causal Collapse": 8,
    "The Institutional Facade": 5,
    "The Shadow-Truth": 4,
    "The Synthesis": 3,
}


def _repair_outline(outline: Outline, fact_sheet: VerifiedFactSheet) -> Outline:
    """Keep the run moving if the architect skips or mis-shapes a scene."""
    chapters_by_index = {ch.index: ch for ch in outline.chapters}
    facts = [c.claim for c in fact_sheet.claims] or [fact_sheet.brief.get("cold_fact", "Use the verified brief anchor.")]
    repaired: list[ChapterOutline] = []

    for idx, title in enumerate(SCENE_LABELS):
        chapter = chapters_by_index.get(idx)
        if chapter is None:
            chapter = ChapterOutline(
                index=idx,
                title=title,
                target_duration_sec=SCENE_DURATIONS_SEC[idx],
                energy=_SCENE_ENERGY[title],
                key_beats=[],
            )
        chapter = chapter.model_copy(update={
            "index": idx,
            "title": title,
            "target_duration_sec": SCENE_DURATIONS_SEC[idx],
            "energy": _SCENE_ENERGY[title],
        })
        beats = list(chapter.key_beats)
        min_beats = _MIN_BEATS[title]
        while len(beats) < min_beats:
            fact = facts[len(beats) % len(facts)]
            beats.append(ChapterBeat(
                description=(
                    f"Develop this scene's required narrative function using this verified fact: {fact}"
                ),
                energy=_SCENE_ENERGY[title],
            ))
        chapter = chapter.model_copy(update={"key_beats": beats})
        repaired.append(chapter)

    if len(outline.chapters) != 9 or any(
        ch.title != SCENE_LABELS[ch.index] or ch.target_duration_sec != SCENE_DURATIONS_SEC[ch.index]
        for ch in repaired
    ):
        log.warning("architect_outline_repaired", chapters_in=len(outline.chapters), chapters_out=len(repaired))

    return Outline(
        topic=outline.topic or fact_sheet.topic,
        target_duration_sec=TARGET_VIDEO_DURATION_SEC,
        chapters=repaired,
    )


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> Outline:
    paths = state_mgr.paths(slug)
    fact_sheet = VerifiedFactSheet.model_validate_json(paths.verified_fact_sheet.read_text())

    # Verify hash hasn't been tampered with since freeze
    stored_hash = state_mgr.get_factsheet_hash(slug)
    if stored_hash is None:
        raise RuntimeError(
            "Fact sheet has not been frozen — run s01c_freeze first."
        )
    if fact_sheet.file_hash != stored_hash:
        raise RuntimeError(
            f"Fact sheet hash mismatch: file has {fact_sheet.file_hash!r}, "
            f"SQLite has {stored_hash!r}. Fact sheet was modified after freeze."
        )
    if not fact_sheet.frozen:
        raise RuntimeError("Fact sheet frozen=False — s01c did not complete successfully.")

    log.info("stage_architect_start", slug=slug, claims=len(fact_sheet.claims))

    llm = LLMClient(api_key=settings.google_api_key, cache=cache)

    with state_mgr.run_stage(slug, PipelineStage.ARCHITECT) as paths:
        # Brief comes from fact_sheet.brief (embedded by s01b from auto_brief.json).
        # This is always populated when the grounding flow ran.
        # Log which source we're using for debugging.
        doctrine = load_master_beat_analysis()
        if doctrine:
            log.info("architect_master_beat_doctrine_loaded", hash=doctrine_hash(doctrine))
        else:
            log.warning("architect_master_beat_doctrine_missing", path="sample/master_beat_analysis.json")

        if fact_sheet.brief:
            log.info("architect_brief_from_factsheet", fields=list(fact_sheet.brief.keys()))
        elif paths.brief_json.exists():
            log.info("architect_brief_from_brief_json")
        else:
            log.warning("architect_no_brief_available", slug=slug)

        outline = await _run_architect(llm, cache, fact_sheet, brief=None, doctrine=doctrine)
        paths.outline.write_text(outline.model_dump_json(indent=2))
        log.info("architect_outline_done", chapters=len(outline.chapters))

        proses = await _run_chapter_writers(llm, outline, fact_sheet, doctrine)

        paths.chapter_prose_dir.mkdir(parents=True, exist_ok=True)
        for i, prose in enumerate(proses):
            (paths.chapter_prose_dir / f"{i:02d}.txt").write_text(prose)
        log.info("architect_chapters_written", count=len(proses))

    return outline


async def _run_architect(
    llm: LLMClient,
    cache: PipelineCache,
    fact_sheet: VerifiedFactSheet,
    brief: dict | None = None,
    doctrine: dict | None = None,
) -> Outline:
    # Cache key: factsheet hash already covers fact_sheet.brief (embedded in the JSON).
    # The brief argument is kept for the rare case of an externally passed brief,
    # but normally brief=None and fact_sheet.brief carries the content.
    brief_hash = hashlib.sha256(
        json.dumps(brief, sort_keys=True).encode() if brief else b""
    ).hexdigest()[:8]
    cache_key = (
        f"{_ARCHITECT_CACHE_PREFIX}:"
        + fact_sheet.file_hash
        + f":{brief_hash}"
        + f":{doctrine_hash(doctrine or {})}"
    )

    cached = cache.get(cache_key)
    if cached is not None:
        log.debug("architect_cache_hit")
        return Outline.model_validate_json(cached)

    # brief=None — architect_prompts.build() reads fact_sheet.brief directly
    system, prompt = architect_prompts.build(
        fact_sheet,
        TARGET_VIDEO_DURATION_SEC,
        brief,
        beat_doctrine=architect_doctrine(doctrine or {}),
    )
    data = await llm.complete_json_with_thinking(
        model=SCRIPT_ARCHITECT_MODEL,
        system=system,
        prompt=prompt,
        thinking_budget=ARCHITECT_THINKING_BUDGET,
        max_tokens=16384,
    )
    outline = _repair_outline(Outline.model_validate(data), fact_sheet)
    cache.set(cache_key, outline.model_dump_json())
    return outline


async def _write_one_chapter(
    llm: LLMClient,
    chapter: ChapterOutline,
    outline: Outline,
    fact_sheet: VerifiedFactSheet,
    doctrine: dict | None = None,
) -> str:
    system, prompt = chapter_writer_prompts.build(
        chapter,
        outline,
        fact_sheet,
        beat_doctrine=chapter_doctrine(doctrine or {}, chapter.index),
    )
    prose = await llm.complete(
        model=SCRIPT_CHAPTER_MODEL,
        system=system,
        prompt=prompt,
        max_tokens=8192,
    )
    log.debug("chapter_written", index=chapter.index, chars=len(prose))
    return prose


async def _run_chapter_writers(
    llm: LLMClient,
    outline: Outline,
    fact_sheet: VerifiedFactSheet,
    doctrine: dict | None = None,
) -> list[str]:
    tasks = [
        _write_one_chapter(llm, ch, outline, fact_sheet, doctrine)
        for ch in outline.chapters
    ]
    return await asyncio.gather(*tasks)
