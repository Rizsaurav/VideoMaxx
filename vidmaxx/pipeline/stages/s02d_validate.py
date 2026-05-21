"""
Stage 2d — Validate (decomposer + dinner-fact anchor check)

Reads:   paths.outline                  (outline.json)
         paths.chapter_prose_final_dir/ ({idx:02d}.txt per chapter)
         paths.verified_fact_sheet      (for dinner_fact_id)
Writes:  paths.script                   (script.json)
Advances: VALIDATE → ASSETS

Runs the decomposer on each chapter's final prose (same logic as the old
s02_script.py Pass 3c). Then:
  - Verifies the dinner_fact_id claim's key terms appear in chapter 0
    (The Strategic Hook).
  - Produces a ScriptDiff summary for audit.
  - Writes script.json.
"""

from __future__ import annotations

import asyncio
import re

import structlog
from pydantic import ValidationError

from vidmaxx.config.constants import (
    SCRIPT_CHAPTER_MODEL,
    SCRIPT_DECOMPOSE_MODEL,
    TARGET_VIDEO_DURATION_SEC,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.factsheet import ScriptDiff, VerifiedFactSheet
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Chapter, ChapterOutline, Outline, Script
from vidmaxx.models.sentence import Sentence
from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.llm.prompts import decomposer as decomposer_prompts
from vidmaxx.services.beat_doctrine import load_master_beat_analysis, runtime_doctrine
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

_PROSE_SPLIT_THRESHOLD = 1000
_MIN_TOTAL_DURATION_SEC = 990
_MAX_TOTAL_DURATION_SEC = 1110
_MAX_RUNTIME_REPAIR_ROUNDS = 2
_MAX_SENTENCE_WORDS = 18

_RUNTIME_EXPAND_SYSTEM = """\
You are the runtime producer for a narrative YouTube script.

Your job is to expand one scene so it reaches its target duration without filler.

Rules:
- Preserve the existing scene's voice and facts.
- Add only new mechanism, counter-argument, data contrast, or personal consequence.
- Use only the verified claims provided.
- No generic transitions, no motivational filler, no recap.
- Every added sentence must reveal, contradict, answer a setup, or drop a number/entity.
- Maximum 18 words per sentence.
- Output the full revised scene prose only. No notes, no markdown.\
"""


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_long_sentence_text(text: str, max_words: int = _MAX_SENTENCE_WORDS) -> list[str]:
    """Split an overlong decomposer sentence into <= max_words chunks.

    The script has a hard 18-word cap. If the decomposer fails to split prose
    cleanly, this deterministic repair keeps `script.json` compliant without
    another LLM call.
    """
    words = re.findall(r"\S+", text.strip())
    if len(words) <= max_words:
        return [text.strip()]

    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if not chunk:
            continue
        if chunk[-1] not in ".!?":
            chunk += "."
        chunks.append(chunk)
    return chunks


def _enforce_sentence_word_cap(raw_items: list[dict]) -> list[dict]:
    """Return decomposer items with every text field capped at 18 words."""
    capped: list[dict] = []
    for item in raw_items:
        text = str(item.get("text", "")).strip()
        chunks = _split_long_sentence_text(text)
        if len(chunks) == 1:
            capped.append(item)
            continue

        original_duration = float(item.get("expected_duration_sec") or 3.0)
        duration_each = max(0.5, original_duration / len(chunks))
        for chunk in chunks:
            clone = dict(item)
            clone["text"] = chunk
            clone["expected_duration_sec"] = duration_each
            clone["pause_before_ms"] = 0
            capped.append(clone)
    return capped


def _chapter_duration(chapter: Chapter) -> float:
    return sum(s.expected_duration_sec for s in chapter.sentences)


def _chapter_duration_map(chapters: list[Chapter]) -> dict[int, float]:
    return {ch.index: _chapter_duration(ch) for ch in chapters}


def _split_prose_at_sentence(prose: str) -> tuple[str, str]:
    mid = len(prose) // 2
    for i in range(mid, 0, -1):
        if prose[i] in ".!?" and i + 1 < len(prose) and prose[i + 1] in " \n":
            return prose[: i + 1].strip(), prose[i + 1 :].strip()
    return prose[:mid], prose[mid:]


async def _call_decomposer(llm: LLMClient, chapter: ChapterOutline, prose: str) -> list:
    system, prompt = decomposer_prompts.build(prose, chapter)
    data = await llm.complete_json(
        model=SCRIPT_DECOMPOSE_MODEL,
        system=system,
        prompt=prompt,
        max_tokens=8192,
    )
    if not isinstance(data, list):
        raise ValueError(
            f"Decomposer returned {type(data).__name__} for chapter {chapter.index}, expected list"
        )
    return data


async def _decompose_one_chapter(llm: LLMClient, chapter: ChapterOutline, prose: str) -> Chapter:
    if len(prose) > _PROSE_SPLIT_THRESHOLD:
        first_prose, second_prose = _split_prose_at_sentence(prose)
        first_data, second_data = await asyncio.gather(
            _call_decomposer(llm, chapter, first_prose),
            _call_decomposer(llm, chapter, second_prose),
        )
        raw_items = first_data + second_data
    else:
        raw_items = await _call_decomposer(llm, chapter, prose)

    capped_items = _enforce_sentence_word_cap(raw_items)
    if len(capped_items) != len(raw_items):
        log.warning(
            "validate_sentence_word_cap_repaired",
            chapter=chapter.index,
            before=len(raw_items),
            after=len(capped_items),
            max_words=_MAX_SENTENCE_WORDS,
        )

    sentences: list[Sentence] = []
    for i, item in enumerate(capped_items):
        item["id"] = f"ch{chapter.index:02d}_s{i:02d}"
        try:
            sentences.append(Sentence.model_validate(item))
        except ValidationError as exc:
            raise ValueError(
                f"Sentence validation failed — chapter {chapter.index}, item {i}: {exc}"
            ) from exc

    log.debug("chapter_decomposed", index=chapter.index, sentences=len(sentences))
    return Chapter(
        index=chapter.index,
        title=chapter.title,
        energy=chapter.energy,
        sentences=sentences,
    )


async def _decompose_all(
    llm: LLMClient,
    outline: Outline,
    proses_by_index: dict[int, str],
) -> list[Chapter]:
    tasks = [
        _decompose_one_chapter(llm, chapter, proses_by_index[chapter.index])
        for chapter in outline.chapters
    ]
    chapters = await asyncio.gather(*tasks)
    return sorted(chapters, key=lambda c: c.index)


async def _expand_chapter_runtime(
    llm: LLMClient,
    chapter_outline: ChapterOutline,
    current_prose: str,
    current_duration: float,
    verified_claims: list,
) -> str:
    deficit = max(0.0, chapter_outline.target_duration_sec - current_duration)
    target_extra_words = min(700, max(120, int(deficit * 2.15)))
    claims_text = "\n".join(
        f"- [{c.status}] {c.claim} (source: {c.source_url})"
        for c in verified_claims
        if c.status in ("CONFIRMED", "PLAUSIBLE", "CONTESTED")
    )
    doctrine_text = runtime_doctrine(load_master_beat_analysis())
    prompt = f"""\
Scene title: {chapter_outline.title}
Scene target duration: {chapter_outline.target_duration_sec:.1f} sec
Current estimated duration: {current_duration:.1f} sec
Runtime deficit: {deficit:.1f} sec
Add approximately {target_extra_words} words.

Scene beats:
{chr(10).join(f"- {b.description}" for b in chapter_outline.key_beats)}

Verified claims available:
{claims_text}

{doctrine_text}

Current scene prose:
{current_prose}

Rewrite the full scene with more causal depth and narrative tension.
Do not pad. Add mechanism, counter-argument, data contrast, or consequence only.\
"""
    return (await llm.complete(
        model=SCRIPT_CHAPTER_MODEL,
        system=_RUNTIME_EXPAND_SYSTEM,
        prompt=prompt,
        max_tokens=8192,
    )).strip()


async def _repair_runtime_if_needed(
    llm: LLMClient,
    outline: Outline,
    fact_sheet: VerifiedFactSheet,
    proses_by_index: dict[int, str],
    chapters: list[Chapter],
    paths,
    slug: str,
) -> tuple[dict[int, str], list[Chapter]]:
    for round_idx in range(_MAX_RUNTIME_REPAIR_ROUNDS):
        total = sum(_chapter_duration(ch) for ch in chapters)
        duration_by_chapter = _chapter_duration_map(chapters)
        log.info(
            "runtime_check",
            slug=slug,
            round=round_idx,
            total_duration=round(total, 1),
            target=TARGET_VIDEO_DURATION_SEC,
            by_chapter={k: round(v, 1) for k, v in duration_by_chapter.items()},
        )

        if _MIN_TOTAL_DURATION_SEC <= total <= _MAX_TOTAL_DURATION_SEC:
            return proses_by_index, chapters

        if total > _MAX_TOTAL_DURATION_SEC:
            log.warning("runtime_over_target_continue", slug=slug, total_duration=round(total, 1))
            return proses_by_index, chapters

        deficits = []
        for chapter_outline in outline.chapters:
            if chapter_outline.title in {"The Strategic Hook", "The Synthesis", "COLD FACT", "HARD EXIT"}:
                continue
            current = duration_by_chapter.get(chapter_outline.index, 0.0)
            deficit = chapter_outline.target_duration_sec - current
            if deficit > 20:
                deficits.append((deficit, chapter_outline))

        if not deficits:
            log.warning("runtime_under_target_no_expandable_chapters", slug=slug, total_duration=round(total, 1))
            return proses_by_index, chapters

        deficits.sort(reverse=True, key=lambda x: x[0])
        targets = [chapter for _, chapter in deficits[:3]]
        log.warning(
            "runtime_under_target_expanding",
            slug=slug,
            round=round_idx,
            total_duration=round(total, 1),
            targets=[ch.index for ch in targets],
        )

        expanded = await asyncio.gather(*[
            _expand_chapter_runtime(
                llm,
                chapter,
                proses_by_index[chapter.index],
                duration_by_chapter.get(chapter.index, 0.0),
                fact_sheet.claims,
            )
            for chapter in targets
        ])
        for chapter, prose in zip(targets, expanded):
            proses_by_index[chapter.index] = prose
            (paths.chapter_prose_final_dir / f"{chapter.index:02d}.txt").write_text(prose)

        chapters = await _decompose_all(llm, outline, proses_by_index)

    return proses_by_index, chapters


_NUMBER_RE = re.compile(r"\b\d[\d,\.%$]*\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _extract_key_terms(claim: str) -> list[str]:
    """Extract numbers, dollar amounts, percentages, and named entities from a claim.

    These are the specific data points that must survive into the narration verbatim.
    A generic paraphrase passes 60% word overlap; key term matching catches that case.
    """
    terms: list[str] = []
    terms.extend(_NUMBER_RE.findall(claim))
    terms.extend(_PROPER_NOUN_RE.findall(claim))
    return [t.lower() for t in terms]


def _check_dinner_fact_anchored(
    chapters: list[Chapter],
    fact_sheet: VerifiedFactSheet,
) -> tuple[bool, list[str]]:
    """Check that the dinner_fact's key terms (numbers + named entities) appear in ch00.

    Returns (anchored: bool, missing_terms: list[str]).
    A generic paraphrase that captures the theme but drops the specific number or
    entity fails this check — the dinner fact must be present with its specificity intact.
    """
    if not fact_sheet.dinner_fact_id:
        return False, ["dinner_fact_id is empty"]

    dinner_claim = next(
        (c for c in fact_sheet.claims if c.id == fact_sheet.dinner_fact_id), None
    )
    if not dinner_claim:
        return False, [f"claim {fact_sheet.dinner_fact_id!r} not found"]

    key_terms = _extract_key_terms(dinner_claim.claim)
    if not key_terms:
        # No numbers or named entities — fall back to checking the claim text as substring
        ch0_text = " ".join(
            s.text.lower() for ch in chapters if ch.index == 0 for s in ch.sentences
        )
        present = dinner_claim.claim.lower()[:40] in ch0_text
        return present, [] if present else ["no key terms and claim text not found in ch00"]

    ch0_text = " ".join(
        s.text.lower() for ch in chapters if ch.index == 0 for s in ch.sentences
    )
    missing = [t for t in key_terms if t not in ch0_text]
    return len(missing) == 0, missing


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> Script:
    paths = state_mgr.paths(slug)
    outline = Outline.model_validate_json(paths.outline.read_text())
    fact_sheet = VerifiedFactSheet.model_validate_json(paths.verified_fact_sheet.read_text())
    log.info("stage_validate_start", slug=slug, chapters=len(outline.chapters))

    llm = LLMClient(api_key=settings.google_api_key, cache=cache)

    with state_mgr.run_stage(slug, PipelineStage.VALIDATE) as paths:
        proses_by_index: dict[int, str] = {}
        for chapter in outline.chapters:
            prose_path = paths.chapter_prose_final_dir / f"{chapter.index:02d}.txt"
            if not prose_path.exists():
                raise FileNotFoundError(
                    f"Final chapter prose missing: {prose_path}. Re-run s02c_optimizer."
                )
            prose = prose_path.read_text()
            proses_by_index[chapter.index] = prose

        chapters = await _decompose_all(llm, outline, proses_by_index)
        proses_by_index, chapters = await _repair_runtime_if_needed(
            llm, outline, fact_sheet, proses_by_index, chapters, paths, slug
        )
        total_sentences = sum(len(c.sentences) for c in chapters)
        total_duration = sum(_chapter_duration(ch) for ch in chapters)

        # Dinner fact anchor check — key terms (numbers + named entities) must be present
        dinner_anchored, missing_terms = _check_dinner_fact_anchored(chapters, fact_sheet)
        if not dinner_anchored:
            log.warning(
                "validate_dinner_fact_not_anchored",
                slug=slug,
                dinner_fact_id=fact_sheet.dinner_fact_id,
                missing_terms=missing_terms,
            )

        # ScriptDiff summary
        diff = ScriptDiff(
            dinner_fact_anchored=dinner_anchored,
            total_sentences=total_sentences,
        )
        log.info(
            "validate_script_diff",
            total_sentences=total_sentences,
            total_duration=round(total_duration, 1),
            dinner_fact_anchored=dinner_anchored,
        )

        script = Script(
            topic=fact_sheet.topic,
            slug=slug,
            target_duration_sec=TARGET_VIDEO_DURATION_SEC,
            chapters=chapters,
        )
        paths.script.write_text(script.model_dump_json(indent=2))
        log.info(
            "stage_validate_done",
            slug=slug,
            sentences=total_sentences,
            total_duration=round(script.total_expected_duration_sec, 1),
        )

    return script
