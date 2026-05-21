"""
Stage 2c — Optimizer

Reads:   paths.chapter_prose_dir/ ({idx:02d}.txt per chapter)
         paths.critic_report      (critic_report.json)
         paths.outline            (for chapter context)
Writes:  paths.chapter_prose_final_dir/ ({idx:02d}.txt per chapter)
Advances: OPTIMIZER → VALIDATE

Receives WARNING and CRITICAL flags from s02b. CRITICAL flags do not halt the run;
the Optimizer removes unsupported factual assertions or rewrites them using only
verified claims supplied in the prompt.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict

import structlog

from vidmaxx.config.constants import SCRIPT_CHAPTER_MODEL
from vidmaxx.config.settings import Settings
from vidmaxx.models.factsheet import CriticReport, VerifiedFactSheet
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Outline
from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.beat_doctrine import critic_doctrine, load_master_beat_analysis
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

_OPTIMIZER_SYSTEM = """\
You are a YouTube script editor fixing flagged narration without inventing facts.

You receive prose with flagged sentences marked inline. Rewrite ONLY the marked sentences.

Rules for rewrites:
- Fix the flagged rule violation.
- For UNGROUNDED_CLAIM, remove the unsupported factual assertion or replace it
  with one of the verified claims provided in the prompt.
- For MAX_18_WORDS, split or shorten the sentence without changing factual meaning.
- For NO_RHETORICAL_Q, turn the question into a direct statement.
- Do not change unflagged sentences
- Maintain the same intent unless the original sentence contains an unsupported claim.
- Every rewritten sentence must satisfy the four-purpose rule:
  → Reveal something new, OR contradict, OR raise a question the next sentence answers, OR drop a number/entity
- Maximum 18 words per sentence. Count them.
- No rhetorical questions (sentences ending with ?)
- No filler transitions ("as we discussed", "moving on", etc.)

Return the full prose with the marked sentences replaced.
Output only the corrected prose — no annotations, no markdown, no explanations.\
"""


def _flag_sids_for_chapter(chapter_index: int, report: CriticReport) -> dict[str, list]:
    """Return {sentence_id: [flags]} for all flags in a given chapter."""
    prefix = f"ch{chapter_index:02d}_"
    result: dict[str, list] = defaultdict(list)
    for flag in report.flags:
        if flag.sentence_id.startswith(prefix):
            result[flag.sentence_id].append(flag)
    return result


def _annotate_prose(prose: str, flagged_sids: dict[str, list]) -> str:
    if not flagged_sids:
        return prose

    raw = re.split(r'(?<=[.!?])\s+', prose.strip())
    out: list[str] = []
    # Extract chapter index from any sid key (e.g. "ch03_s01" → "03")
    chapter_idx_str = next(iter(flagged_sids)).split("_")[0][2:]
    for i, sentence in enumerate(raw):
        sid = f"ch{chapter_idx_str}_s{i:02d}"
        if sid in flagged_sids:
            rules = "; ".join(
                f"{f.severity}:{f.rule_violated}"
                + (f" ungrounded={f.ungrounded_claim}" if f.ungrounded_claim else "")
                for f in flagged_sids[sid]
            )
            out.append(f"[FIX: {rules}] {sentence}")
        else:
            out.append(sentence)
    return " ".join(out)


async def _optimize_chapter(
    llm: LLMClient,
    chapter_index: int,
    prose: str,
    report: CriticReport,
    outline: Outline,
    verified_claims: list,
    doctrine_text: str = "",
) -> str:
    flagged_sids = _flag_sids_for_chapter(chapter_index, report)
    if not flagged_sids:
        return prose

    chapter_outline = next((c for c in outline.chapters if c.index == chapter_index), None)
    context = f"Scene: {chapter_outline.title if chapter_outline else 'unknown'}"
    claims_text = "\n".join(
        f"- [{c.status}] {c.claim} (source: {c.source_url})"
        for c in verified_claims
        if c.status in ("CONFIRMED", "PLAUSIBLE")
    )

    annotated = _annotate_prose(prose, flagged_sids)
    prompt = (
        f"{context}\n"
        f"{doctrine_text}\n\n"
        f"Verified claims you may use for factual rewrites:\n{claims_text}\n\n"
        f"Sentences to fix: {len(flagged_sids)}\n\n"
        f"Prose with marked sentences:\n{annotated}"
    )

    rewritten = await llm.complete(
        model=SCRIPT_CHAPTER_MODEL,
        system=_OPTIMIZER_SYSTEM,
        prompt=prompt,
        max_tokens=4096,
    )
    log.debug("chapter_optimized", chapter=chapter_index, flags_fixed=len(flagged_sids))
    return rewritten.strip()


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> None:
    paths = state_mgr.paths(slug)
    outline = Outline.model_validate_json(paths.outline.read_text())
    report = CriticReport.model_validate_json(paths.critic_report.read_text())
    fact_sheet = VerifiedFactSheet.model_validate_json(paths.verified_fact_sheet.read_text())

    chapters_with_warnings = {
        f.sentence_id.split("_")[0][2:]
        for f in report.flags
    }
    log.info(
        "stage_optimizer_start",
        slug=slug,
        warning_flags=len(report.warning_flags),
        critical_flags=len(report.critical_flags),
        chapters_affected=len(chapters_with_warnings),
    )

    llm = LLMClient(api_key=settings.google_api_key, cache=cache)
    doctrine_text = critic_doctrine(load_master_beat_analysis())

    with state_mgr.run_stage(slug, PipelineStage.OPTIMIZER) as paths:
        paths.chapter_prose_final_dir.mkdir(parents=True, exist_ok=True)

        async def process_chapter(chapter):
            prose_path = paths.chapter_prose_dir / f"{chapter.index:02d}.txt"
            if not prose_path.exists():
                raise FileNotFoundError(
                    f"Chapter prose missing: {prose_path}. Re-run s02a_architect."
                )
            prose = prose_path.read_text()
            final_prose = await _optimize_chapter(
                llm,
                chapter.index,
                prose,
                report,
                outline,
                fact_sheet.claims,
                doctrine_text,
            )
            (paths.chapter_prose_final_dir / f"{chapter.index:02d}.txt").write_text(final_prose)

        await asyncio.gather(*[process_chapter(ch) for ch in outline.chapters])

        optimized = sum(1 for ch in outline.chapters if f"{ch.index:02d}" in chapters_with_warnings)
        log.info("stage_optimizer_done", slug=slug, optimized=optimized, copied=len(outline.chapters) - optimized)
