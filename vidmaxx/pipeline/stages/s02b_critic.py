"""
Stage 2b — Critic

Reads:   paths.outline            (outline.json)
         paths.chapter_prose_dir/ ({idx:02d}.txt per chapter)
         paths.verified_fact_sheet (for ungrounded-claim checking)
Writes:  paths.critic_report      (critic_report.json)
Advances: CRITIC → OPTIMIZER. CRITICAL flags do not halt; they are passed to
          Optimizer for removal/grounded rewrite.

Deterministic checks (no LLM — regex + word count):
  MAX_18_WORDS        — CRITICAL  — sentence has >18 words
  NO_RHETORICAL_Q     — CRITICAL  — sentence ends with '?'
  NO_FILLER_TRANSITION — WARNING  — banned transition phrases
  FOUR_PURPOSE (pre)  — WARNING   — sentence fails all four deterministic signals
                                    (number, named entity, ellipsis, contradiction word)

LLM checks (per chapter, batched):
  FOUR_PURPOSE (final) — WARNING  — LLM confirms sentence truly has no purpose
  UNGROUNDED_CLAIM    — CRITICAL  — factual assertion not traceable to verified claims

CRITICAL flag policy: if any CRITICAL flag exists, the report records them and
the pipeline continues. Optimizer must remove unsupported claims or rewrite them
using verified facts only.

WARNING flags proceed to OPTIMIZER for surgical rewriting.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict

import structlog

from vidmaxx.config.constants import SCRIPT_DECOMPOSE_MODEL
from vidmaxx.config.settings import Settings
from vidmaxx.models.factsheet import CriticFlag, CriticReport, VerifiedFactSheet
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Outline
from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.beat_doctrine import critic_doctrine, load_master_beat_analysis
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Deterministic pattern checks
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = re.compile(
    r"\b(as we (discussed|mentioned|said)|moving on|now let'?s|"
    r"in this (chapter|section|video)|interestingly enough|"
    r"that'?s (actually |really )?(fascinating|interesting|important)|"
    r"it'?s worth noting|having said that|with that (said|in mind))\b",
    re.IGNORECASE,
)

_CONTRADICTION_WORDS = frozenset(
    ["but", "except", "however", "actually", "despite", "contrary", "wrong", "not"]
)

_NUMBER_RE = re.compile(r"\d")
_NAMED_ENTITY_RE = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+")


def _has_purpose(sentence: str) -> bool:
    """Deterministic FOUR_PURPOSE signal check.

    Returns True if the sentence has at least one purpose signal:
      - contains a digit (number/date/amount)
      - contains a two-word proper noun (named entity heuristic)
      - contains an ellipsis (revelation technique)
      - contains a contradiction/concession word
    """
    if _NUMBER_RE.search(sentence):
        return True
    if _NAMED_ENTITY_RE.search(sentence):
        return True
    if "..." in sentence:
        return True
    lower = sentence.lower()
    if any(w in lower.split() for w in _CONTRADICTION_WORDS):
        return True
    return False


def _check_word_count(sentence_id: str, text: str) -> CriticFlag | None:
    words = text.split()
    if len(words) > 18:
        return CriticFlag(
            sentence_id=sentence_id,
            severity="CRITICAL",
            reason=f"Sentence is {len(words)} words — max is 18.",
            rule_violated="MAX_18_WORDS",
        )
    return None


def _check_rhetorical_q(sentence_id: str, text: str) -> CriticFlag | None:
    if text.strip().endswith("?"):
        return CriticFlag(
            sentence_id=sentence_id,
            severity="CRITICAL",
            reason="Sentence is a question — rhetorical questions are banned.",
            rule_violated="NO_RHETORICAL_Q",
        )
    return None


def _check_filler(sentence_id: str, text: str) -> CriticFlag | None:
    if _FILLER_PATTERNS.search(text):
        return CriticFlag(
            sentence_id=sentence_id,
            severity="WARNING",
            reason="Sentence contains a banned filler or transition phrase.",
            rule_violated="NO_FILLER_TRANSITION",
        )
    return None


def _check_four_purpose_deterministic(sentence_id: str, text: str) -> CriticFlag | None:
    """WARNING pre-flag for sentences that fail all deterministic purpose signals.

    These go to the LLM for a final call. The LLM may clear the flag if it
    identifies a non-obvious purpose (conceptual contradiction, implicit reveal).
    """
    if not _has_purpose(text):
        return CriticFlag(
            sentence_id=sentence_id,
            severity="WARNING",
            reason="No number, named entity, ellipsis, or contradiction word detected.",
            rule_violated="FOUR_PURPOSE",
        )
    return None


# ---------------------------------------------------------------------------
# LLM checks
# ---------------------------------------------------------------------------

_LLM_CRITIC_SYSTEM = """\
You are a strict YouTube script critic. You receive narration sentences with IDs.

Evaluate each sentence against two rules only:

  FOUR_PURPOSE: Confirm or clear pre-flagged sentences. A sentence passes if it:
    - Reveals new information the viewer didn't know
    - Contradicts something just said
    - Raises a question the very next sentence answers
    - Drops a specific number, percentage, named entity, or dollar amount
    Clear the flag (do NOT emit it) if the sentence does serve a purpose — even
    non-obviously. Only emit if the sentence truly just summarizes or restates
    with zero informational movement.

  UNGROUNDED_CLAIM: A factual assertion in the narration not traceable to any of
    the provided verified claims. Only flag assertions of fact — not style or
    structure. Severity: CRITICAL.

Output a JSON array of flag objects. For each flag:
{
  "sentence_id": "<id>",
  "severity": "<CRITICAL|WARNING>",
  "reason": "<one sentence explanation>",
  "rule_violated": "<FOUR_PURPOSE|UNGROUNDED_CLAIM>",
  "ungrounded_claim": "<verbatim text — only for UNGROUNDED_CLAIM, else null>"
}

If a sentence passes both rules, do NOT include it in the output.
If no sentences fail, output an empty array [].
Output only the JSON array — no prose, no markdown fences.\
"""


def _split_sentences(prose: str) -> list[tuple[str, str]]:
    sentences = re.split(r'(?<=[.!?])\s+', prose.strip())
    return [(f"s{i:02d}", s.strip()) for i, s in enumerate(sentences) if s.strip()]


async def _llm_critic_pass(
    llm: LLMClient,
    chapter_index: int,
    all_sentences: list[tuple[str, str]],
    pre_flagged_sids: set[str],
    verified_claims: list,
    doctrine_text: str = "",
) -> list[CriticFlag]:
    """LLM reviews all sentences for UNGROUNDED_CLAIM + confirms/clears FOUR_PURPOSE pre-flags."""
    if not all_sentences:
        return []

    # Mark which sentences were pre-flagged so the LLM knows to adjudicate them
    sentences_text = "\n".join(
        f"{sid}{' [PRE-FLAGGED FOUR_PURPOSE]' if sid in pre_flagged_sids else ''}: {text}"
        for sid, text in all_sentences
    )
    claims_text = "\n".join(
        f"- {getattr(c, 'claim', None) or getattr(c, 'text', '')}"
        for c in verified_claims[:20]
    )
    prompt = (
        f"Chapter index: {chapter_index}\n\n"
        f"{doctrine_text}\n\n"
        f"Verified research claims (only these facts are grounded):\n{claims_text}\n\n"
        f"Narration sentences:\n{sentences_text}"
    )

    try:
        data = await llm.complete_json(
            model=SCRIPT_DECOMPOSE_MODEL,
            system=_LLM_CRITIC_SYSTEM,
            prompt=prompt,
            max_tokens=4096,
        )
        if not isinstance(data, list):
            return []
        return [CriticFlag(**f) for f in data if isinstance(f, dict)]
    except Exception as exc:
        log.warning("critic_llm_failed", chapter=chapter_index, error=str(exc))
        return []


async def _critic_one_chapter(
    llm: LLMClient,
    chapter_index: int,
    prose: str,
    verified_claims: list,
    doctrine_text: str = "",
) -> list[CriticFlag]:
    raw_sentences = _split_sentences(prose)
    sentences = [
        (f"ch{chapter_index:02d}_{sid}", text)
        for sid, text in raw_sentences
    ]

    flags: list[CriticFlag] = []
    pre_flagged_sids: set[str] = set()

    # --- Deterministic checks ---
    for sid, text in sentences:
        f = _check_word_count(sid, text)
        if f:
            flags.append(f)

        f = _check_rhetorical_q(sid, text)
        if f:
            flags.append(f)

        f = _check_filler(sid, text)
        if f:
            flags.append(f)

        f = _check_four_purpose_deterministic(sid, text)
        if f:
            # Don't append yet — LLM adjudicates; track for prompt marking
            pre_flagged_sids.add(sid)

    # --- LLM checks (FOUR_PURPOSE adjudication + UNGROUNDED_CLAIM) ---
    llm_flags = await _llm_critic_pass(
        llm,
        chapter_index,
        sentences,
        pre_flagged_sids,
        verified_claims,
        doctrine_text,
    )

    # LLM output is authoritative for FOUR_PURPOSE — include only what LLM confirms
    # For UNGROUNDED_CLAIM, include everything the LLM flagged
    for f in llm_flags:
        flags.append(f)

    return flags


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> CriticReport:
    paths = state_mgr.paths(slug)
    outline = Outline.model_validate_json(paths.outline.read_text())
    fact_sheet = VerifiedFactSheet.model_validate_json(paths.verified_fact_sheet.read_text())
    log.info("stage_critic_start", slug=slug, chapters=len(outline.chapters))

    llm = LLMClient(api_key=settings.google_api_key, cache=cache)
    doctrine_text = critic_doctrine(load_master_beat_analysis())

    with state_mgr.run_stage(slug, PipelineStage.CRITIC) as paths:
        tasks = []
        for chapter in outline.chapters:
            prose_path = paths.chapter_prose_dir / f"{chapter.index:02d}.txt"
            if not prose_path.exists():
                raise FileNotFoundError(
                    f"Chapter prose missing: {prose_path}. Re-run s02a_architect."
                )
            prose = prose_path.read_text()
            tasks.append(_critic_one_chapter(llm, chapter.index, prose, fact_sheet.claims, doctrine_text))

        chapter_flag_lists = await asyncio.gather(*tasks)
        all_flags = [f for flags in chapter_flag_lists for f in flags]

        has_critical = any(f.severity == "CRITICAL" for f in all_flags)
        report = CriticReport(has_critical=has_critical, flags=all_flags)
        paths.critic_report.write_text(report.model_dump_json(indent=2))

        critical_count = len(report.critical_flags)
        warning_count = len(report.warning_flags)
        log.info(
            "stage_critic_done",
            slug=slug,
            total_flags=len(all_flags),
            critical=critical_count,
            warnings=warning_count,
        )

        if has_critical:
            flagged_ids = [f.sentence_id for f in report.critical_flags]
            ungrounded = [
                f"{f.sentence_id}: \"{f.ungrounded_claim}\""
                for f in report.critical_flags
                if f.rule_violated == "UNGROUNDED_CLAIM" and f.ungrounded_claim
            ]
            structural = [
                f"{f.sentence_id} ({f.rule_violated})"
                for f in report.critical_flags
                if f.rule_violated != "UNGROUNDED_CLAIM"
            ]

            log.warning(
                "critic_critical_flags_continue_to_optimizer",
                slug=slug,
                critical=critical_count,
                flagged_ids=flagged_ids,
                ungrounded=ungrounded[:5],
                structural=structural[:5],
            )

    return report
