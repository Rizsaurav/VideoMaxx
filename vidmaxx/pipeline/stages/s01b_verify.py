"""
Stage 1b — Verify (Gemini 2.5 Flash + Google Grounding, parallel)

Reads:   paths.auto_brief_json  (auto_brief.json, written by s01a)
Writes:  paths.verified_fact_sheet  (verified_fact_sheet.json, frozen=False)
Advances: VERIFY → FREEZE

For each claim extracted from auto_brief.json, runs an adversarial verification
using Gemini 2.5 Flash + google_search grounding. Flash is used here (not Pro)
because verification is narrower than research — one claim at a time, ~512 tokens.

All verify calls run in parallel (asyncio.gather). Each call:
  1. Searches specifically for contradicting evidence, updated figures, misleading context
  2. Returns CONFIRMED | PLAUSIBLE | CONTESTED | UNVERIFIED + reasoning

Anchor repairs:
  - cold_fact is replaced with the best verified/available fallback if contested
  - dinner_fact is replaced with the best CONFIRMED fallback when possible

CONTESTED and UNVERIFIED claims are logged. CONTESTED claims are retained with
their low-trust status so downstream prompts know not to present them as settled.

Claims extracted from auto_brief.json (10 total):
  cl_cold_fact    ← brief["cold_fact"] + brief["cold_fact_source"]
  cl_dinner_fact  ← brief["dinner_fact"] + brief["dinner_fact_source"]
  cl_layer_0/1/2  ← brief["three_layers"][0..2] + matching three_layer_sources
  cl_support_0..4 ← brief["supporting_facts"][0..4] + matching supporting_fact_sources

After CONTESTED/UNVERIFIED filtering, freeze gate of >=8 CONFIRMED+PLAUSIBLE passes
comfortably even if 1-2 supporting facts are rejected.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime

import structlog
from google import genai
from google.genai import types

from vidmaxx.config.constants import (
    GEMINI_GCP_LOCATION,
    GEMINI_GROUNDING_MODEL,
    GEMINI_STRUCTURE_MODEL,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.factsheet import VerifiedClaim, VerifiedFactSheet
from vidmaxx.models.project import PipelineStage
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

# Call 1: free-form grounded research — no JSON constraint, search tool enabled.
_VERIFY_RESEARCH_PROMPT = """\
You are an adversarial fact-checker. Search for evidence that the following claim \
is FALSE, outdated, exaggerated, or missing critical context.

Search specifically for:
1. Studies or reports contradicting this number or mechanism
2. More recent data that supersedes this figure
3. Context making this statistic misleading
4. Primary sources (government, academic, regulatory) giving a different number

Claim: {claim}
Alleged source: {source_url}

Write a short factual summary of what you found: whether the claim holds up, what \
contradicts it (if anything), whether the source is primary or secondary, and what \
the correct number is if the claim's figure is wrong. Be concise — 3-5 sentences.\
"""

# Call 2: pure structuring — no tools, JSON output enforced.
_VERIFY_STRUCTURE_PROMPT = """\
Based on the following fact-check research, return a JSON object with exactly these fields:

Research findings:
{research}

JSON output (no markdown, no explanation):
{{
  "status": "CONFIRMED" | "PLAUSIBLE" | "CONTESTED" | "UNVERIFIED",
  "source_type": "primary" | "secondary",
  "contradiction_found": true | false,
  "contradiction_detail": "one sentence max, 200 chars max, or null",
  "verified_number": "corrected figure if different from claim, or null",
  "confidence_reason": "one sentence max, 150 chars max"
}}

CONFIRMED = found in 2+ independent sources, at least one primary (.gov/.edu/court filing)
PLAUSIBLE = secondary sources only, no contradiction found
CONTESTED = conflicting numbers or mechanisms across authoritative sources
UNVERIFIED = cannot confirm from any source found\
"""


def _source_at(sources: list, index: int, fallback: str = "") -> str:
    if index < len(sources) and sources[index]:
        return str(sources[index])
    return fallback


def _build_raw_claims(brief: dict) -> list[dict]:
    layer_sources = brief.get("three_layer_sources", [])
    support_sources = brief.get("supporting_fact_sources", [])
    fallback_source = brief.get("cold_fact_source", "")
    return [
        {
            "id": "cl_cold_fact",
            "claim": brief["cold_fact"],
            "source_url": brief.get("cold_fact_source", ""),
        },
        {
            "id": "cl_dinner_fact",
            "claim": brief["dinner_fact"],
            "source_url": brief.get("dinner_fact_source", ""),
        },
        *[
            {
                "id": f"cl_layer_{i}",
                "claim": layer,
                "source_url": _source_at(layer_sources, i, fallback_source),
            }
            for i, layer in enumerate(brief.get("three_layers", []))
        ],
        *[
            {
                "id": f"cl_support_{i}",
                "claim": fact,
                "source_url": _source_at(support_sources, i, fallback_source),
            }
            for i, fact in enumerate(brief.get("supporting_facts", []))
        ],
    ]


def _best_verified_anchor(
    results: list[dict],
    *,
    exclude_ids: set[str] | None = None,
    allow_plausible: bool = False,
) -> dict | None:
    exclude_ids = exclude_ids or set()
    allowed_statuses = {"CONFIRMED"} | ({"PLAUSIBLE"} if allow_plausible else set())
    candidates = [
        r for r in results
        if r.get("id") not in exclude_ids and r.get("status") in allowed_statuses
    ]
    if not candidates:
        return None

    def score(r: dict) -> tuple[int, int, int]:
        claim = r.get("claim", "")
        is_confirmed = 1 if r.get("status") == "CONFIRMED" else 0
        has_number = 1 if re.search(r"\d", claim) else 0
        short_enough = 1 if len(claim.split()) <= 28 else 0
        return (is_confirmed, has_number, short_enough)

    return max(candidates, key=score)


def _best_available_anchor(results: list[dict], *, exclude_ids: set[str] | None = None) -> dict | None:
    exclude_ids = exclude_ids or set()
    candidates = [
        r for r in results
        if r.get("id") not in exclude_ids and r.get("status") != "UNVERIFIED"
    ]
    if not candidates:
        return None
    status_rank = {"CONFIRMED": 3, "PLAUSIBLE": 2, "CONTESTED": 1}
    return max(
        candidates,
        key=lambda r: (
            status_rank.get(r.get("status"), 0),
            1 if re.search(r"\d", r.get("claim", "")) else 0,
            1 if len(r.get("claim", "").split()) <= 28 else 0,
        ),
    )


async def _verify_one(
    client: genai.Client,
    claim_id: str,
    claim_text: str,
    source_url: str,
) -> dict:
    # Call 1: grounded search — free-form text output, reasoning needed
    research_response = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_GROUNDING_MODEL,
        contents=_VERIFY_RESEARCH_PROMPT.format(
            claim=claim_text, source_url=source_url or "unknown"
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )
    research_text = research_response.text

    # Call 2: pure text-to-schema mapping — Gemini 2.5 Flash with thinking disabled
    structure_response = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_STRUCTURE_MODEL,
        contents=_VERIFY_STRUCTURE_PROMPT.format(research=research_text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    try:
        result = json.loads(structure_response.text)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning(
            "verify_json_parse_failed",
            claim_id=claim_id,
            error=str(exc),
            raw=structure_response.text[:200],
        )
        result = {
            "status": "UNVERIFIED",
            "source_type": "secondary",
            "contradiction_found": False,
            "contradiction_detail": f"parse error: {exc}",
            "verified_number": None,
            "confidence_reason": "Verify call returned unparseable response.",
        }
    result["id"] = claim_id
    result["claim"] = claim_text
    result["source_url"] = source_url
    return result


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> VerifiedFactSheet:
    paths = state_mgr.paths(slug)

    if not paths.auto_brief_json.exists():
        raise FileNotFoundError("auto_brief.json not found. Run RESEARCH (s01a) first.")

    brief = json.loads(paths.auto_brief_json.read_text())
    log.info("stage_verify_start", slug=slug)

    client = genai.Client(
        vertexai=True,
        **({} if not settings.gcp_project_id else {"project": settings.gcp_project_id}),
        location=GEMINI_GCP_LOCATION,
    )

    # Build the full list of claims to verify from the brief.
    # 2 anchors + 3 layers + 5 supporting = 10 raw claims.
    # After CONTESTED/UNVERIFIED are filtered, freeze gate of 8 passes comfortably.
    raw_claims = _build_raw_claims(brief)

    with state_mgr.run_stage(slug, PipelineStage.VERIFY) as paths:
        results = await asyncio.gather(*[
            _verify_one(client, c["id"], c["claim"], c["source_url"])
            for c in raw_claims
        ])

        # Separate by status
        confirmed = [r for r in results if r["status"] == "CONFIRMED"]
        plausible = [r for r in results if r["status"] == "PLAUSIBLE"]
        contested = [r for r in results if r["status"] in ("CONTESTED", "UNVERIFIED")]

        for r in contested:
            log.warning(
                "claim_contested_or_unverified",
                claim_id=r["id"],
                status=r["status"],
                detail=r.get("contradiction_detail"),
                slug=slug,
            )

        # Anchor policy:
        # - dinner_fact prefers CONFIRMED because it is the shareable claim.
        # - cold_fact should be CONFIRMED, but if the original is contested,
        #   swap in the strongest verified number-bearing claim instead of
        #   halting the whole narrative pipeline.
        cold_result = next((r for r in results if r["id"] == "cl_cold_fact"), None)
        dinner_result = next((r for r in results if r["id"] == "cl_dinner_fact"), None)

        if not cold_result or cold_result["status"] != "CONFIRMED":
            replacement = _best_verified_anchor(
                results,
                exclude_ids={"cl_dinner_fact"},
                allow_plausible=True,
            )
            if replacement is None:
                replacement = _best_available_anchor(results, exclude_ids={"cl_dinner_fact"})
            if replacement is None:
                replacement = cold_result
            brief["original_cold_fact"] = brief.get("cold_fact", "")
            brief["original_cold_fact_source"] = brief.get("cold_fact_source", "")
            if replacement:
                brief["cold_fact"] = replacement["claim"]
                brief["cold_fact_source"] = replacement["source_url"]
                brief["cold_fact_id"] = replacement["id"]
            else:
                brief["cold_fact_id"] = "cl_cold_fact"
            brief["cold_fact_replaced"] = True
            brief["cold_fact_replacement_reason"] = (
                f"Original cl_cold_fact was {cold_result['status'] if cold_result else 'missing'}; "
                f"using {replacement['id']} ({replacement['status']}) instead." if replacement
                else "no replacement available; keeping original brief anchor."
            )
            log.warning(
                "cold_fact_replaced",
                slug=slug,
                original_status=cold_result["status"] if cold_result else "missing",
                replacement_id=replacement["id"] if replacement else "cl_cold_fact",
                replacement_status=replacement["status"] if replacement else "missing",
            )
        else:
            brief["cold_fact_id"] = "cl_cold_fact"
            brief["cold_fact_replaced"] = False

        if not dinner_result or dinner_result["status"] != "CONFIRMED":
            replacement = _best_verified_anchor(
                results,
                exclude_ids={brief.get("cold_fact_id", ""), "cl_cold_fact"},
                allow_plausible=False,
            )
            if replacement is None:
                replacement = _best_available_anchor(
                    results,
                    exclude_ids={brief.get("cold_fact_id", ""), "cl_cold_fact"},
                )
            if replacement is None:
                replacement = dinner_result
            brief["original_dinner_fact"] = brief.get("dinner_fact", "")
            brief["original_dinner_fact_source"] = brief.get("dinner_fact_source", "")
            if replacement:
                brief["dinner_fact"] = replacement["claim"]
                brief["dinner_fact_source"] = replacement["source_url"]
            brief["dinner_fact_replaced"] = True
            dinner_fact_id = replacement["id"] if replacement else "cl_dinner_fact"
            log.warning(
                "dinner_fact_replaced",
                slug=slug,
                original_status=dinner_result["status"] if dinner_result else "missing",
                replacement_id=dinner_fact_id,
                replacement_status=replacement["status"] if replacement else "missing",
            )
        else:
            brief["dinner_fact_replaced"] = False
            dinner_fact_id = "cl_dinner_fact"

        # Build claim list. Keep CONTESTED claims in the sheet as explicitly
        # labeled low-trust material so downstream prompts can avoid presenting
        # them as settled facts, while the run keeps moving.
        verified_claims = [
            VerifiedClaim(
                id=r["id"],
                claim=r["claim"],
                status=r["status"],
                source_url=r["source_url"],
                source_type=r.get("source_type", "secondary"),
                number_present=bool(re.search(r"\d", r["claim"])),
                contradiction_checked=True,
            )
            for r in results
            if r["status"] in ("CONFIRMED", "PLAUSIBLE", "CONTESTED")
        ]
        if not verified_claims and results:
            fallback = next((r for r in results if r["id"] == "cl_cold_fact"), results[0])
            verified_claims.append(
                VerifiedClaim(
                    id=fallback["id"],
                    claim=fallback["claim"],
                    status="CONTESTED",
                    source_url=fallback["source_url"],
                    source_type=fallback.get("source_type", "secondary"),
                    number_present=bool(re.search(r"\d", fallback["claim"])),
                    contradiction_checked=True,
                )
            )
            log.warning(
                "verify_all_claims_unverified_added_contested_fallback",
                slug=slug,
                claim_id=fallback["id"],
            )

        sheet = VerifiedFactSheet(
            topic=state_mgr.load(slug).topic,
            created_at=datetime.utcnow(),
            claims=verified_claims,
            dinner_fact_id=dinner_fact_id,
            brief=brief,  # full brief carried forward to Architect
        )
        paths.verified_fact_sheet.write_text(sheet.model_dump_json(indent=2))

        log.info(
            "stage_verify_done",
            slug=slug,
            confirmed=len(confirmed),
            plausible=len(plausible),
            contested=len(contested),
            total_verified=len(verified_claims),
        )

    return sheet
