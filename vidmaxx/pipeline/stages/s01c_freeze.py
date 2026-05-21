"""
Stage 1c — Freeze (hash + minimum claims gate)

Reads:   paths.verified_fact_sheet  (frozen=False, written by s01b)
Writes:  paths.verified_fact_sheet  (updated: file_hash set, frozen=True)
         state_mgr.factsheet_hashes table  (SHA-256 recorded)
Advances: FREEZE → ARCHITECT

Guards (non-blocking):
  - Minimum 8 claims preferred. If fewer, mark risk in fact_sheet.brief and continue.
  - At least 1 CONFIRMED claim preferred. If none, mark risk and continue.
  - dinner_fact_id should point to the strongest available claim. If missing or contested,
    repair/select a fallback and continue.
  - At least 1 primary source (warning only — YouTube topics don't always have filings)

SHA-256 is computed over canonical JSON (frozen=False, file_hash="") so the hash
reflects content only, not the frozen flag itself.
"""

from __future__ import annotations

import hashlib

import structlog

from vidmaxx.config.settings import Settings
from vidmaxx.models.factsheet import VerifiedFactSheet
from vidmaxx.models.project import PipelineStage
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

_MIN_CLAIMS = 8


def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> VerifiedFactSheet:
    paths = state_mgr.paths(slug)
    fact_sheet = VerifiedFactSheet.model_validate_json(paths.verified_fact_sheet.read_text())
    log.info("stage_freeze_start", slug=slug, claims=len(fact_sheet.claims))

    guardrail_warnings: list[str] = []

    # Gate 1: preferred minimum claim count
    if len(fact_sheet.claims) < _MIN_CLAIMS:
        guardrail_warnings.append(
            f"only {len(fact_sheet.claims)} claims; preferred minimum is {_MIN_CLAIMS}"
        )
        log.warning("freeze_low_claim_count", slug=slug, claims=len(fact_sheet.claims), preferred=_MIN_CLAIMS)

    # Gate 2: at least one CONFIRMED claim
    confirmed = [c for c in fact_sheet.claims if c.status == "CONFIRMED"]
    if not confirmed:
        guardrail_warnings.append("0 CONFIRMED claims; continuing with lower-trust factsheet")
        log.warning("freeze_no_confirmed_claims", slug=slug, claims=len(fact_sheet.claims))

    # Gate 3: dinner_fact_id should point to the strongest available claim.
    dinner_fact_id = fact_sheet.dinner_fact_id
    if fact_sheet.dinner_fact_id:
        dinner_claim = next(
            (c for c in fact_sheet.claims if c.id == fact_sheet.dinner_fact_id), None
        )
        if dinner_claim is None:
            guardrail_warnings.append(
                f"dinner_fact_id {fact_sheet.dinner_fact_id!r} not found; selecting fallback"
            )
            dinner_fact_id = ""
        elif dinner_claim.status != "CONFIRMED":
            guardrail_warnings.append(
                f"dinner_fact {fact_sheet.dinner_fact_id!r} is {dinner_claim.status}; continuing with risk flag"
            )
            log.warning(
                "freeze_dinner_fact_not_confirmed",
                slug=slug,
                dinner_fact_id=fact_sheet.dinner_fact_id,
                status=dinner_claim.status,
            )
    else:
        guardrail_warnings.append("dinner_fact_id empty; selecting fallback")

    if not dinner_fact_id and fact_sheet.claims:
        preferred = (
            next((c for c in fact_sheet.claims if c.status == "CONFIRMED" and c.number_present), None)
            or next((c for c in fact_sheet.claims if c.status == "CONFIRMED"), None)
            or next((c for c in fact_sheet.claims if c.number_present), None)
            or fact_sheet.claims[0]
        )
        dinner_fact_id = preferred.id
        log.warning("freeze_dinner_fact_fallback_selected", slug=slug, dinner_fact_id=dinner_fact_id)

    # Warning (not a hard stop): primary source coverage
    primary = [c for c in fact_sheet.claims if c.source_type == "primary"]
    if not primary:
        log.warning("freeze_no_primary_sources", slug=slug, claims=len(fact_sheet.claims))

    brief = dict(fact_sheet.brief or {})
    if guardrail_warnings:
        brief["freeze_guardrail_warnings"] = guardrail_warnings
        brief["freeze_risk_level"] = "elevated"
    hash_input = fact_sheet.model_copy(
        update={"dinner_fact_id": dinner_fact_id, "brief": brief}
    )

    with state_mgr.run_stage(slug, PipelineStage.FREEZE) as paths:
        # Compute hash over content-only JSON (stable: frozen=False, file_hash="")
        content_json = hash_input.model_copy(
            update={"frozen": False, "file_hash": ""}
        ).model_dump_json(indent=2)
        file_hash = hashlib.sha256(content_json.encode()).hexdigest()

        state_mgr.set_factsheet_hash(slug, file_hash)

        frozen = hash_input.model_copy(
            update={
                "file_hash": file_hash,
                "frozen": True,
            }
        )
        paths.verified_fact_sheet.write_text(frozen.model_dump_json(indent=2))

        log.info(
            "stage_freeze_done",
            slug=slug,
            hash=file_hash[:12],
            claims=len(frozen.claims),
            confirmed=len(confirmed),
            primary=len(primary),
            dinner_fact_id=frozen.dinner_fact_id,
        )

    return frozen
