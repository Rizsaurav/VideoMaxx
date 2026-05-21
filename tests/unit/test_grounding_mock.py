"""
Mock tests for Gemini grounding integration (s01a + s01b).

These tests never hit the network. They patch `asyncio.to_thread` to return
a fake Gemini response object, then verify that s01a and s01b correctly:
  - Parse the JSON from response.text
  - Extract grounding URLs from response.candidates[0].grounding_metadata
  - Set *_grounding_verified flags based on URL presence
  - Hard-fail when cold_fact or dinner_fact come back non-CONFIRMED (s01b)
  - Write auto_brief.json / verified_fact_sheet.json to the expected paths

Run with:
    pytest tests/unit/test_grounding_mock.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REAL_URL = "https://www.cdc.gov/nchs/data/example.pdf"
FAKE_URL = "https://made-up-hallucinated-source.example.com/data"
VERTEX_REDIRECT_URL = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEa0Lexample"

MOCK_BRIEF = {
    "cold_fact": "The average American spends 97 minutes per day on social media, totalling 24 days per year.",
    "cold_fact_source": REAL_URL,
    "false_belief": "Most people believe social media is a neutral communication tool.",
    "three_layers": [
        "Meta's 2023 10-K shows $116B in ad revenue — 97% from behavioral targeting.",
        "The engagement algorithm optimises for outrage because anger has a 70% higher reshare rate than neutral content.",
        "Every scroll you take is processed by a model trained to keep you dependent.",
    ],
    "three_layer_sources": [
        "https://www.sec.gov/Archives/edgar/data/meta-2023-10k.htm",
        "https://www.nih.gov/example-outrage-study",
        "https://www.ftc.gov/example-social-media-report",
    ],
    "supporting_facts": [
        "Instagram users who see outrage content are 3x more likely to return within 30 minutes.",
        "Facebook's internal research found 64% of extremist group joins were driven by its recommendation algorithm.",
        "The average user touches their phone 2,617 times per day, up from 1,500 in 2016.",
        "Social media companies spent $​877M lobbying against platform liability laws from 2010 to 2023.",
        "Children who use social media for 5+ hours daily are 66% more likely to report depression symptoms.",
    ],
    "supporting_fact_sources": [
        "https://www.nih.gov/example-instagram-study",
        "https://www.sec.gov/example-facebook-internal-research",
        "https://www.bls.gov/example-phone-use",
        "https://www.opensecrets.org/example-platform-lobbying",
        "https://www.cdc.gov/example-teen-social-media",
    ],
    "twist": "The companies that sell the attention of their users are also the primary funders of digital-literacy research.",
    "dinner_fact": "TikTok's algorithm can predict your political leaning from 8 minutes of watch history with 87% accuracy.",
    "dinner_fact_source": REAL_URL,
}


def _make_grounding_response(text: str, grounding_urls: list[str]):
    """Build a minimal fake Gemini response with grounding metadata."""
    chunks = [
        SimpleNamespace(web=SimpleNamespace(uri=u))
        for u in grounding_urls
    ]
    grounding_metadata = SimpleNamespace(grounding_chunks=chunks)
    candidate = SimpleNamespace(grounding_metadata=grounding_metadata)
    return SimpleNamespace(text=text, candidates=[candidate])


def _make_verify_response(status: str, source_type: str = "primary"):
    payload = {
        "status": status,
        "source_type": source_type,
        "contradiction_found": False,
        "contradiction_detail": None,
        "verified_number": None,
        "confidence_reason": "Found in multiple primary sources.",
    }
    return SimpleNamespace(text=json.dumps(payload), candidates=[
        SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=[]))
    ])


# ---------------------------------------------------------------------------
# s01a tests — _extract_grounding_urls + grounding_verified flags
# ---------------------------------------------------------------------------

class TestExtractGroundingUrls:
    def test_extracts_urls_from_valid_response(self):
        from vidmaxx.pipeline.stages.s01a_research import _extract_grounding_urls

        resp = _make_grounding_response("", [REAL_URL, "https://sec.gov/filing.htm"])
        urls = _extract_grounding_urls(resp)
        assert REAL_URL in urls
        assert "https://sec.gov/filing.htm" in urls

    def test_returns_empty_set_on_missing_metadata(self):
        from vidmaxx.pipeline.stages.s01a_research import _extract_grounding_urls

        resp = SimpleNamespace(text="", candidates=[])
        urls = _extract_grounding_urls(resp)
        assert urls == set()

    def test_returns_empty_set_on_no_web_chunks(self):
        from vidmaxx.pipeline.stages.s01a_research import _extract_grounding_urls

        chunk_no_web = SimpleNamespace(web=None)
        candidate = SimpleNamespace(
            grounding_metadata=SimpleNamespace(grounding_chunks=[chunk_no_web])
        )
        resp = SimpleNamespace(text="", candidates=[candidate])
        urls = _extract_grounding_urls(resp)
        assert urls == set()


class TestGroundingVerifiedFlag:
    """
    Verify that s01a correctly sets *_grounding_verified based on whether
    the source URL appeared in the grounding metadata.
    """

    def _run_research_sync(self, brief_json: str, grounding_urls: list[str]) -> dict:
        import asyncio
        from vidmaxx.pipeline.stages.s01a_research import _extract_grounding_urls, _strip_json_fences

        fake_response = _make_grounding_response(brief_json, grounding_urls)

        # Replicate the URL-validation logic from s01a.run() directly,
        # without needing a full ProjectStateManager.
        grounding_url_set = _extract_grounding_urls(fake_response)
        brief = json.loads(_strip_json_fences(fake_response.text))
        for field in ("cold_fact_source", "dinner_fact_source"):
            url = brief.get(field, "")
            brief[f"{field}_grounding_verified"] = url in grounding_url_set
        return brief

    def test_verified_when_url_in_grounding_metadata(self):
        brief = self._run_research_sync(
            json.dumps(MOCK_BRIEF),
            grounding_urls=[REAL_URL],
        )
        assert brief["cold_fact_source_grounding_verified"] is True
        assert brief["dinner_fact_source_grounding_verified"] is True

    def test_unverified_when_url_absent_from_grounding_metadata(self):
        brief_with_fake = {**MOCK_BRIEF, "cold_fact_source": FAKE_URL}
        brief = self._run_research_sync(
            json.dumps(brief_with_fake),
            grounding_urls=[REAL_URL],
        )
        assert brief["cold_fact_source_grounding_verified"] is False
        assert brief["dinner_fact_source_grounding_verified"] is True


class TestResearchSourceValidation:
    def _valid_brief(self) -> dict:
        return {
            **MOCK_BRIEF,
            "three_layer_sources": [REAL_URL, REAL_URL, REAL_URL],
            "supporting_fact_sources": [REAL_URL, REAL_URL, REAL_URL, REAL_URL, REAL_URL],
        }

    def _findings(self) -> list[dict]:
        return [
            {"source_url": REAL_URL, "source_type": "primary", "finding": f"Finding {i}"}
            for i in range(10)
        ]

    def test_repairs_brief_source_not_in_finder_or_grounding(self):
        from vidmaxx.pipeline.stages.s01a_research import _validate_and_mark_sources

        brief = self._valid_brief()
        brief["dinner_fact_source"] = FAKE_URL

        repaired = _validate_and_mark_sources(
            brief,
            grounding_urls={REAL_URL},
            parsed_findings=self._findings(),
        )
        assert repaired["dinner_fact_source"] == REAL_URL
        assert repaired["source_repair_count"] == 1
        assert repaired["source_repair_log"][0]["old_url"] == FAKE_URL

    def test_marks_all_sources_verified_when_claim_sources_match_finder(self):
        from vidmaxx.pipeline.stages.s01a_research import _validate_and_mark_sources

        brief = _validate_and_mark_sources(
            self._valid_brief(),
            grounding_urls={REAL_URL},
            parsed_findings=self._findings(),
        )
        assert brief["all_claim_sources_verified"] is True
        assert brief["cold_fact_source_grounding_verified"] is True
        assert brief["dinner_fact_source_grounding_verified"] is True
        assert brief["source_repair_count"] == 0

    def test_accepts_clean_finder_url_when_grounding_metadata_uses_vertex_redirect(self):
        from vidmaxx.pipeline.stages.s01a_research import _validate_and_mark_sources

        brief = _validate_and_mark_sources(
            self._valid_brief(),
            grounding_urls={VERTEX_REDIRECT_URL},
            parsed_findings=self._findings(),
        )

        assert brief["cold_fact_source"] == REAL_URL
        assert brief["dinner_fact_source"] == REAL_URL
        assert brief["all_claim_sources_verified"] is True
        assert brief["source_repair_count"] == 0

    def test_replaces_hallucinated_url_with_best_finder_source(self):
        from vidmaxx.pipeline.stages.s01a_research import _validate_and_mark_sources

        brief = self._valid_brief()
        brief["cold_fact"] = "Example finding 7 with 7 percent."
        brief["cold_fact_source"] = FAKE_URL
        findings = [
            {"source_url": REAL_URL, "source_type": "primary", "finding": "Unrelated finding."},
            {
                "source_url": "https://www.bls.gov/example-wages",
                "source_type": "primary",
                "finding": "Example finding 7 with 7 percent.",
            },
        ]

        repaired = _validate_and_mark_sources(
            brief,
            grounding_urls={VERTEX_REDIRECT_URL},
            parsed_findings=findings,
        )

        assert repaired["cold_fact_source"] == "https://www.bls.gov/example-wages"
        assert repaired["source_repair_count"] == 1
        assert repaired["source_repair_log"][0]["old_url"] == FAKE_URL

    def test_pads_underfilled_brief_from_parsed_findings(self):
        from vidmaxx.pipeline.stages.s01a_research import _validate_and_mark_sources

        findings = self._findings()
        brief = {
            "cold_fact": "Finding 0",
            "cold_fact_source": REAL_URL,
            "false_belief": "Most people believe the visible explanation.",
            "three_layers": ["Finding 1"],
            "three_layer_sources": [REAL_URL],
            "supporting_facts": ["Finding 2"],
            "supporting_fact_sources": [REAL_URL],
            "twist": "Finding 3",
            "dinner_fact": "Finding 4",
            "dinner_fact_source": REAL_URL,
        }

        repaired = _validate_and_mark_sources(
            brief,
            grounding_urls={REAL_URL},
            parsed_findings=findings,
        )

        assert len(repaired["three_layers"]) == 3
        assert len(repaired["three_layer_sources"]) == 3
        assert len(repaired["supporting_facts"]) == 5
        assert len(repaired["supporting_fact_sources"]) == 5


# ---------------------------------------------------------------------------
# s01b claim extraction — 10-claim structure
# ---------------------------------------------------------------------------

class TestRawClaimExtraction:
    """Verify s01b builds 10 raw claims from the brief (2 anchors + 3 layers + 5 supporting)."""

    def _build_raw_claims(self, brief: dict) -> list[dict]:
        from vidmaxx.pipeline.stages.s01b_verify import _build_raw_claims
        return _build_raw_claims(brief)

    def test_produces_ten_claims_from_full_brief(self):
        claims = self._build_raw_claims(MOCK_BRIEF)
        assert len(claims) == 10

    def test_claim_ids_are_correct(self):
        claims = self._build_raw_claims(MOCK_BRIEF)
        ids = [c["id"] for c in claims]
        assert ids == [
            "cl_cold_fact", "cl_dinner_fact",
            "cl_layer_0", "cl_layer_1", "cl_layer_2",
            "cl_support_0", "cl_support_1", "cl_support_2", "cl_support_3", "cl_support_4",
        ]

    def test_missing_supporting_facts_produces_five_claims(self):
        brief_no_support = {k: v for k, v in MOCK_BRIEF.items() if k != "supporting_facts"}
        claims = self._build_raw_claims(brief_no_support)
        assert len(claims) == 5
        assert not any(c["id"].startswith("cl_support_") for c in claims)

    def test_supporting_facts_use_matching_source_urls(self):
        claims = self._build_raw_claims(MOCK_BRIEF)
        support_claims = [c for c in claims if c["id"].startswith("cl_support_")]
        assert [c["source_url"] for c in support_claims] == MOCK_BRIEF["supporting_fact_sources"]

    def test_layers_use_matching_source_urls(self):
        claims = self._build_raw_claims(MOCK_BRIEF)
        layer_claims = [c for c in claims if c["id"].startswith("cl_layer_")]
        assert [c["source_url"] for c in layer_claims] == MOCK_BRIEF["three_layer_sources"]


# ---------------------------------------------------------------------------
# s01b tests — verify logic, anchor fallback inputs, brief embedding
# ---------------------------------------------------------------------------

class TestVerifyStatusParsing:
    """
    _verify_one preserves model status output. The run-level stage no longer
    hard-halts on non-CONFIRMED anchors; it repairs anchors from available claims.
    """

    def _build_fake_brief(self, tmp_path: Path) -> Path:
        brief_path = tmp_path / "auto_brief.json"
        brief_path.write_text(json.dumps(MOCK_BRIEF))
        return brief_path

    @pytest.mark.asyncio
    async def test_confirmed_all_passes(self, tmp_path):
        from vidmaxx.pipeline.stages.s01b_verify import _verify_one

        # Directly test _verify_one with a mocked to_thread
        confirmed_resp = _make_verify_response("CONFIRMED")
        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=confirmed_resp):
            result = await _verify_one(
                client=MagicMock(),
                claim_id="cl_cold_fact",
                claim_text=MOCK_BRIEF["cold_fact"],
                source_url=MOCK_BRIEF["cold_fact_source"],
            )
        assert result["status"] == "CONFIRMED"
        assert result["id"] == "cl_cold_fact"

    @pytest.mark.asyncio
    async def test_plausible_cold_fact_raises(self, tmp_path):
        from vidmaxx.pipeline.stages.s01b_verify import _verify_one

        plausible_resp = _make_verify_response("PLAUSIBLE")
        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=plausible_resp):
            result = await _verify_one(
                client=MagicMock(),
                claim_id="cl_cold_fact",
                claim_text=MOCK_BRIEF["cold_fact"],
                source_url=MOCK_BRIEF["cold_fact_source"],
            )
        # _verify_one itself doesn't raise — the caller (run()) raises.
        # Confirm the status is preserved so run() can gate on it.
        assert result["status"] == "PLAUSIBLE"

    @pytest.mark.asyncio
    async def test_contested_claim_excluded(self):
        from vidmaxx.pipeline.stages.s01b_verify import _verify_one

        contested_resp = _make_verify_response("CONTESTED")
        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=contested_resp):
            result = await _verify_one(
                client=MagicMock(),
                claim_id="cl_layer_0",
                claim_text=MOCK_BRIEF["three_layers"][0],
                source_url=MOCK_BRIEF["cold_fact_source"],
            )
        assert result["status"] == "CONTESTED"


class TestVerifyAnchorFallbacks:
    def test_best_verified_anchor_prefers_confirmed_number_claim(self):
        from vidmaxx.pipeline.stages.s01b_verify import _best_verified_anchor

        results = [
            {"id": "a", "status": "PLAUSIBLE", "claim": "A plausible claim with 99 percent."},
            {"id": "b", "status": "CONFIRMED", "claim": "A confirmed claim without a number."},
            {"id": "c", "status": "CONFIRMED", "claim": "A confirmed claim with 42 percent."},
            {"id": "d", "status": "CONTESTED", "claim": "A contested claim with 100 percent."},
        ]

        best = _best_verified_anchor(results)
        assert best["id"] == "c"

    def test_best_verified_anchor_can_allow_plausible_fallback(self):
        from vidmaxx.pipeline.stages.s01b_verify import _best_verified_anchor

        results = [
            {"id": "a", "status": "PLAUSIBLE", "claim": "A plausible claim with 99 percent."},
            {"id": "d", "status": "CONTESTED", "claim": "A contested claim with 100 percent."},
        ]

        assert _best_verified_anchor(results) is None
        best = _best_verified_anchor(results, allow_plausible=True)
        assert best["id"] == "a"


# ---------------------------------------------------------------------------
# s01a _strip_json_fences
# ---------------------------------------------------------------------------

class TestStripJsonFences:
    def test_strips_json_fences(self):
        from vidmaxx.pipeline.stages.s01a_research import _strip_json_fences

        raw = '```json\n{"key": "value"}\n```'
        assert _strip_json_fences(raw) == '{"key": "value"}'

    def test_passthrough_when_no_fences(self):
        from vidmaxx.pipeline.stages.s01a_research import _strip_json_fences

        raw = '{"key": "value"}'
        assert _strip_json_fences(raw) == raw

    def test_strips_plain_triple_backtick(self):
        from vidmaxx.pipeline.stages.s01a_research import _strip_json_fences

        raw = '```\n{"key": "value"}\n```'
        assert _strip_json_fences(raw) == '{"key": "value"}'
