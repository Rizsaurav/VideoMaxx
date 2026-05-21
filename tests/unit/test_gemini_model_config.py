from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vidmaxx.config.constants import (
    GEMINI_GCP_LOCATION,
    GEMINI_GROUNDING_MODEL,
    GEMINI_STRUCTURE_MODEL,
)


def _grounding_response(text: str = "finding"):
    chunk = SimpleNamespace(web=SimpleNamespace(uri="https://example.gov/report"))
    metadata = SimpleNamespace(grounding_chunks=[chunk])
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(grounding_metadata=metadata)])


def _brief_response():
    return SimpleNamespace(text=json.dumps({
        "cold_fact": "Example cold fact with 42 percent.",
        "cold_fact_source": "https://example.gov/report",
        "false_belief": "Most people believe the visible explanation.",
        "three_layers": ["Layer 1", "Layer 2", "Layer 3"],
        "three_layer_sources": [
            "https://example.gov/report",
            "https://example.gov/report",
            "https://example.gov/report",
        ],
        "supporting_facts": ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"],
        "supporting_fact_sources": [
            "https://example.gov/report",
            "https://example.gov/report",
            "https://example.gov/report",
            "https://example.gov/report",
            "https://example.gov/report",
        ],
        "twist": "The deeper implication is structural.",
        "dinner_fact": "Example dinner fact with 99 percent.",
        "dinner_fact_source": "https://example.gov/report",
    }))


def _verify_structure_response():
    return SimpleNamespace(text=json.dumps({
        "status": "CONFIRMED",
        "source_type": "primary",
        "contradiction_found": False,
        "contradiction_detail": None,
        "verified_number": None,
        "confidence_reason": "Confirmed from primary sources.",
    }))


def _assert_grounding_call(call: dict) -> None:
    assert call["model"] == GEMINI_GROUNDING_MODEL
    cfg = call["config"]
    assert cfg.response_mime_type is None
    assert cfg.thinking_config is None
    assert cfg.tools, "grounding call must include google_search tool"


def _assert_structure_call(call: dict) -> None:
    assert call["model"] == GEMINI_STRUCTURE_MODEL
    cfg = call["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.thinking_config.thinking_budget == 0
    assert cfg.tools is None
    assert cfg.max_output_tokens == 8192


@pytest.mark.asyncio
async def test_s01b_verify_uses_grounding_then_json_structure_models():
    from vidmaxx.pipeline.stages.s01b_verify import _verify_one

    calls: list[dict] = []

    async def fake_to_thread(_fn, *args, **kwargs):
        calls.append(kwargs)
        return _grounding_response("Research says the claim is confirmed.") if len(calls) == 1 else _verify_structure_response()

    with patch("asyncio.to_thread", new=AsyncMock(side_effect=fake_to_thread)):
        result = await _verify_one(
            client=MagicMock(),
            claim_id="cl_cold_fact",
            claim_text="Example claim with 42 percent.",
            source_url="https://example.gov/report",
        )

    assert result["status"] == "CONFIRMED"
    assert len(calls) == 2
    _assert_grounding_call(calls[0])
    _assert_structure_call(calls[1])


class _FakeStateManager:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._paths = SimpleNamespace(
            audio_dir=root / "audio",
            assets_dir=root / "assets",
            out_dir=root / "out",
            brief_json=root / "brief.json",
            auto_brief_json=root / "auto_brief.json",
            discovery_json=root / "discovery.json",
        )

    def load(self, _slug: str):
        return SimpleNamespace(topic="test topic")

    @contextmanager
    def run_stage(self, _slug: str, _stage):
        yield self._paths


@pytest.mark.asyncio
async def test_s01a_research_uses_grounding_then_json_structure_models(tmp_path):
    from vidmaxx.pipeline.stages import s01a_research

    calls: list[dict] = []

    async def fake_to_thread(_fn, *args, **kwargs):
        calls.append(kwargs)
        findings = "\n".join(
            f"[https://example.gov/report] | primary | Example finding {i} with {i} percent."
            for i in range(10)
        )
        return _grounding_response(findings) if len(calls) == 1 else _brief_response()

    fake_client = MagicMock()
    with (
        patch("vidmaxx.pipeline.stages.s01a_research.genai.Client", return_value=fake_client) as client_ctor,
        patch("asyncio.to_thread", new=AsyncMock(side_effect=fake_to_thread)),
    ):
        brief = await s01a_research.run(
            slug="slug",
            state_mgr=_FakeStateManager(tmp_path),
            cache=MagicMock(),
            settings=SimpleNamespace(gcp_project_id=""),
        )

    client_ctor.assert_called_once_with(vertexai=True, location=GEMINI_GCP_LOCATION)
    assert brief["cold_fact_source_grounding_verified"] is True
    assert brief["dinner_fact_source_grounding_verified"] is True
    assert len(calls) == 2
    _assert_grounding_call(calls[0])
    _assert_structure_call(calls[1])
