from __future__ import annotations

import json
import os

import pytest
from google import genai
from google.genai import types

from vidmaxx.config.constants import (
    GEMINI_GCP_LOCATION,
    GEMINI_GROUNDING_MODEL,
    GEMINI_STRUCTURE_MODEL,
)


pytestmark = pytest.mark.skipif(
    os.getenv("VIDMAXX_LIVE_VERTEX") != "1",
    reason="Set VIDMAXX_LIVE_VERTEX=1 to make real Vertex AI model calls.",
)


def _client() -> genai.Client:
    project = os.getenv("GCP_PROJECT_ID", "")
    return genai.Client(
        vertexai=True,
        **({} if not project else {"project": project}),
        location=os.getenv("GCP_LOCATION", GEMINI_GCP_LOCATION),
    )


def test_structure_model_live_json_mode_loads_and_returns_json():
    client = _client()
    response = client.models.generate_content(
        model=GEMINI_STRUCTURE_MODEL,
        contents='Return exactly this JSON object: {"ok": true, "kind": "structure"}',
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
            max_output_tokens=512,
        ),
    )

    data = json.loads(response.text)
    assert data["ok"] is True
    assert data["kind"] == "structure"


def test_grounding_model_live_search_tool_loads():
    client = _client()
    response = client.models.generate_content(
        model=GEMINI_GROUNDING_MODEL,
        contents=(
            "Use Google Search grounding. In one short sentence, say what the "
            "official current Vertex AI Gemini Flash model family is called."
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            max_output_tokens=2048,
        ),
    )

    assert response.text
    assert response.candidates[0].grounding_metadata is not None
    assert response.candidates[0].grounding_metadata.web_search_queries
