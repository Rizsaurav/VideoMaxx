"""
Stage 1a — Research (Gemini 2.5 Pro + Google Grounding)

Reads:   Project.topic
         paths.brief_json  (optional — if present, skips Gemini call entirely)
Writes:  paths.auto_brief_json  (auto_brief.json)
Advances: RESEARCH → VERIFY

Gemini 2.5 Pro with google_search grounding fetches real web pages and returns
a structured brief with source URLs. The grounding metadata contains the actual
URLs Gemini retrieved — source URLs in the JSON that don't appear in grounding
metadata are flagged as unverified (not hard-failed here; s01b verifies them).

If brief.json exists in the project dir (user-supplied via `vidmaxx new -b`),
it is copied to auto_brief.json and the Gemini call is skipped. This lets users
pre-research topics and bypass the automated research pass.

Output schema (auto_brief.json):
{
  "cold_fact": "single most alarming verifiable stat. specific number. max 2 sentences.",
  "cold_fact_source": "exact URL retrieved",
  "cold_fact_source_grounding_verified": true/false,
  "false_belief": "what most people currently believe — genuinely held, not obvious",
  "three_layers": [
    "surface system with specific number and named entity",
    "hidden mechanism — causal chain",
    "personal consequence — what this does to the viewer specifically"
  ],
  "twist": "implication one level deeper than obvious conclusion",
  "dinner_fact": "one sentence. specific number. different from cold_fact.",
  "dinner_fact_source": "exact URL retrieved",
  "dinner_fact_source_grounding_verified": true/false
}
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urlparse

import structlog
from google import genai
from google.genai import types

from vidmaxx.config.constants import (
    GEMINI_GCP_LOCATION,
    GEMINI_GROUNDING_MODEL,
    GEMINI_STRUCTURE_MODEL,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

# Pass 1 — Finder: fetch raw data, return URL + finding pairs. No narrative, no brief.
# Grounding is enabled so the model must actually fetch pages. Output is free-form text.
_FINDER_PROMPT = """\
You are a research librarian. Your ONLY job is to find primary source data on the \
following topic. Do not write a brief or narrative. Do not infer or guess.

Topic: {topic}

Search the web and return a list of what you actually found. For each finding:
- Copy the exact statistic or fact verbatim from the source
- Include the exact URL you retrieved it from
- Note whether it is a primary source (.gov, .edu, court filing, SEC filing, \
central bank report, peer-reviewed study) or secondary

Format as plain text, one finding per line:
[URL] | [primary/secondary] | [verbatim stat or finding from that page]

Find at least 10 distinct findings from at least 5 different sources. \
Stop when you have enough to cover: a shocking opening stat, the mechanism \
behind the topic, personal impact on the viewer, a counterintuitive twist, \
and a shareable dinner-party fact.\
"""

# Pass 2 — Architect: write the brief using ONLY what the Finder returned.
# No tools — the model cannot go back to the web. It is constrained to the provided data.
_BRIEF_FROM_FINDINGS_PROMPT = """\
You are writing a research brief for a YouTube video. You MUST use only the \
findings listed below. Do not add any fact, statistic, or URL that is not \
in this list. If you cannot support a claim with a provided finding, omit it.

Findings from primary research:
{findings}

Using ONLY the above findings, return a JSON object:

{{
  "cold_fact": "most alarming stat verbatim from findings. specific number. 2 sentences max.",
  "cold_fact_source": "exact URL from the findings list above",
  "false_belief": "what most people currently believe about this topic. \
genuinely held, not obviously wrong.",
  "three_layers": [
    "surface system: visible institution or policy — use a specific number and named entity from findings",
    "hidden mechanism: causal chain — use a specific mechanism from findings",
    "personal consequence: what this does to the viewer — use a specific impact from findings"
  ],
  "three_layer_sources": [
    "exact URL from findings for three_layers[0]",
    "exact URL from findings for three_layers[1]",
    "exact URL from findings for three_layers[2]"
  ],
  "supporting_facts": [
    "verbatim or near-verbatim stat #1 from findings",
    "verbatim or near-verbatim stat #2 from findings",
    "verbatim or near-verbatim stat #3 from findings",
    "verbatim or near-verbatim stat #4 from findings",
    "verbatim or near-verbatim stat #5 from findings"
  ],
  "supporting_fact_sources": [
    "exact URL from findings for supporting_facts[0]",
    "exact URL from findings for supporting_facts[1]",
    "exact URL from findings for supporting_facts[2]",
    "exact URL from findings for supporting_facts[3]",
    "exact URL from findings for supporting_facts[4]"
  ],
  "twist": "implication one level deeper than the obvious conclusion — from findings only.",
  "dinner_fact": "one sentence. specific number. different from cold_fact. from findings only.",
  "dinner_fact_source": "exact URL from the findings list above"
}}

Rules:
- Every number must appear verbatim in the findings list above
- Every *_source value must be a URL that appears in the findings list
- Each three_layers item must use its matching three_layer_sources URL
- Each supporting_facts item must use its matching supporting_fact_sources URL
- No hedging language
- No facts from outside the provided findings\
"""

_FINDING_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"
    r"(?:\[(?P<bracket_url>https?://[^\]\s]+)\]|(?P<plain_url>https?://\S+))"
    r"\s*\|\s*(?P<source_type>primary|secondary)\s*\|\s*(?P<finding>.+?)\s*$",
    re.IGNORECASE,
)


def _extract_grounding_urls(response) -> set[str]:
    try:
        chunks = (
            response.candidates[0]
            .grounding_metadata
            .grounding_chunks
        )
        return {
            chunk.web.uri
            for chunk in chunks
            if chunk.web and chunk.web.uri
        }
    except (AttributeError, IndexError, TypeError):
        return set()


def _url_key(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/')}"


def _url_host(url: str) -> str:
    return urlparse(url.strip()).netloc.lower().removeprefix("www.")


def _is_vertex_grounding_redirect(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    return (
        host == "vertexaisearch.cloud.google.com"
        and "/grounding-api-redirect/" in parsed.path
    )


def _grounding_has_redirects(grounding_urls: set[str]) -> bool:
    return any(_is_vertex_grounding_redirect(url) for url in grounding_urls)


def _parse_findings(text: str) -> list[dict]:
    findings: list[dict] = []
    for line in text.splitlines():
        match = _FINDING_RE.match(line)
        if not match:
            continue
        url = (match.group("bracket_url") or match.group("plain_url")).rstrip(".,)")
        findings.append({
            "source_url": url,
            "source_type": match.group("source_type").lower(),
            "finding": match.group("finding").strip(),
        })
    return findings


def _source_is_verified(url: str, grounding_urls: set[str], finding_urls: set[str]) -> bool:
    if not url:
        return False
    url_key = _url_key(url)
    grounding_keys = {_url_key(u) for u in grounding_urls}
    finding_keys = {_url_key(u) for u in finding_urls}
    if url_key in grounding_keys:
        return True
    if url_key in finding_keys and _grounding_has_redirects(grounding_urls):
        return True
    if grounding_urls and url_key in finding_keys:
        return _url_host(url) in {_url_host(u) for u in grounding_urls}
    return not grounding_urls and url_key in finding_keys


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def _best_finding_for_claim(claim: str, parsed_findings: list[dict]) -> dict | None:
    claim_tokens = _token_set(claim)
    if not claim_tokens:
        return parsed_findings[0] if parsed_findings else None

    best: tuple[int, dict] | None = None
    for finding in parsed_findings:
        score = len(claim_tokens & _token_set(finding.get("finding", "")))
        if best is None or score > best[0]:
            best = (score, finding)
    return best[1] if best else None


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.removeprefix("```json").removeprefix("```").strip()
    return text.removesuffix("```").strip()


def _load_json_response(text: str) -> dict:
    raw = _strip_json_fences(text)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning(
            "research_json_parse_failed",
            error=str(exc),
            raw=raw[:500],
            raw_chars=len(raw),
        )
        raise


def _fallback_brief_from_findings(topic: str, parsed_findings: list[dict]) -> dict:
    if not parsed_findings:
        raise RuntimeError("RESEARCH HALT: finder produced zero parseable findings.")

    def finding_at(index: int) -> dict:
        return parsed_findings[index % len(parsed_findings)]

    cold = finding_at(0)
    dinner = finding_at(1 if len(parsed_findings) > 1 else 0)
    layers = [finding_at(i + 2) for i in range(3)]
    support = [finding_at(i + 5) for i in range(5)]
    return {
        "cold_fact": cold["finding"],
        "cold_fact_source": cold["source_url"],
        "false_belief": f"Most people think {topic} is mainly an individual choice.",
        "three_layers": [f["finding"] for f in layers],
        "three_layer_sources": [f["source_url"] for f in layers],
        "supporting_facts": [f["finding"] for f in support],
        "supporting_fact_sources": [f["source_url"] for f in support],
        "twist": support[-1]["finding"],
        "dinner_fact": dinner["finding"],
        "dinner_fact_source": dinner["source_url"],
        "brief_repaired_from_findings": True,
        "brief_repair_reason": "Structure pass failed or under-filled; built directly from parsed finder output.",
    }


def _validate_and_mark_sources(brief: dict, grounding_urls: set[str], parsed_findings: list[dict]) -> dict:
    finding_urls = {f["source_url"] for f in parsed_findings}
    repairs: list[dict] = []

    def repair_source(field: str, claim: str, current_url: str, list_name: str | None = None, index: int | None = None) -> None:
        if _source_is_verified(current_url, grounding_urls, finding_urls):
            return
        replacement = _best_finding_for_claim(claim, parsed_findings)
        if not replacement:
            repairs.append({
                "field": field,
                "old_url": current_url,
                "new_url": current_url,
                "matched_finding": "",
                "unrepaired": True,
            })
            return
        new_url = replacement["source_url"]
        if list_name is None:
            brief[field] = new_url
        else:
            brief[list_name][index] = new_url
        repairs.append({
            "field": field,
            "old_url": current_url,
            "new_url": new_url,
            "matched_finding": replacement.get("finding", ""),
        })

    scalar_sources: list[tuple[str, str, str]] = [
        ("cold_fact_source", brief.get("cold_fact", ""), brief.get("cold_fact_source", "")),
        ("dinner_fact_source", brief.get("dinner_fact", ""), brief.get("dinner_fact_source", "")),
    ]

    layers = list(brief.get("three_layers", []))[:3]
    layer_sources = list(brief.get("three_layer_sources", []))[:3]
    while len(layers) < 3 and parsed_findings:
        replacement = parsed_findings[len(layers) % len(parsed_findings)]
        layers.append(replacement["finding"])
        layer_sources.append(replacement["source_url"])
    while len(layer_sources) < len(layers):
        layer_sources.append(brief.get("cold_fact_source", ""))
    brief["three_layers"] = layers
    brief["three_layer_sources"] = layer_sources

    support = list(brief.get("supporting_facts", []))[:5]
    support_sources = list(brief.get("supporting_fact_sources", []))[:5]
    while len(support) < 5 and parsed_findings:
        replacement = parsed_findings[(len(support) + 3) % len(parsed_findings)]
        support.append(replacement["finding"])
        support_sources.append(replacement["source_url"])
    while len(support_sources) < len(support):
        support_sources.append(brief.get("cold_fact_source", ""))
    brief["supporting_facts"] = support
    brief["supporting_fact_sources"] = support_sources

    for field, claim, url in scalar_sources:
        repair_source(field, claim, url)
    for i, (claim, url) in enumerate(zip(layers, layer_sources)):
        repair_source(f"three_layer_sources[{i}]", claim, url, "three_layer_sources", i)
    for i, (claim, url) in enumerate(zip(support, support_sources)):
        repair_source(f"supporting_fact_sources[{i}]", claim, url, "supporting_fact_sources", i)

    brief["source_repair_log"] = repairs
    brief["source_repair_count"] = len(repairs)
    brief["cold_fact_source_grounding_verified"] = _source_is_verified(
        brief.get("cold_fact_source", ""), grounding_urls, finding_urls
    )
    brief["dinner_fact_source_grounding_verified"] = _source_is_verified(
        brief.get("dinner_fact_source", ""), grounding_urls, finding_urls
    )
    brief["all_claim_sources_verified"] = all(
        _source_is_verified(url, grounding_urls, finding_urls)
        for url in [
            brief.get("cold_fact_source", ""),
            brief.get("dinner_fact_source", ""),
            *brief.get("three_layer_sources", []),
            *brief.get("supporting_fact_sources", []),
        ]
    )
    return brief


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> dict:
    project = state_mgr.load(slug)
    topic = project.topic
    log.info("stage_research_start", slug=slug, topic=topic)

    with state_mgr.run_stage(slug, PipelineStage.RESEARCH) as paths:
        paths.audio_dir.mkdir(parents=True, exist_ok=True)
        paths.assets_dir.mkdir(parents=True, exist_ok=True)
        paths.out_dir.mkdir(parents=True, exist_ok=True)

        # Manual brief short-circuits the Gemini call entirely
        if paths.brief_json.exists():
            log.info("manual_brief_detected_skipping_research", slug=slug)
            brief = json.loads(paths.brief_json.read_text())
            paths.auto_brief_json.write_text(json.dumps(brief, indent=2))
            log.info("manual_brief_copied_to_auto_brief", slug=slug)
            return brief

        client = genai.Client(
            vertexai=True,
            **({} if not settings.gcp_project_id else {"project": settings.gcp_project_id}),
            location=GEMINI_GCP_LOCATION,
        )

        # Pass 1 — Finder: grounding-enabled, free-form text output.
        # Model must actually fetch pages. Output is raw findings, not a brief.
        finder_response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_GROUNDING_MODEL,
            contents=_FINDER_PROMPT.format(topic=topic),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        grounding_urls = _extract_grounding_urls(finder_response)
        findings_text = finder_response.text
        parsed_findings = _parse_findings(findings_text)
        paths.discovery_json.write_text(json.dumps({
            "topic": topic,
            "grounding_urls": sorted(grounding_urls),
            "parsed_findings": parsed_findings,
            "raw_findings": findings_text,
        }, indent=2))
        log.info(
            "research_finder_done",
            slug=slug,
            grounding_urls=len(grounding_urls),
            parsed_findings=len(parsed_findings),
            findings_chars=len(findings_text),
        )
        if len(parsed_findings) < 8:
            log.warning(
                "research_finder_low_parseable_findings_continue",
                slug=slug,
                parsed_findings=len(parsed_findings),
            )
        if not parsed_findings:
            raise RuntimeError("RESEARCH HALT: finder returned zero parseable findings. Check discovery.json.")

        # Pass 2 — Architect: constrained to the findings from Pass 1, no web access.
        # Uses Gemini 2.5 Flash with thinking disabled — pure text-to-schema mapping.
        try:
            architect_response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_STRUCTURE_MODEL,
                contents=_BRIEF_FROM_FINDINGS_PROMPT.format(findings=findings_text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
            brief = _load_json_response(architect_response.text)
        except Exception as exc:
            log.warning("research_structure_failed_using_fallback_brief", slug=slug, error=str(exc))
            brief = _fallback_brief_from_findings(topic, parsed_findings)

        brief = _validate_and_mark_sources(brief, grounding_urls, parsed_findings)
        if brief.get("source_repair_count", 0):
            log.warning(
                "research_sources_repaired",
                slug=slug,
                count=brief["source_repair_count"],
            )

        paths.auto_brief_json.write_text(json.dumps(brief, indent=2))
        log.info(
            "stage_research_done",
            slug=slug,
            cold_fact_verified=brief.get("cold_fact_source_grounding_verified"),
            dinner_fact_verified=brief.get("dinner_fact_source_grounding_verified"),
        )

    return brief
