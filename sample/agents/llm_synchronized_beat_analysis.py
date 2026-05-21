"""
LLM Synchronized Beat-Analysis Agent.

This is the real writing-study agent. It reads the six sample transcript JSON
files, asks an LLM to deeply analyze each creator's writing mechanics, then
synthesizes those analyses into a master pacing/writing blueprint.

Unlike synchronized_beat_analysis.py, this does not pretend keyword heuristics
understand narrative. The LLM studies structure, rhetoric, sentence craft,
pacing, tension, evidence usage, transitions, visual language, and creator
fingerprints.

Run:
    python sample/agents/llm_synchronized_beat_analysis.py

Writes:
    sample/analysis/<source>_writing_analysis.json
    sample/master_beat_analysis.json
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm


ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "json"
ANALYSIS_DIR = ROOT / "analysis"
OUT_PATH = ROOT / "master_beat_analysis.json"
TARGET_DURATION_SEC = 1020
MODEL = os.getenv("VIDMAXX_SAMPLE_ANALYSIS_MODEL", "vertex_ai/gemini-2.5-pro")


BEAT_CONTRACT: list[dict[str, Any]] = [
    {"beat": 1, "name": "The Strategic Hook", "range": "0:00 - 1:45", "focus": "Power Multiplier / Impact"},
    {"beat": 2, "name": "The Mechanism Flow", "range": "1:45 - 3:30", "focus": "Technical Supply Chain"},
    {"beat": 3, "name": "The Incongruity Drop", "range": "3:30 - 5:15", "focus": "The Slightly Wrong Detail"},
    {"beat": 4, "name": "The Dependency Link", "range": "5:15 - 7:00", "focus": "Resource/Vulnerability Pairing"},
    {"beat": 5, "name": "The Human Anchor", "range": "7:00 - 8:45", "focus": "Analog Reality / The Life"},
    {"beat": 6, "name": "The Causal Collapse", "range": "8:45 - 12:00", "focus": "Systemic Failure Map"},
    {"beat": 7, "name": "The Institutional Facade", "range": "12:00 - 14:00", "focus": "Rhetoric vs. Reality"},
    {"beat": 8, "name": "The Shadow-Truth", "range": "14:00 - 15:45", "focus": "The Darker Pivot"},
    {"beat": 9, "name": "The Synthesis", "range": "15:45 - 17:00", "focus": "Liminal Conclusion"},
]


SYSTEM_ANALYZE_ONE = """\
You are a senior narrative analyst studying elite YouTube documentary writing.

Your job is not to summarize the transcript. Your job is to reverse-engineer
the writing machine behind it so another agent can reproduce the craft.

Analyze the transcript at three levels:
1. Macro architecture: how the video moves through time.
2. Meso craft: how scenes, pivots, examples, evidence, and tension are sequenced.
3. Micro craft: sentence types, transitions, reveals, contradictions, specificity,
   rhythm, compression, expansion, and line-to-line propulsion.

Be exhaustive and concrete. Quote short evidence snippets only when necessary.
Prefer precise observations over praise. Never say "engaging" unless you explain
the mechanical reason.

Return ONLY valid JSON matching the requested shape.\
"""


ANALYZE_ONE_PROMPT = """\
Source file: {file}
Title: {title}
Creator/theme: {creator}
Duration: {duration}

Transcript segments are normalized. Each row is:
[timestamp | normalized_percent] text

{transcript}

Return JSON with this exact top-level shape:
{{
  "source_file": "{file}",
  "title": "{title}",
  "creator": "{creator}",
  "duration": "{duration}",
  "executive_fingerprint": {{
    "one_sentence_dna": "what makes this video's writing work",
    "dominant_engine": "curiosity | dread | logistics | mystery | contradiction | awe | critique | other",
    "core_viewer_question": "the implicit question keeping viewers watching",
    "primary_tension_source": "what pressure drives the video forward"
  }},
  "temporal_beat_map": [
    {{
      "beat": 1,
      "beat_name": "mapped beat name",
      "source_start_timestamp": "m:ss or h:mm:ss",
      "source_end_timestamp": "m:ss or h:mm:ss",
      "normalized_start_pct": 0.0,
      "normalized_end_pct": 0.0,
      "narrative_function": "what this section does",
      "tension_state": "setup | escalation | rupture | explanation | consequence | reversal | synthesis",
      "dominant_writing_moves": ["specific move names"],
      "evidence_snippets": ["short transcript snippets that prove the mapping"]
    }}
  ],
  "writing_characteristics": {{
    "hook_design": {{
      "opening_move": "specific technique",
      "time_to_first_specific_entity_or_number": "timestamp or estimate",
      "why_it_creates_forward_motion": "mechanical explanation",
      "reusable_rule": "rule a writer agent can follow"
    }},
    "information_architecture": {{
      "exposition_strategy": "how context is introduced without flattening tension",
      "data_to_story_ratio": "qualitative ratio with explanation",
      "how_complexity_is_layered": ["observable technique"],
      "what_is_delayed": ["facts or explanations intentionally withheld"],
      "what_is_frontloaded": ["facts or stakes put early"]
    }},
    "tension_mechanics": {{
      "open_loops": ["unanswered questions created"],
      "pivot_types": ["but/however/then/the weird part/etc."],
      "escalation_pattern": "how tension rises",
      "release_pattern": "how tension is resolved or transformed",
      "failure_or_reversal_timestamp": "where the system visibly fails or flips"
    }},
    "sentence_level_craft": {{
      "sentence_length_profile": "short/medium/long pattern and why",
      "reveal_sentences": ["examples of reveal mechanics"],
      "contradiction_sentences": ["examples of contradiction mechanics"],
      "entity_sentences": ["examples of specificity/entity placement"],
      "question_answer_patterns": ["implicit or explicit question-answer moves"],
      "forbidden_filler_patterns_to_avoid": ["phrases or structures this creator does not rely on"]
    }},
    "transition_system": {{
      "how_sections_turn": "how the script moves without generic transitions",
      "common_transition_shapes": ["move from X to Y by doing Z"],
      "best_transition_examples": ["short snippets or paraphrases"]
    }},
    "evidence_and_specificity": {{
      "number_usage": "how numbers are introduced and interpreted",
      "named_entity_usage": "how people/institutions/places are deployed",
      "source_or_authority_handling": "how credibility is built",
      "concrete_detail_density": "low | medium | high with explanation"
    }},
    "visual_language": {{
      "objects_places_and_images_the_script_implies": ["concrete visual categories"],
      "incongruity_moves": ["ordinary thing plus wrong detail"],
      "visual_query_rules_for_pipeline": ["rules for B-roll/visual generation"]
    }},
    "voice_and_rhythm": {{
      "tone": "dominant tonal blend",
      "cadence": "how rhythm changes over time",
      "humor_or_irony": "how irony/humor appears, if at all",
      "emotional_temperature_curve": "how feeling changes beat by beat"
    }},
    "ending_design": {{
      "final_move": "what the ending does",
      "resolution_level": "closed | open | liminal | indictment | invitation",
      "last_impression": "what feeling/idea remains"
    }},
    "transplantable_rules": [
      "specific rule for writer agents"
    ],
    "anti_patterns": [
      "what not to copy from this creator/video"
    ]
  }},
  "agent_notes_for_17_minute_systemic_video": {{
    "what_to_copy": ["specific craft operations"],
    "what_to_avoid": ["failure modes"],
    "best_use_in_our_pipeline": "where this creator's DNA should influence the 9-beat structure"
  }}
}}

The temporal_beat_map must include 9 beats. If this source does not have the
same shape, map the closest functional equivalent and explain the mismatch in
that beat's narrative_function.\
"""


SYSTEM_SYNTHESIZE = """\
You are the coordinating architect for a multi-agent YouTube writing pipeline.

You receive six deep writing analyses. Synthesize them into a master blueprint
that writer agents can execute. Do not average away the interesting differences.
Preserve specific craft rules, failure modes, and timing constraints.

Return ONLY valid JSON.\
"""


SYNTHESIS_PROMPT = """\
Target duration: 17:00 / 1020 seconds.

Locked 9-beat contract:
{beat_contract}

Per-source analyses:
{analyses}

Create sample/master_beat_analysis.json content with this exact top-level shape:
{{
  "agent": "LLM Synchronized Beat-Analysis Agent",
  "version": 2,
  "target_duration_sec": 1020,
  "files_analyzed": ["..."],
  "unified_pacing_map": [
    {{
      "beat": 1,
      "name": "The Strategic Hook",
      "final_range": "0:00 - 1:45",
      "narrative_objective": "specific objective",
      "writing_jobs": ["what writer must accomplish"],
      "tension_job": "what pressure this beat creates",
      "information_job": "what facts are allowed here",
      "sentence_shape_rules": ["micro-rules"],
      "transition_out": "how to move to next beat",
      "visual_hook_rules": ["incongruity/visual rules"],
      "reference_patterns": [
        {{"source": "creator/file", "pattern": "what to borrow"}}
      ],
      "failure_modes": ["how this beat goes flat"]
    }}
  ],
  "master_writing_characteristics": {{
    "macro_structure_rules": ["rules"],
    "tension_rules": ["rules"],
    "evidence_rules": ["rules"],
    "sentence_rules": ["rules"],
    "transition_rules": ["rules"],
    "visual_rules": ["rules"],
    "runtime_rules": ["rules"],
    "voice_rules": ["rules"]
  }},
  "creator_fingerprint_library": [
    {{
      "source_file": "file",
      "creator": "creator",
      "best_borrowable_moves": ["moves"],
      "dangerous_to_copy": ["things that fail outside original context"],
      "where_to_use_in_our_9_beats": ["beat usage"]
    }}
  ],
  "writer_agent_contract": {{
    "must_do": ["non-negotiable execution rules"],
    "must_not_do": ["anti-patterns"],
    "scene_planning_checklist": ["what planner checks before drafting"],
    "sentence_classifier_labels": ["REVEAL", "CONTRADICT", "QUESTION", "ENTITY", "FILLER"],
    "delete_policy": "how FILLER gets handled",
    "runtime_policy": "how to preserve 17 minutes without padding"
  }}
}}

Make this detailed enough that a writing agent can plan and enforce every scene
without asking a human what the style means.\
"""


@dataclass
class Transcript:
    file: str
    title: str
    creator: str
    duration_sec: float
    rows: list[dict[str, Any]]


def parse_timestamp(raw: str) -> float:
    parts = [int(p) for p in raw.strip().split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    raise ValueError(f"Unsupported timestamp: {raw!r}")


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    start_obj = text.find("{") if "{" in text else len(text)
    start_arr = text.find("[") if "[" in text else len(text)
    start = min(start_obj, start_arr)
    end = max(text.rfind("}"), text.rfind("]")) + 1
    if start < end:
        text = text[start:end]
    return json.loads(text)


def load_transcripts() -> list[Transcript]:
    transcripts: list[Transcript] = []
    for path in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        raw_rows = [
            item for item in data.get("transcript_data", [])
            if item.get("timestamp") and str(item.get("text", "")).strip()
        ]
        if not raw_rows:
            continue
        seconds = [parse_timestamp(item["timestamp"]) for item in raw_rows]
        duration = max(seconds) or 1.0
        rows = [
            {
                "timestamp": item["timestamp"],
                "sec": round(sec, 2),
                "pct": round((sec / duration) * 100, 2),
                "text": str(item["text"]).strip(),
            }
            for item, sec in zip(raw_rows, seconds)
        ]
        transcripts.append(
            Transcript(
                file=path.name,
                title=data.get("title", path.stem),
                creator=data.get("creator", data.get("theme", path.stem)),
                duration_sec=duration,
                rows=rows,
            )
        )
    return transcripts


def transcript_for_prompt(transcript: Transcript) -> str:
    return "\n".join(
        f"[{row['timestamp']} | {row['pct']:05.2f}%] {row['text']}"
        for row in transcript.rows
    )


async def call_json(system: str, prompt: str, *, max_tokens: int = 65536) -> Any:
    response = await litellm.acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return extract_json(response.choices[0].message.content or "")


async def analyze_one(transcript: Transcript) -> dict[str, Any]:
    out_path = ANALYSIS_DIR / f"{Path(transcript.file).stem}_writing_analysis.json"
    if out_path.exists() and os.getenv("VIDMAXX_SAMPLE_FORCE", "") != "1":
        return json.loads(out_path.read_text())

    prompt = (
        ANALYZE_ONE_PROMPT
        .replace("{file}", transcript.file)
        .replace("{title}", transcript.title.replace('"', "'"))
        .replace("{creator}", transcript.creator.replace('"', "'"))
        .replace("{duration}", fmt_time(transcript.duration_sec))
        .replace("{transcript}", transcript_for_prompt(transcript))
    )
    analysis = await call_json(SYSTEM_ANALYZE_ONE, prompt)
    out_path.write_text(json.dumps(analysis, indent=2))
    return analysis


async def synthesize(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = (
        SYNTHESIS_PROMPT
        .replace("{beat_contract}", json.dumps(BEAT_CONTRACT, indent=2))
        .replace("{analyses}", json.dumps(analyses, indent=2))
    )
    result = await call_json(SYSTEM_SYNTHESIZE, prompt, max_tokens=65536)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


async def run() -> dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    transcripts = load_transcripts()
    analyses = await asyncio.gather(*[analyze_one(t) for t in transcripts])
    return await synthesize(list(analyses))


if __name__ == "__main__":
    result = asyncio.run(run())
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote per-source analyses to {ANALYSIS_DIR}")
    print(f"Analyzed {len(result.get('files_analyzed', []))} files")
