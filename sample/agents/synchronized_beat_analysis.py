"""
Synchronized Beat-Analysis Agent.

Reads the six sample transcript JSON files and produces a unified temporal
beat map for a 17-minute narrative video.

The analysis is deterministic by design: no LLM calls, no external APIs. The
agents use timestamp normalization, keyword/position scoring, and aggregate
statistics so the output is reproducible and auditable.

Run:
    python sample/agents/synchronized_beat_analysis.py

Writes:
    sample/master_beat_analysis.json
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "json"
OUT_PATH = ROOT / "master_beat_analysis.json"
TARGET_DURATION_SEC = 1020


BEAT_BLUEPRINT: list[dict[str, Any]] = [
    {
        "beat": 1,
        "name": "The Strategic Hook",
        "objective": "Power Multiplier",
        "focus": "Impact",
        "start_sec": 0,
        "end_sec": 105,
        "keywords": ["single", "power", "change", "strongest", "famous", "impact", "enough"],
    },
    {
        "beat": 2,
        "name": "The Mechanism Flow",
        "objective": "Technical Supply Chain",
        "focus": "How the system works",
        "start_sec": 105,
        "end_sec": 210,
        "keywords": ["how", "works", "system", "process", "chain", "function", "because", "built"],
    },
    {
        "beat": 3,
        "name": "The Incongruity Drop",
        "objective": "The Slightly Wrong Detail",
        "focus": "Something does not fit",
        "start_sec": 210,
        "end_sec": 315,
        "keywords": ["but", "however", "strange", "wrong", "odd", "secret", "problem", "instead"],
    },
    {
        "beat": 4,
        "name": "The Dependency Link",
        "objective": "Resource/Vulnerability Pairing",
        "focus": "What the system depends on",
        "start_sec": 315,
        "end_sec": 420,
        "keywords": ["depends", "requires", "resource", "vulnerable", "supply", "needs", "relies"],
    },
    {
        "beat": 5,
        "name": "The Human Anchor",
        "objective": "Analog Reality",
        "focus": "The life inside the system",
        "start_sec": 420,
        "end_sec": 525,
        "keywords": ["people", "person", "you", "life", "child", "family", "worker", "users"],
    },
    {
        "beat": 6,
        "name": "The Causal Collapse",
        "objective": "Systemic Failure Map",
        "focus": "Failure chain",
        "start_sec": 525,
        "end_sec": 720,
        "keywords": ["failed", "collapse", "crisis", "cost", "decline", "couldn't", "failure", "broke"],
    },
    {
        "beat": 7,
        "name": "The Institutional Facade",
        "objective": "Rhetoric vs. Reality",
        "focus": "Official story versus actual behavior",
        "start_sec": 720,
        "end_sec": 840,
        "keywords": ["company", "government", "official", "announced", "ceo", "law", "policy", "public"],
    },
    {
        "beat": 8,
        "name": "The Shadow-Truth",
        "objective": "The Darker Pivot",
        "focus": "The truth under the stated truth",
        "start_sec": 840,
        "end_sec": 945,
        "keywords": ["truth", "actually", "hidden", "secret", "darker", "real", "worse", "behind"],
    },
    {
        "beat": 9,
        "name": "The Synthesis",
        "objective": "Liminal Conclusion",
        "focus": "Meaning after the reveal",
        "start_sec": 945,
        "end_sec": 1020,
        "keywords": ["ultimately", "means", "future", "end", "still", "why", "because", "everything"],
    },
]


DATA_RE = re.compile(r"\d|%|\$|\b(million|billion|trillion|percent|miles|kilometers|year|years)\b", re.I)
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
NARRATIVE_RE = re.compile(r"\b(you|we|i|people|person|family|story|life|remember|feel|watched|wanted)\b", re.I)
FAILURE_RE = re.compile(
    r"\b(fail|failed|failure|collapse|crisis|broke|broken|cost|decline|lost|scandal|problem|wrong)\b",
    re.I,
)


@dataclass
class TranscriptSegment:
    sec: float
    pct: float
    text: str


@dataclass
class Transcript:
    file: str
    title: str
    creator: str
    duration_sec: float
    segments: list[TranscriptSegment]
    skipped_segments: int = 0


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


def load_transcripts() -> list[Transcript]:
    transcripts: list[Transcript] = []
    for path in sorted(JSON_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        raw_segments = data.get("transcript_data", [])
        if not raw_segments:
            continue
        usable_items = [
            item for item in raw_segments
            if item.get("timestamp") and str(item.get("text", "")).strip()
        ]
        skipped = len(raw_segments) - len(usable_items)
        starts = [parse_timestamp(item["timestamp"]) for item in usable_items]
        duration = max(starts) or 1.0
        segments = [
            TranscriptSegment(
                sec=sec,
                pct=sec / duration,
                text=str(item.get("text", "")).strip(),
            )
            for sec, item in zip(starts, usable_items)
        ]
        transcripts.append(
            Transcript(
                file=path.name,
                title=data.get("title", path.stem),
                creator=data.get("creator", data.get("theme", path.stem)),
                duration_sec=duration,
                segments=segments,
                skipped_segments=skipped,
            )
        )
    return transcripts


def positional_score(pct: float, target_pct: float, width: float = 0.13) -> float:
    return math.exp(-((pct - target_pct) ** 2) / (2 * width * width))


def beat_score(segment: TranscriptSegment, beat: dict[str, Any]) -> float:
    target_pct = ((beat["start_sec"] + beat["end_sec"]) / 2) / TARGET_DURATION_SEC
    text = segment.text.lower()
    keyword_hits = sum(1 for keyword in beat["keywords"] if keyword in text)
    return keyword_hits * 2.0 + positional_score(segment.pct, target_pct)


def detect_beat_positions(transcript: Transcript) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    previous_pct = 0.0
    for beat in BEAT_BLUEPRINT:
        if beat["beat"] == 1:
            chosen = transcript.segments[0]
        else:
            best = max(transcript.segments, key=lambda s: beat_score(s, beat))
            prior_pct = beat["start_sec"] / TARGET_DURATION_SEC
            # Blend evidence with the known narrative prior. This prevents one
            # stray keyword from pulling a beat wildly out of sequence.
            detected_pct = (best.pct * 0.6) + (prior_pct * 0.4)
            detected_pct = max(previous_pct + 0.01, min(0.98, detected_pct))
            chosen = TranscriptSegment(
                sec=detected_pct * transcript.duration_sec,
                pct=detected_pct,
                text=best.text,
            )
        previous_pct = chosen.pct
        detections.append({
            "beat": beat["beat"],
            "name": beat["name"],
            "detected_pct": round(chosen.pct * 100, 2),
            "detected_sec_in_source": round(chosen.sec, 2),
            "detected_timestamp_in_source": fmt_time(chosen.sec),
            "evidence_text": chosen.text[:220],
        })
    return detections


def classify_data_to_narrative(transcript: Transcript) -> dict[str, Any]:
    data_count = 0
    narrative_count = 0
    hybrid_count = 0
    for segment in transcript.segments:
        has_data = bool(DATA_RE.search(segment.text) or ENTITY_RE.search(segment.text))
        has_narrative = bool(NARRATIVE_RE.search(segment.text))
        if has_data and has_narrative:
            hybrid_count += 1
        elif has_data:
            data_count += 1
        elif has_narrative:
            narrative_count += 1
    total = max(1, data_count + narrative_count + hybrid_count)
    return {
        "data_segments": data_count,
        "narrative_segments": narrative_count,
        "hybrid_segments": hybrid_count,
        "data_ratio": round((data_count + hybrid_count * 0.5) / total, 3),
        "narrative_ratio": round((narrative_count + hybrid_count * 0.5) / total, 3),
    }


def detect_failure_point(transcript: Transcript) -> dict[str, Any]:
    candidates = [
        segment for segment in transcript.segments
        if FAILURE_RE.search(segment.text)
    ]
    if not candidates:
        target = 525 / TARGET_DURATION_SEC
        chosen = min(transcript.segments, key=lambda s: abs(s.pct - target))
    else:
        chosen = max(
            candidates,
            key=lambda s: (
                len(FAILURE_RE.findall(s.text)),
                positional_score(s.pct, 525 / TARGET_DURATION_SEC, width=0.22),
            ),
        )
    return {
        "pct": round(chosen.pct * 100, 2),
        "source_timestamp": fmt_time(chosen.sec),
        "source_sec": round(chosen.sec, 2),
        "evidence_text": chosen.text[:220],
    }


def aggregate_unified_map(per_transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unified: list[dict[str, Any]] = []
    for beat in BEAT_BLUEPRINT:
        observed = [
            t["beat_detections"][beat["beat"] - 1]["detected_pct"]
            for t in per_transcript
        ]
        observed_start_pct = statistics.median(observed) / 100
        # The observed median is diagnostic evidence. The Writer contract stays
        # locked to the requested 17-minute blueprint so downstream agents do
        # not drift into a different runtime shape.
        final_start_sec = beat["start_sec"]
        unified.append({
            "beat": beat["beat"],
            "name": beat["name"],
            "narrative_objective": beat["objective"],
            "logical_focus": beat["focus"],
            "observed_median_start_pct": round(observed_start_pct * 100, 2),
            "final_start_sec": final_start_sec,
            "final_start_timestamp": fmt_time(final_start_sec),
            "target_start_sec": beat["start_sec"],
            "target_end_sec": beat["end_sec"],
            "target_range": f"{fmt_time(beat['start_sec'])} - {fmt_time(beat['end_sec'])}",
            "buffer_sec": 10,
        })
    for i, beat in enumerate(unified):
        end_sec = unified[i + 1]["final_start_sec"] if i + 1 < len(unified) else TARGET_DURATION_SEC
        beat["final_end_sec"] = end_sec
        beat["final_range"] = f"{beat['final_start_timestamp']} - {fmt_time(end_sec)}"
    return unified


def run() -> dict[str, Any]:
    transcripts = load_transcripts()
    per_transcript: list[dict[str, Any]] = []
    for transcript in transcripts:
        beat_detections = detect_beat_positions(transcript)
        pacing = classify_data_to_narrative(transcript)
        failure = detect_failure_point(transcript)
        per_transcript.append({
            "file": transcript.file,
            "title": transcript.title,
            "creator": transcript.creator,
            "duration_sec": round(transcript.duration_sec, 2),
            "duration_timestamp": fmt_time(transcript.duration_sec),
            "segment_count": len(transcript.segments),
            "skipped_segments": transcript.skipped_segments,
            "beat_detections": beat_detections,
            "pacing_profile": pacing,
            "system_failure_point": failure,
        })

    unified_map = aggregate_unified_map(per_transcript)
    failure_pcts = [t["system_failure_point"]["pct"] for t in per_transcript]
    pivot_pcts = [
        t["beat_detections"][5]["detected_pct"]  # beat 6, Causal Collapse
        for t in per_transcript
    ]
    output = {
        "agent": "Synchronized Beat-Analysis Agent",
        "version": 1,
        "input_dir": str(JSON_DIR),
        "target_duration_sec": TARGET_DURATION_SEC,
        "target_duration_timestamp": fmt_time(TARGET_DURATION_SEC),
        "files_analyzed": [t["file"] for t in per_transcript],
        "agent_outputs": {
            "agent_a_structural": {
                "task": "Average hook-to-pivot timing across all samples.",
                "pivot_definition": "Start of Beat 6: The Causal Collapse.",
                "average_pivot_pct": round(statistics.mean(pivot_pcts), 2),
                "median_pivot_pct": round(statistics.median(pivot_pcts), 2),
                "recommended_pivot_timestamp": fmt_time((statistics.median(pivot_pcts) / 100) * TARGET_DURATION_SEC),
            },
            "agent_b_pacing": {
                "task": "Estimate data-to-narrative ratio per creator.",
                "profiles": [
                    {
                        "file": t["file"],
                        "creator": t["creator"],
                        **t["pacing_profile"],
                    }
                    for t in per_transcript
                ],
            },
            "agent_c_tension_mapper": {
                "task": "Detect where the system visibly fails in each sample.",
                "average_failure_pct": round(statistics.mean(failure_pcts), 2),
                "median_failure_pct": round(statistics.median(failure_pcts), 2),
                "recommended_failure_timestamp": fmt_time((statistics.median(failure_pcts) / 100) * TARGET_DURATION_SEC),
                "per_file": [
                    {
                        "file": t["file"],
                        "creator": t["creator"],
                        **t["system_failure_point"],
                    }
                    for t in per_transcript
                ],
            },
        },
        "unified_pacing_map": unified_map,
        "per_transcript_analysis": per_transcript,
        "writer_agent_contract": {
            "rule": "Every generated script must follow the nine beats in order.",
            "duration_policy": "Use final_range as the target with +/-10 seconds tolerance per beat.",
            "anti_repetition_policy": "No two adjacent beats may use the same logical focus.",
            "sentence_policy": "Every sentence must reveal, contradict, raise/answer a question, or drop a concrete number/entity.",
            "visual_policy": "Use incongruous concrete visuals: ordinary object plus one wrong or alarming detail.",
        },
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    result = run()
    print(f"Wrote {OUT_PATH}")
    print(f"Analyzed {len(result['files_analyzed'])} files")
