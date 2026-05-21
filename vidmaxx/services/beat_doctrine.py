"""Loads the sample-derived master beat analysis for production script agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_DEFAULT_PATH = Path("sample/master_beat_analysis.json")


def load_master_beat_analysis(root: Path | None = None) -> dict[str, Any]:
    path = (root or Path.cwd()) / _DEFAULT_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def doctrine_hash(doctrine: dict[str, Any]) -> str:
    if not doctrine:
        return "no_doctrine"
    raw = json.dumps(doctrine, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _compact_list(items: list[Any], limit: int = 8) -> str:
    lines: list[str] = []
    for item in items[:limit]:
        if isinstance(item, str):
            lines.append(f"- {item}")
        elif isinstance(item, dict):
            label = item.get("source") or item.get("creator") or item.get("source_file") or "reference"
            value = item.get("pattern") or item.get("best_borrowable_moves") or item
            lines.append(f"- {label}: {value}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) if lines else "- none"


def _compact_mapping(mapping: dict[str, Any], limit: int = 8) -> str:
    if not mapping:
        return "- none"
    lines: list[str] = []
    for key, value in list(mapping.items())[:limit]:
        if isinstance(value, dict):
            inner = "; ".join(f"{k}: {v}" for k, v in value.items())
            lines.append(f"- {key}: {inner}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def architect_doctrine(doctrine: dict[str, Any]) -> str:
    if not doctrine:
        return ""
    beats = doctrine.get("unified_pacing_map", [])
    characteristics = doctrine.get("master_writing_characteristics", {})
    polish = doctrine.get("forensic_polish", {})
    beat_lines = []
    for beat in beats:
        beat_lines.append(
            "\n".join([
                f"{beat.get('beat')}. {beat.get('name')} ({beat.get('final_range')})",
                f"  Objective: {beat.get('narrative_objective')}",
                f"  Writing jobs: {', '.join(beat.get('writing_jobs', [])[:4])}",
                f"  Polish notes: {beat.get('forensic_polish', {})}",
                f"  Transition out: {beat.get('transition_out')}",
            ])
        )
    return f"""\
MASTER BEAT ANALYSIS CONTRACT — follow this over any generic YouTube structure:
{chr(10).join(beat_lines)}

Macro structure rules:
{_compact_list(characteristics.get('macro_structure_rules', []), 10)}

Tension rules:
{_compact_list(characteristics.get('tension_rules', []), 10)}

Evidence rules:
{_compact_list(characteristics.get('evidence_rules', []), 10)}

Forensic polish notes:
{_compact_mapping(polish, 10)}
"""


def chapter_doctrine(doctrine: dict[str, Any], chapter_index: int) -> str:
    if not doctrine:
        return ""
    beats = doctrine.get("unified_pacing_map", [])
    beat = next((b for b in beats if b.get("beat") == chapter_index + 1), None)
    if not beat:
        return ""
    characteristics = doctrine.get("master_writing_characteristics", {})
    contract = doctrine.get("writer_agent_contract", {})
    return f"""\
MASTER BEAT CONTRACT FOR THIS SCENE:
Beat {beat.get('beat')}: {beat.get('name')} ({beat.get('final_range')})
Narrative objective: {beat.get('narrative_objective')}
Tension job: {beat.get('tension_job')}
Information job: {beat.get('information_job')}

Writing jobs:
{_compact_list(beat.get('writing_jobs', []), 8)}

Beat-specific forensic polish:
{_compact_mapping(beat.get('forensic_polish', {}), 8)}

Sentence shape rules:
{_compact_list(beat.get('sentence_shape_rules', []), 8)}

Transition out:
- {beat.get('transition_out')}

Reference patterns:
{_compact_list(beat.get('reference_patterns', []), 6)}

Failure modes:
{_compact_list(beat.get('failure_modes', []), 6)}

Global sentence rules:
{_compact_list(characteristics.get('sentence_rules', []), 8)}

Global transition rules:
{_compact_list(characteristics.get('transition_rules', []), 8)}

Writer must-do:
{_compact_list(contract.get('must_do', []), 8)}

Writer must-not-do:
{_compact_list(contract.get('must_not_do', []), 8)}
"""


def critic_doctrine(doctrine: dict[str, Any]) -> str:
    if not doctrine:
        return ""
    characteristics = doctrine.get("master_writing_characteristics", {})
    contract = doctrine.get("writer_agent_contract", {})
    polish = doctrine.get("forensic_polish", {})
    return f"""\
MASTER WRITING CONTRACT:
Sentence classifier labels: {', '.join(contract.get('sentence_classifier_labels', []))}
Delete policy: {contract.get('delete_policy', '')}

Sentence rules:
{_compact_list(characteristics.get('sentence_rules', []), 8)}

Transition rules:
{_compact_list(characteristics.get('transition_rules', []), 8)}

Must not do:
{_compact_list(contract.get('must_not_do', []), 8)}

Forensic polish notes:
{_compact_mapping(polish, 10)}
"""


def runtime_doctrine(doctrine: dict[str, Any]) -> str:
    if not doctrine:
        return ""
    characteristics = doctrine.get("master_writing_characteristics", {})
    contract = doctrine.get("writer_agent_contract", {})
    polish = doctrine.get("forensic_polish", {})
    return f"""\
Runtime policy: {contract.get('runtime_policy', '')}

Runtime rules:
{_compact_list(characteristics.get('runtime_rules', []), 8)}

Forensic polish:
{_compact_mapping(polish, 10)}

Expansion must add evidence, mechanism, contrast, or consequence. Never padding.
"""
