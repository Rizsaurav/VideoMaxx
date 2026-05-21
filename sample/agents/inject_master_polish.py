"""
Inject Forensic Polish notes into sample/master_beat_analysis.json.

This is intentionally light-touch. It does not replace the master analysis or
turn polish into global law. It adds beat-level refinement notes so production
agents preserve the original structure while sharpening specific moments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_beat_analysis.json"


FORENSIC_POLISH: dict[str, Any] = {
    "beat_6_redundancy_swap": {
        "target_beat": 6,
        "intent": "Kill redundancy without changing the beat's original collapse function.",
        "note": (
            "If the script repeats the same metric, cut the second use. Replace it "
            "with a distinct systemic failure example that proves a different part "
            "of the mechanism."
        ),
        "example_pattern": (
            "Use an output-versus-human-outcome gap: the system improves by its own "
            "metric while the viewer-facing promise gets worse."
        ),
    },
    "civic_comparison_injection": {
        "target_beats": [2, 5, 6],
        "intent": "Use human-scale comparison only when a metric needs emotional weight.",
        "note": (
            "For Beats 2, 5, and 6, add a Civic Equivalent or human-scale comparison "
            "only when a number needs to hit emotionally. Leave ordinary support "
            "numbers clean and direct."
        ),
        "comparison_modes": [
            "household consequence",
            "time-to-earn equivalent",
            "local civic budget equivalent",
            "physical-scale equivalent",
            "daily-life tradeoff",
        ],
    },
    "beat_8_liminal_pivot": {
        "target_beats": [8, 9],
        "intent": "Stop the ending from becoming a summary.",
        "note": (
            "Use Beat 8 to pivot from situation summary into the darker realization. "
            "Beat 9 should land on the idea that the system is not broken; it is "
            "calibrated. This is an ending direction, not mandatory wording."
        ),
        "ending_feel": (
            "Haunting, cyclical, and unresolved. No solutions. No checklist. "
            "No inspirational release."
        ),
    },
}


def _beat_by_number(data: dict[str, Any], beat_num: int) -> dict[str, Any]:
    for beat in data.get("unified_pacing_map", []):
        if beat.get("beat") == beat_num:
            return beat
    raise ValueError(f"Beat {beat_num} not found in unified_pacing_map")


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _remove_previous_overcorrection(data: dict[str, Any]) -> None:
    data.pop("protocol_injection", None)

    for beat in data.get("unified_pacing_map", []):
        beat.pop("protocol_constraints", None)
        beat.pop("causal_chain_engine", None)
        beat.pop("institutional_bias_audit", None)
        beat.pop("endgame_protocol", None)
        beat.pop("forensic_polish", None)

        beat["writing_jobs"] = [
            item for item in beat.get("writing_jobs", [])
            if item not in {
                "Include one counter-intuitive fact that invalidates the viewer's mainstream assumption.",
                "Build the collapse through the triad: Normal Baseline → Divergence → Compounding Collapse.",
                "Use an output-versus-human-outcome gap as the systemic failure example instead of repeating the same metric.",
                "Juxtapose neutral institutional language against brutal lived reality while the narrator stays detached.",
                "Pivot from summary into the darker premise: the system is not broken, it is calibrated.",
                "End on a cyclical realization that leaves the viewer trapped inside the mechanism.",
            }
        ]
        beat["failure_modes"] = [
            item for item in beat.get("failure_modes", [])
            if item not in {
                "Raw numbers appear without a civic or household equivalent.",
                "The beat confirms the mainstream assumption instead of overturning it.",
                "The ending offers solutions, advice, or inspirational release.",
            }
        ]
        beat["sentence_shape_rules"] = [
            item for item in beat.get("sentence_shape_rules", [])
            if item != "Final sentence must imply: the system is not broken; it is calibrated."
        ]


def inject(data: dict[str, Any]) -> dict[str, Any]:
    _remove_previous_overcorrection(data)
    data["forensic_polish"] = FORENSIC_POLISH

    beat6 = _beat_by_number(data, 6)
    beat6["forensic_polish"] = {
        "redundancy_swap": FORENSIC_POLISH["beat_6_redundancy_swap"],
        "civic_comparison": FORENSIC_POLISH["civic_comparison_injection"],
    }
    _append_unique(
        beat6["writing_jobs"],
        "If a key metric repeats, replace the repetition with a distinct systemic failure example.",
    )

    for beat_num in (2, 5):
        beat = _beat_by_number(data, beat_num)
        beat["forensic_polish"] = {
            "civic_comparison": FORENSIC_POLISH["civic_comparison_injection"],
        }

    beat8 = _beat_by_number(data, 8)
    beat8["forensic_polish"] = {
        "liminal_pivot": FORENSIC_POLISH["beat_8_liminal_pivot"],
    }
    _append_unique(
        beat8["writing_jobs"],
        "Pivot from explaining the situation into the darker realization the ending will crystallize.",
    )

    beat9 = _beat_by_number(data, 9)
    beat9["forensic_polish"] = {
        "liminal_pivot": FORENSIC_POLISH["beat_8_liminal_pivot"],
    }
    _append_unique(
        beat9["sentence_shape_rules"],
        "Final turn should imply: the system is not broken, it is calibrated.",
    )
    _append_unique(
        beat9["failure_modes"],
        "Ending as a summary instead of a haunting liminal realization.",
    )

    return data


def main() -> None:
    data = json.loads(MASTER_PATH.read_text())
    updated = inject(data)
    MASTER_PATH.write_text(json.dumps(updated, indent=2))
    print(f"Updated {MASTER_PATH}")


if __name__ == "__main__":
    main()
