from __future__ import annotations


def test_master_beat_constants_match_17_minute_contract():
    from vidmaxx.config.constants import SCENE_DURATIONS_SEC, SCENE_LABELS, TARGET_VIDEO_DURATION_SEC

    assert TARGET_VIDEO_DURATION_SEC == 1020
    assert sum(SCENE_DURATIONS_SEC) == TARGET_VIDEO_DURATION_SEC
    assert SCENE_LABELS == [
        "The Strategic Hook",
        "The Mechanism Flow",
        "The Incongruity Drop",
        "The Dependency Link",
        "The Human Anchor",
        "The Causal Collapse",
        "The Institutional Facade",
        "The Shadow-Truth",
        "The Synthesis",
    ]


def test_beat_doctrine_loads_and_compacts_master_analysis():
    from vidmaxx.services.beat_doctrine import (
        architect_doctrine,
        chapter_doctrine,
        critic_doctrine,
        load_master_beat_analysis,
        runtime_doctrine,
    )

    doctrine = load_master_beat_analysis()
    assert doctrine
    assert "unified_pacing_map" in doctrine

    architect_text = architect_doctrine(doctrine)
    chapter_text = chapter_doctrine(doctrine, 5)
    critic_text = critic_doctrine(doctrine)
    runtime_text = runtime_doctrine(doctrine)

    assert "MASTER BEAT ANALYSIS CONTRACT" in architect_text
    assert "The Causal Collapse" in chapter_text
    assert "Sentence classifier labels" in critic_text
    assert "Civic Equivalent" in architect_text
    assert "output-versus-human-outcome gap" in chapter_text
    assert "system is not broken" in critic_text
    assert "only when a number needs to hit emotionally" in runtime_text


def test_master_json_contains_polish_notes_without_protocol_overcorrection():
    from vidmaxx.services.beat_doctrine import load_master_beat_analysis

    doctrine = load_master_beat_analysis()
    beats = {b["beat"]: b for b in doctrine["unified_pacing_map"]}

    assert "protocol_injection" not in doctrine
    assert "forensic_polish" in doctrine
    assert "protocol_constraints" not in beats[8]
    assert "endgame_protocol" not in beats[9]
    assert "forensic_polish" in beats[6]
    assert "forensic_polish" in beats[8]
    assert "forensic_polish" in beats[9]
    assert "wealth-share drop" not in str(doctrine)
    assert "Productivity-Pay Gap" not in str(doctrine)
    assert "yearly grocery budget" not in str(doctrine)
    assert "every major figure" not in str(doctrine)
    assert "same metric" in beats[6]["forensic_polish"]["redundancy_swap"]["note"]
    assert "only when a number needs to hit emotionally" in beats[2]["forensic_polish"]["civic_comparison"]["note"]
    assert "only when a number needs to hit emotionally" in beats[5]["forensic_polish"]["civic_comparison"]["note"]
    assert "only when a number needs to hit emotionally" in beats[6]["forensic_polish"]["civic_comparison"]["note"]
    assert any("calibrated" in rule for rule in beats[9]["sentence_shape_rules"])
