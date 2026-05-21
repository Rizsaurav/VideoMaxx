from __future__ import annotations


def test_split_long_sentence_text_caps_chunks_at_18_words():
    from vidmaxx.pipeline.stages.s02d_validate import _split_long_sentence_text

    text = " ".join(f"word{i}" for i in range(41))
    chunks = _split_long_sentence_text(text)

    assert len(chunks) == 3
    assert all(len(chunk.split()) <= 18 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)


def test_enforce_sentence_word_cap_clones_decomposer_item_metadata():
    from vidmaxx.pipeline.stages.s02d_validate import _enforce_sentence_word_cap

    raw = [{
        "id": "ch00_s00",
        "text": " ".join(f"word{i}" for i in range(20)),
        "pace": "medium",
        "emphasis_words": [],
        "pause_before_ms": 700,
        "pause_after_ms": 350,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "expected_duration_sec": 6.0,
        "visual_query": "paycheck negative net pay",
        "emotional_register": "strategic_hook",
        "pre_silence_ms": 0,
    }]

    capped = _enforce_sentence_word_cap(raw)

    assert len(capped) == 2
    assert all(len(item["text"].split()) <= 18 for item in capped)
    assert capped[0]["visual_query"] == "paycheck negative net pay"
    assert capped[1]["emotional_register"] == "strategic_hook"
    assert capped[1]["pause_before_ms"] == 0
    assert capped[0]["expected_duration_sec"] == 3.0
