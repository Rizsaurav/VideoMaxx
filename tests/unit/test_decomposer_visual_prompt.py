from __future__ import annotations


def test_decomposer_visual_query_prompt_enforces_incongruity_principle():
    from vidmaxx.services.llm.prompts import decomposer

    assert "Incongruity Principle" in decomposer._SYSTEM
    assert "slightly wrong" in decomposer._SYSTEM
    assert "paycheck with negative net pay" in decomposer._SYSTEM
    assert "shrinking pie chart" in decomposer._SYSTEM
