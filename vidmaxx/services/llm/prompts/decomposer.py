"""
Builds the (system, user) prompt pair for the decomposer pass (Stage 3c).

Input:  chapter prose + ChapterOutline (for energy/index/title context)
Output: JSON array matching the Sentence schema

Gemini 2.5 Flash handles this — it's mechanical extraction, not creative work.
"""

from vidmaxx.config.constants import (
    PAUSE_MS_COMMA,
    PAUSE_MS_EXCLAMATION,
    PAUSE_MS_PARAGRAPH_BREAK,
    PAUSE_MS_PERIOD,
    PAUSE_MS_QUESTION,
    PATTERN_INTERRUPT_INDICES,
    PATTERN_INTERRUPT_PRE_SILENCE_MS,
    TTS_DEFAULT_CFG_WEIGHT,
    TTS_DEFAULT_EXAGGERATION,
    TTS_DRAMATIC_EXAGGERATION,
    TTS_FAST_CFG_WEIGHT,
)
from vidmaxx.models.script import ChapterOutline

# Scene title → emotional register mapping (must match VISUAL_REGISTER_QUERIES keys)
_SCENE_REGISTER: dict[str, str] = {
    "The Strategic Hook":       "strategic_hook",
    "The Mechanism Flow":       "mechanism_flow",
    "The Incongruity Drop":     "incongruity_drop",
    "The Dependency Link":      "dependency_link",
    "The Human Anchor":         "human_anchor",
    "The Causal Collapse":      "causal_collapse",
    "The Institutional Facade": "institutional_facade",
    "The Shadow-Truth":         "shadow_truth",
    "The Synthesis":            "synthesis",
    # Legacy titles kept for old cached outlines.
    "COLD FACT":            "clinical_shock",
    "FALSE REALITY":        "false_normal",
    "LAYER ONE: SURFACE":   "infrastructure",
    "PATTERN INTERRUPT 1":  "pattern_interrupt_1",
    "LAYER TWO: MECHANISM": "systemic_flow",
    "PATTERN INTERRUPT 2":  "pattern_interrupt_2",
    "LAYER THREE: YOU":     "personal_consequence",
    "THE TWIST":            "twist",
    "HARD EXIT":            "exit",
}

_SYSTEM = f"""\
You are a TTS script decomposer. Split prose narration into sentence-level JSON objects.

Output a JSON array. Each element matches this schema exactly:

{{
  "id": "<string>",
  "text": "<the sentence, verbatim from the prose>",
  "pace": "<slow|medium|fast>",
  "emphasis_words": ["<word>", ...],
  "pause_before_ms": <integer>,
  "pause_after_ms": <integer>,
  "exaggeration": <float 0.0-1.0>,
  "cfg_weight": <float 0.0-1.0>,
  "expected_duration_sec": <float>,
  "visual_query": "<3-8 word concrete visual description for B-roll>",
  "emotional_register": "<one of the eight registers — provided per scene below>",
  "pre_silence_ms": <integer — 0 for most sentences, 1000 for first sentence of PATTERN INTERRUPT scenes>,
  "visualization_type": "<none|counting_number|comparison|flow|timeline|shrink>"
}}

Rules:

IDs: format is "ch{{chapter_index:02d}}_s{{sentence_index:02d}}" — zero-padded, 0-based.

Splitting: split on sentence-ending punctuation (. ! ?). Never split mid-clause.
  Preserve the punctuation at the end of each sentence.

pace:
  - "slow" for emotional beats, surprising reveals, or when the chapter energy is "low"
  - "fast" for lists, momentum-building, high-energy sections
  - "medium" for everything else

emphasis_words: 1-3 words per sentence that carry the most meaning. Empty list is fine.

pause_after_ms defaults by ending punctuation:
  . = {PAUSE_MS_PERIOD}
  , (sentence ending with comma — rare) = {PAUSE_MS_COMMA}
  ? = {PAUSE_MS_QUESTION}
  ! = {PAUSE_MS_EXCLAMATION}
  paragraph break (last sentence before a new paragraph) = {PAUSE_MS_PARAGRAPH_BREAK}

pause_before_ms: 0 for most sentences. Use {PAUSE_MS_PARAGRAPH_BREAK} only for the first
  sentence after a paragraph break that immediately follows a high-energy beat.

exaggeration: {TTS_DEFAULT_EXAGGERATION} default. Use {TTS_DRAMATIC_EXAGGERATION} for dramatic/surprising sentences.
cfg_weight: {TTS_DEFAULT_CFG_WEIGHT} default. Use {TTS_FAST_CFG_WEIGHT} for fast-paced sentences.

expected_duration_sec: estimate based on word count. Average speaking pace is ~140 wpm
  at "medium" pace. Adjust: slow=110 wpm, fast=170 wpm.

visual_query: a concrete, searchable phrase for stock footage. Not abstract.
  Use the Incongruity Principle: show an everyday object, place, or document
  with one slightly wrong, alarming, or impossible detail.
  Prefer inspectable physical scenes over charts, graphs, sad people, or generic metaphors.
  BAD: "concept of time", "technology metaphor", "sad person money problems", "shrinking pie chart"
  GOOD: "paycheck with negative net pay", "piggy bank leaking coins", "office cubicle with locked wallet",
        "tax form stamped denied", "empty grocery cart receipt"

emotional_register: use the register provided in the scene context below.
  All sentences in this chapter share the same register.
  Valid values include strategic_hook, mechanism_flow, incongruity_drop,
  dependency_link, human_anchor, causal_collapse, institutional_facade,
  shadow_truth, synthesis.

pre_silence_ms:
  - Set to {PATTERN_INTERRUPT_PRE_SILENCE_MS} for the FIRST sentence only of any PATTERN INTERRUPT scene.
  - Set to 0 for all other sentences everywhere.
  This inserts real silence before the first spoken word of a pattern interrupt,
  creating the hard tonal break. Do not apply to any other sentence.

visualization_type: classify each sentence for animated visualization.
  Use "none" for most sentences (stock footage will be used).
  Use a viz type ONLY when the sentence contains a concrete number, statistic,
  comparison, monetary flow, historical date sequence, or depletion sequence.

  counting_number — sentence states a specific dollar amount, percentage, or count
    that would be more powerful animated counting up.
    Examples: "Americans spend $1.2 trillion on housing every year."
              "The top 1% owns 38% of all wealth."
              "The average worker gets a 2.3% raise — inflation runs at 4.1%."

  comparison — sentence directly compares two quantities, rates, or groups side by side.
    Examples: "CEOs earn 350 times what their median worker makes."
              "A white applicant is 50% more likely to get a callback than an equally qualified Black applicant."

  flow — sentence describes money, power, or data moving between entities in a loop or chain.
    Examples: "Your rent payment goes to a landlord, who sends it to a REIT, which pays dividends to shareholders."
              "Tax dollars flow from workers to the federal budget to defense contractors."

  timeline — sentence references multiple dates or a historical sequence of events.
    Examples: "In 1971 Nixon ended the gold standard. By 1980 inflation had peaked at 13%."
              "The minimum wage was last raised in 2009 — sixteen years ago."

  shrink — sentence describes something being depleted chunk by chunk (taxes, fees, costs).
    Examples: "After federal tax, state tax, FICA, and health insurance, a $60,000 salary becomes $41,000."
              "The $500 stimulus check covered rent for 2 days, groceries for 4, and left $12."

  Rules:
  - At most 25% of sentences in a chapter should have a viz type — be selective.
  - When in doubt, use "none". Bad animations are worse than stock footage.
  - Never assign a viz type to transitional, emotional, or opinion sentences.
  - Only assign to sentences where the NUMBER is the whole point of the sentence.

Output only the JSON array. No prose, no markdown fences.\
"""


def build(chapter_prose: str, chapter: ChapterOutline) -> tuple[str, str]:
    register = _SCENE_REGISTER.get(chapter.title, "systemic_flow")
    is_pattern_interrupt = chapter.index in PATTERN_INTERRUPT_INDICES

    user = f"""\
Chapter index: {chapter.index} (use this for the "ch" part of IDs)
Chapter energy: {chapter.energy}
Chapter title: {chapter.title}
Emotional register for all sentences in this chapter: {register}
Is pattern interrupt scene: {is_pattern_interrupt}
  (if True, set pre_silence_ms={PATTERN_INTERRUPT_PRE_SILENCE_MS} on the FIRST sentence only, 0 on all others)

Prose to decompose:
{chapter_prose}\
"""
    return _SYSTEM, user
