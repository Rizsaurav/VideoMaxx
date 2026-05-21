"""
Builds the (system, user) prompt pair for the chapter writer pass (Stage 3b).

Input:  ChapterOutline + full Outline (context) + FactSheet
Output: prose string — the spoken narration for this chapter

One async call per chapter; all chapters are gathered in parallel.
The decomposer (Stage 3c) then extracts sentence-level structure from this prose.
"""

from __future__ import annotations

from vidmaxx.models.factsheet import VerifiedFactSheet
from vidmaxx.models.script import ChapterOutline, Outline

_SYSTEM = """\
You are a YouTube scriptwriter. You write spoken narration — words that will be \
read aloud by a text-to-speech voice.

Style rules (follow all of them):
- Conversational. A person talking to one person, not a lecture.
- Maximum 18 words per sentence. Hard limit. Count them.
- No bullet points, no headers, no markdown — pure flowing prose.
- Vary sentence rhythm. Short punchy sentences after long buildup ones.
- Use second person ("you") to pull the viewer in when the beat is human or consequence-driven.
- No filler transitions like "In this chapter", "As we discussed", "Moving on".
- Facts should feel discovered, not recited.
- End each scene on a line that creates forward momentum or genuine surprise.
- Hit the target duration through more mechanism and consequence, not padding.
- If a scene is long, deepen the causal chain: defense, mechanism, data contrast,
  policy/institutional detail, consequence over time, then reveal.

Sentence rules — every sentence must do exactly ONE of four things:
  → Reveal something new to the viewer
  → Contradict something just said
  → Raise a question the very next sentence answers
  → Drop a specific number or named entity

If a sentence does none of the four things — cut it. No exceptions.

Banned elements (zero tolerance):
  ✗ Rhetorical questions
  ✗ Transition phrases ("moving on", "as we mentioned", "now let's")
  ✗ Filler enthusiasm ("that's actually really fascinating", "interestingly")
  ✗ Sentences over 18 words
  ✗ Any sentence that only summarizes without adding new information
  ✗ Padding to hit length

Ellipsis technique: use ellipses before every major revelation.
  "The company knew... and had known for eleven years."

Burst technique: use 4-6 word bursts before reveals to build tension.
  "Think about that. Every single day. Without your knowledge."

Output only the prose narration. No titles, no stage directions, no metadata.\
"""


def build(
    chapter: ChapterOutline,
    outline: Outline,
    fact_sheet: VerifiedFactSheet,
    beat_doctrine: str = "",
) -> tuple[str, str]:
    beats_text = "\n".join(
        f"  {i + 1}. [{beat.energy} energy] {beat.description}"
        for i, beat in enumerate(chapter.key_beats)
    )

    other_chapters = "\n".join(
        f"  {ch.title} ({ch.energy} energy)"
        for ch in outline.chapters
        if ch.index != chapter.index
    )

    relevant_claims = "\n".join(
        f"  - [{c.status}] {c.claim}"
        for c in fact_sheet.claims[:30]
    )

    scene_overrides = {
        "The Strategic Hook": {
            "words": (210, 270),
            "sentences": "about 18-24 short sentences",
            "contract": (
                "Frame the subject through scale, power, or a surprising artifact. "
                "Create the central puzzle and narrator authority quickly."
            ),
        },
        "The Incongruity Drop": {
            "words": (190, 250),
            "sentences": "about 16-22 short sentences",
            "contract": "Reveal the slightly wrong detail. Make the promise/reality mismatch impossible to unsee.",
        },
        "The Shadow-Truth": {
            "words": (190, 250),
            "sentences": "about 16-22 short sentences",
            "contract": "Darker pivot. Say what the earlier evidence implies but did not yet name.",
        },
        "The Synthesis": {
            "words": (135, 190),
            "sentences": "about 10-16 short sentences",
            "contract": "Resolve through meaning, not recap. Leave one clean final image.",
        },
    }

    if chapter.title in scene_overrides:
        min_words, max_words = scene_overrides[chapter.title]["words"]
        target_sentences = scene_overrides[chapter.title]["sentences"]
        depth_contract = scene_overrides[chapter.title]["contract"]
    else:
        # Spoken narration averages roughly 140 wpm. Ask for slightly under target
        # so pauses/silence can fill the rest without bloating prose.
        target_words = max(70, int(chapter.target_duration_sec * 2.1))
        min_words = int(target_words * 0.85)
        max_words = int(target_words * 1.15)
        target_sentences = f"about {max(5, round(target_words / 12))} short sentences"
        depth_contract = {
            "LAYER ONE: SURFACE": (
                "Use this scene to show the visible story and the strongest defense "
                "of the system. Then make that defense feel incomplete."
            ),
            "LAYER TWO: MECHANISM": (
                "Use this scene for causal machinery. Walk through each link slowly: "
                "incentive, permission, policy/institution, measurement, payoff."
            ),
            "LAYER THREE: YOU": (
                "Use this scene for lived consequences. Stretch the impact across "
                "paychecks, debt, housing, family choices, aging, and one cohort."
            ),
            "THE TWIST": (
                "Use this scene to reveal the deeper implication. Do not summarize; "
                "make the viewer re-interpret the whole story."
            ),
            "The Mechanism Flow": (
                "Use this scene as problem-solution machinery. Explain how the system works "
                "while making each answer create the next question."
            ),
            "The Dependency Link": (
                "Use this scene to pair resource with vulnerability. Show what the system needs, "
                "then why that need becomes leverage or fragility."
            ),
            "The Human Anchor": (
                "Use this scene to make the system analog. Move from institution to person, "
                "habit, paycheck, room, commute, family, body, or daily constraint."
            ),
            "The Causal Collapse": (
                "Use this scene as the longest failure-chain map. Walk through incentives, "
                "permissions, constraints, consequences, and the moment the system breaks."
            ),
            "The Institutional Facade": (
                "Use this scene to put official rhetoric against behavior. Let the institution's "
                "own language become evidence."
            ),
        }.get(chapter.title, "Keep it tight. Every sentence must move the narrative.")

    user = f"""\
Video topic: {outline.topic}
Full video outline (for context — do NOT repeat what other scenes cover):
{other_chapters}

Your scene:
  Title: {chapter.title}
  Index: {chapter.index} of {len(outline.chapters) - 1}
  Target duration: ~{int(chapter.target_duration_sec)} seconds (~{chapter.target_duration_sec / 60:.1f} min of speech)
  Target length: {min_words}-{max_words} words, {target_sentences}
  Energy: {chapter.energy}
  Depth contract: {depth_contract}

Beats to hit in order:
{beats_text}

Verified facts available to weave in (use what's relevant, ignore the rest):
{relevant_claims}

{beat_doctrine}

Write the narration for this scene only.
Do not stop early just because all beats are mentioned once.
If you need more length, deepen the mechanism or consequence using verified facts.
Enforce all sentence rules above.\
"""
    return _SYSTEM, user
