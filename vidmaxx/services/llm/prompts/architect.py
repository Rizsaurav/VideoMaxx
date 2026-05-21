"""
Builds the (system, user) prompt pair for the architect pass (Stage 3a).

Input:  FactSheet + optional brief dict + target_duration_sec
Output: JSON matching the Outline schema — exactly 9 scenes, fixed structure.

The nine-scene structure is locked. The architect maps research facts onto
each scene's key_beats. Extended thinking is used so the model can reason
through which facts belong where and how to sequence the reveal.
"""

from __future__ import annotations

from vidmaxx.config.constants import SCENE_DURATIONS_SEC, SCENE_LABELS
from vidmaxx.models.factsheet import VerifiedFactSheet

_SYSTEM = """\
You are a YouTube video architect. Your job is to map research facts onto a \
fixed nine-scene narrative structure.

The video has EXACTLY 9 scenes. Scene names, timings, and energies are fixed. \
You populate key_beats for each scene using the provided research. \
Every beat must come from the provided research — do not invent facts.

Scene structure (do not change names, indices, timings, or energies):

  Index 0 | The Strategic Hook       | 105 sec | energy: high
    Frame the topic through a surprising power claim, rule, artifact, or scale.
    Establish the core puzzle and narrator authority with one specific entity,
    date, number, or primary-source detail in the first beat.

  Index 1 | The Mechanism Flow       | 105 sec | energy: medium
    Explain the technical supply chain as problem → solution → new problem.
    Make the system legible without flattening tension.

  Index 2 | The Incongruity Drop     | 105 sec | energy: high
    Reveal the slightly wrong detail: the promise/reality mismatch, odd artifact,
    contradiction, underwhelming outcome, or impossible-seeming fact.

  Index 3 | The Dependency Link      | 105 sec | energy: medium
    Show what the system depends on and where that dependency creates
    vulnerability. Pair resource with fragility.

  Index 4 | The Human Anchor         | 105 sec | energy: medium
    Translate the system into analog life. Who feels it? What does it do to a
    person, family, worker, user, city, or viewer?

  Index 5 | The Causal Collapse      | 195 sec | energy: high
    The longest payoff beat. Map the failure chain in granular detail.
    Show how the system breaks through incentives, policy, physics, money,
    institutional behavior, or measurement choices.

  Index 6 | The Institutional Facade | 120 sec | energy: medium
    Put official rhetoric against observable behavior. Use public claims,
    marketing, press releases, policy language, or institutional posture as
    evidence against the system.

  Index 7 | The Shadow-Truth         | 105 sec | energy: high
    The darker pivot. Reveal what the previous beats imply but did not yet say.
    This is the social-currency moment.

  Index 8 | The Synthesis            | 75 sec  | energy: medium
    Resolve through meaning, not recap. Leave the viewer with a larger systemic
    interpretation and one clean final image.

Output a single JSON object. No prose, no markdown fences.

Schema:
{
  "topic": "<string>",
  "target_duration_sec": <number>,
  "chapters": [
    {
      "index": <0-8>,
      "title": "<scene name exactly as listed above>",
      "target_duration_sec": <number — use the fixed timing above>,
      "energy": "<low|medium|high — use the fixed energy above>",
      "key_beats": [
        {
          "description": "<one sentence: the specific fact, mechanism, or entity from the research that goes in this beat>",
          "energy": "<low|medium|high>"
        }
      ]
    }
  ]
}

Rules:
- Output exactly 9 chapters indexed 0-8. Titles exactly as listed above.
- The Strategic Hook: 4-6 beats.
- The Mechanism Flow: 5-7 beats.
- The Incongruity Drop: 4-6 beats.
- The Dependency Link: 5-7 beats.
- The Human Anchor: 5-7 beats.
- The Causal Collapse: 8-12 beats, each beat advancing one causal link.
- The Institutional Facade: 5-7 beats.
- The Shadow-Truth: 4-6 beats.
- The Synthesis: 3-5 beats.
- Scene 0 beat 0: the most powerful verifiable fact, artifact, rule, or scale
  detail from the research, stated as it would sound spoken aloud.
- Scene 7 beat: must be something even informed people don't say out loud —
  one level deeper than the obvious conclusion.
- If a brief is provided, cold_fact anchors scene 0; three_layers distribute
  across scenes 1, 3, and 5; personal consequence anchors scene 4; twist anchors
  scene 7.
- Length must come from mechanism, dependency, human consequence, institutional
  facade, and collapse-chain depth.
  Never add generic motivational filler or repeated summaries.
- Every long-scene beat must be one of:
  power claim, mechanism, incongruity, dependency, human anchor, causal collapse,
  institutional facade, shadow-truth implication, or synthesis.
"""


def build(
    fact_sheet: VerifiedFactSheet,
    target_duration_sec: float,
    brief: dict | None = None,
    beat_doctrine: str = "",
) -> tuple[str, str]:
    if not isinstance(fact_sheet, VerifiedFactSheet):
        raise TypeError(
            f"architect.build() requires a VerifiedFactSheet, got {type(fact_sheet).__name__}. "
            "Run s01a → s01b → s01c before calling the Architect."
        )
    if not fact_sheet.frozen:
        raise ValueError(
            "VerifiedFactSheet.frozen is False — run s01c_freeze before calling the Architect. "
            "An unfrozen factsheet has no hash guarantee."
        )

    # Brief can come from two sources:
    #   1. fact_sheet.brief — auto_brief.json embedded by s01b (Gemini grounding flow)
    #   2. brief argument — legacy manual brief passed directly (still supported)
    # fact_sheet.brief takes precedence; the argument is a fallback.
    effective_brief = fact_sheet.brief or brief or {}

    entities = ", ".join(fact_sheet.key_entities[:8]) or "none identified"
    dates = ", ".join(fact_sheet.key_dates[:6]) or "none identified"
    claims_text = "\n".join(
        f"- [{c.status}] {c.claim} (source: {c.source_url})"
        for c in fact_sheet.claims
    )

    scene_ref = "\n".join(
        f"  {i}. {label} — {SCENE_DURATIONS_SEC[i]}s"
        for i, label in enumerate(SCENE_LABELS)
    )

    # Brief block — always populated when the grounding flow ran (fact_sheet.brief is set).
    # The brief gives the Architect pre-researched narrative anchors so it doesn't need
    # to invent the structure from scratch. Each field maps to specific scenes.
    brief_block = ""
    if effective_brief:
        layers = effective_brief.get("three_layers") or ["", "", ""]
        supporting = effective_brief.get("supporting_facts") or []
        supporting_text = "\n".join(f"    - {fact}" for fact in supporting[:8]) or "    - none"
        brief_block = f"""
Pre-researched narrative brief — use these to anchor the corresponding scenes verbatim:
  Scene 0 (The Strategic Hook) — open with: {effective_brief.get('cold_fact', '')}
  Scene 1 (The Mechanism Flow) — mechanism anchor: {layers[0] if len(layers) > 0 else ''}
  Scene 2 (The Incongruity Drop) — false belief / contradiction to crack: {effective_brief.get('false_belief', '')}
  Scene 3 (The Dependency Link) — dependency anchor: {layers[1] if len(layers) > 1 else ''}
  Scene 4 (The Human Anchor) — human consequence anchor: {layers[2] if len(layers) > 2 else ''}
  Scene 5 (The Causal Collapse) — use supporting facts to map the failure chain.
  Scene 6 (The Institutional Facade) — use source/institution/official rhetoric from verified claims.
  Scene 7 (The Shadow-Truth) — anchor: {effective_brief.get('twist', '')}
  Scene 8 (The Synthesis) — dinner fact the viewer will repeat: {effective_brief.get('dinner_fact', '')}

Supporting facts to distribute across scenes 1, 3, 5, 6, and 7:
{supporting_text}

These are grounding-verified facts. Use them as the primary content for their scenes.
Do not rephrase the cold_fact — Scene 0 beat 0 must use it verbatim or near-verbatim.\
"""

    user = f"""\
Topic: {fact_sheet.topic}
Total target duration: {int(target_duration_sec)} seconds ({target_duration_sec / 60:.1f} minutes)

Fixed scene durations:
{scene_ref}

Key entities: {entities}
Key dates: {dates}

Verified claims (CONFIRMED = independently corroborated; PLAUSIBLE = single source):
{claims_text}
{brief_block}
{beat_doctrine}
Map these facts onto the nine-scene structure. Output all 9 chapters. \
Do not add scenes or change scene names.

To hit the 17-minute runtime without filler, expand the scenes like this:
- Scene 1: make mechanism legible through problem-solution chains.
- Scene 2: reveal the slightly wrong detail that destabilizes the premise.
- Scene 3: show resource/dependency and why it creates vulnerability.
- Scene 4: translate the system into lived human stakes.
- Scene 5: spend the longest runtime mapping causal collapse.
- Scene 6: contrast institutional rhetoric with observable behavior.
- Scene 7: crystallize the darker implication instead of summarizing.
Every beat must create new narrative information.\
"""
    return _SYSTEM, user
