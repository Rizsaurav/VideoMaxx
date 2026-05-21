# VidMaxx — Agent Reference

Local Python pipeline: topic → fully-edited YouTube video on M2 Mac.
Owner: Saurav (junior CS student, TXST). Calibrate responses at intermediate-advanced level.

---

## Environment

- Python 3.11 in `.venv/` — `source .venv/bin/activate`
- Install: `pip install numpy && pip install -r requirements.txt && pip install -e .`
- Run: `vidmaxx new "topic"` / `vidmaxx run <slug>` / `vidmaxx reset <slug> --to <stage>`
- Working dir must be `/Users/saurav/Desktop/VidMaxx` — `.env` and `style_pack/` resolve relative to CWD
- Vertex AI auth: `gcloud auth application-default login && gcloud config set project PROJECT_ID`

---

## Stack

| Layer | Tech |
|-------|------|
| LLM (script) | litellm → Vertex AI (gcloud ADC, no API key needed) |
| Research | google-genai SDK → Gemini 2.5 Pro + Google Search grounding (real page fetching) |
| Verification | google-genai SDK → Gemini 2.5 Flash + Google Search grounding (parallel, per claim) |
| TTS | Kokoro via mlx-audio (Apple Silicon native, MPS) |
| Alignment | WhisperX float32 on MPS (float16 crashes on MPS) |
| Vision/ranking | open-clip-torch ViT-B-32 (CLIP similarity ranking) |
| Asset sources | Pexels, Pixabay, Wikimedia |
| Render | FFmpeg h264_videotoolbox (hardware encoder, 5-10x faster than libx264) |
| State | SQLite WAL + JSON state snapshots |
| Cache | diskcache (LLM responses, CLIP embeddings — keyed by sha256) |
| CLI | typer + rich + structlog |

---

## Pipeline Stages (sequential, 17 stages)

```
CREATED
  → RESEARCH      s01a: Gemini 2.5 Pro + Google grounding → auto_brief.json
  → VERIFY        s01b: Gemini 2.5 Flash + Google grounding, parallel per claim → verified_fact_sheet.json
  → FREEZE        s01c: SHA-256 hash, preferred-gate warnings, fallback dinner_fact
  → ARCHITECT     s02a: hash-verified factsheet → 9-scene outline + chapter prose
  → CRITIC        s02b: deterministic + LLM sentence-rule enforcement (HALT on CRITICAL)
  → OPTIMIZER     s02c: rewrites WARNING-flagged sentences only
  → VALIDATE      s02d: decomposer → script.json, dinner_fact key term anchor check
  → ASSETS        s03:  per-sentence Pexels/Pixabay/Wikimedia + CLIP rank
  → TTS           s04:  Kokoro per sentence → audio/{id}.wav → full_narration.wav
  → ALIGNMENT     s05:  WhisperX word-level alignment → alignment.json
  → TIMELINE      s06:  compile timeline.json (music_duck via emotional_register)
  → RENDER        s07:  FFmpeg chapter renders (parallel) → concat → out/video.mp4
  → SHORTS_SELECT s08:  score short candidates (Beat 1 boundary from strategic_hook register)
  → SHORTS_RENDER       (user triggers via `vidmaxx shorts --pick`)
  → PUBLISH       s09:  YouTube upload (optional)
  → DONE
```

The orchestrator (`pipeline/orchestrator.py`) loops through `STAGE_ORDER`, skipping already-completed stages. It resumes from the current stage on retry — `fail()` keeps the stage value unchanged.

**Terminal stages:** `SHORTS_RENDER` (waits for user `vidmaxx shorts --pick`), `DONE`.

**No-halt scripting behavior:** research, verification, freeze, critic, optimizer, and validate are designed to keep the scripting run moving. Guardrails write warnings, repair metadata, critic flags, and source audit trails instead of stopping unless a file is missing or an API call fails unrecoverably.

---

## Anti-Hallucination Architecture

This is the core design principle. Every factual claim in the final video must be traceable to a grounding-verified, sourced claim in the VerifiedFactSheet.

```
Research (s01a) — Gemini 2.5 Pro + Google Search grounding
  ↓ Pass 1 finder fetches real pages and writes discovery.json.
  ↓ Pass 2 structures a brief using only finder output, no web access.
  ↓ Every claim source is checked against finder/grounding evidence.
  ↓ Unverified source URLs are repaired to the closest parsed finding source.
  ↓ auto_brief.json written with per-claim source arrays.

Verify (s01b) — Gemini 2.5 Flash + Google Search grounding, one call per claim, parallel
  ↓ Adversarial: searches specifically for contradicting evidence, updated figures,
  ↓ misleading context. CONFIRMED | PLAUSIBLE | CONTESTED | UNVERIFIED per claim.
  ↓ cold_fact may be replaced with the best verified fallback if contested.
  ↓ dinner_fact is replaced with a CONFIRMED fallback when possible.
  ↓ CONTESTED claims retained as low-trust; UNVERIFIED claims excluded unless all claims fail.
  ↓ full auto_brief.json embedded in fact_sheet.brief for Architect access.

Freeze (s01c) — preferred gates logged as warnings, SHA-256 hash
  ↓ Hash written to SQLite. frozen=True set in JSON.

Architect (s02a) — hash verified against SQLite before proceeding
  ↓ fact_sheet.brief supplies pre-researched scene anchors directly.
  ↓ Only frozen, grounding-verified facts become beats.

Critic (s02b) — UNGROUNDED_CLAIM (CRITICAL): narration not traceable to VerifiedClaim
  ↓ CRITICAL = audit flag. Pipeline continues.
  ↓ Optimizer removes unsupported claims or grounds them in verified facts.

Optimizer (s02c) — receives WARNING and CRITICAL flags
```

**The guarantee:** unsupported facts are not silently trusted. They are labeled in critic output and sent to Optimizer, which must remove them or rewrite them using verified claims. Grounding ensures source URLs were actually fetched or repaired from discovery output, not blindly trusted from model memory.

---

## Master Beat Structure (locked)

The video structure is fixed from `sample/master_beat_analysis.json`. The Architect maps verified facts onto these beats; it does not choose the structure.

| Index | Title | Duration | Energy | Notes |
|-------|-------|----------|--------|-------|
| 0 | The Strategic Hook | 105s | high | Power multiplier, impact, central puzzle. |
| 1 | The Mechanism Flow | 105s | medium | Technical supply chain as problem → solution → new problem. |
| 2 | The Incongruity Drop | 105s | high | The slightly wrong detail; promise vs. reality. |
| 3 | The Dependency Link | 105s | medium | Resource/vulnerability pairing. |
| 4 | The Human Anchor | 105s | medium | Analog life inside the system. |
| 5 | The Causal Collapse | 195s | high | Longest payoff: systemic failure chain. |
| 6 | The Institutional Facade | 120s | medium | Rhetoric vs. reality. |
| 7 | The Shadow-Truth | 105s | high | Darker pivot and social-currency reveal. |
| 8 | The Synthesis | 75s | medium | Liminal conclusion, no recap padding. |

Total: 1020 seconds (17:00). `SCENE_DURATIONS_SEC` in `constants.py` must sum to 1020.

The old full-chapter pattern interrupt system is disabled for this structure. Tonal ruptures now live in writing beats (`The Incongruity Drop`, `The Shadow-Truth`), not timeline-level music ducking.

---

## Emotional Registers

Every sentence has an `emotional_register` field (set by decomposer, maps to scene title):

| Register | Scene |
|----------|-------|
| `strategic_hook` | The Strategic Hook |
| `mechanism_flow` | The Mechanism Flow |
| `incongruity_drop` | The Incongruity Drop |
| `dependency_link` | The Dependency Link |
| `human_anchor` | The Human Anchor |
| `causal_collapse` | The Causal Collapse |
| `institutional_facade` | The Institutional Facade |
| `shadow_truth` | The Shadow-Truth |
| `synthesis` | The Synthesis |

`VISUAL_REGISTER_QUERIES` in `constants.py`: per-register stock footage hints prepended to `visual_query` before Pexels/Pixabay search. The hint leads the query because Pexels weights the beginning of query strings. Hint is skipped for `systemic_flow` (default register, no emotional specificity needed).

---

## Key Files

### `vidmaxx/config/constants.py`
All hard-coded pipeline constants — change here, treat as a code change.

```python
SCRIPT_ARCHITECT_MODEL    = "vertex_ai/gemini-2.5-pro"
SCRIPT_CHAPTER_MODEL      = "vertex_ai/gemini-2.5-pro"
SCRIPT_DECOMPOSE_MODEL    = "vertex_ai/gemini-2.5-flash"
RESEARCH_MODEL            = "vertex_ai/gemini-2.5-pro"
ARCHITECT_THINKING_BUDGET = 8000          # extended thinking token budget

TARGET_VIDEO_DURATION_SEC = 1020          # 17:00 — sample-derived master beat structure
SCENE_DURATIONS_SEC       = [105, 105, 105, 105, 105, 195, 120, 105, 75]
SCENE_LABELS              = [...]         # 9 locked scene titles
PATTERN_INTERRUPT_INDICES = frozenset()   # tonal ruptures are writing beats now

PATTERN_INTERRUPT_PRE_SILENCE_MS = 1000  # real numpy silence prepended to WAV
PAUSE_MS_PERIOD    = 350                  # injected at timeline compile time, not TTS
PAUSE_MS_COMMA     = 150
PAUSE_MS_QUESTION  = 400
PAUSE_MS_EXCLAMATION = 380
PAUSE_MS_PARAGRAPH_BREAK = 700

VIDEO_ENCODER = "h264_videotoolbox"       # Apple Silicon hardware encoder
SHORTS_WIDTH  = 1080
SHORTS_HEIGHT = 1920
SHORTS_COUNT  = 3
```

### `vidmaxx/config/settings.py`
Pydantic BaseSettings. Reads `.env`. `google_api_key` is ignored for Vertex AI (gcloud ADC).
Path fields resolve relative to CWD: `projects_root`, `style_pack_root`, `cache_dir`.

### `vidmaxx/models/project.py`
`PipelineStage` StrEnum (Python 3.9 backport shim included). All values uppercase.
`STAGE_ORDER` list drives `advance()` and orchestrator skip logic.
`Project.has_failed()` returns `error is not None` — there is no FAILED stage.

`ProjectPaths` properties (all derived from `root/`):
- `verified_fact_sheet` → `verified_fact_sheet.json`
- `outline` → `outline.json`
- `chapter_prose_dir` → `chapter_prose/`
- `chapter_prose_final_dir` → `chapter_prose_final/`
- `critic_report` → `critic_report.json`
- `script` → `script.json`
- `brief_json` → `brief.json`
- `run_log` → `run_log.json`

### `vidmaxx/models/factsheet.py`
```python
class VerifiedClaim(BaseModel):
    id: str                    # "claim_01", "claim_02", ...
    claim: str                 # one factual sentence
    status: Literal["CONFIRMED", "PLAUSIBLE"]
    source_url: str
    source_type: Literal["primary", "secondary"]
    number_present: bool       # contains a specific number/date/amount?
    contradiction_checked: bool

class VerifiedFactSheet(BaseModel):
    topic: str
    file_hash: str = ""        # SHA-256 of content JSON; set by s01c
    frozen: bool = False       # set True by s01c after hash is written
    claims: list[VerifiedClaim]
    key_entities: list[str]
    key_dates: list[str]
    raw_summary: str
    dinner_fact_id: str        # claim id chosen by s01b for the cold open

class CriticFlag(BaseModel):
    sentence_id: str           # "ch03_s01" or "" for chapter-level
    severity: Literal["CRITICAL", "WARNING"]
    reason: str
    rule_violated: str         # "MAX_18_WORDS" | "NO_RHETORICAL_Q" | "FOUR_PURPOSE" | ...
    ungrounded_claim: str | None

class CriticReport(BaseModel):
    has_critical: bool
    flags: list[CriticFlag]
    # .critical_flags and .warning_flags properties

class ScriptDiff(BaseModel):
    dinner_fact_anchored: bool
    total_sentences: int
    modified_sentence_ids: list[str]
    unflagged_modifications: list[str]
    critical_flags_resolved: int
```

### `vidmaxx/models/sentence.py`
```python
class Sentence(BaseModel):
    id: str                       # "ch03_s12"
    text: str
    pace: Literal["slow","medium","fast"]
    emphasis_words: list[str]
    pause_before_ms: int
    pause_after_ms: int
    exaggeration: float           # written by decomposer; ignored at TTS inference
    cfg_weight: float             # written by decomposer; ignored at TTS inference
    expected_duration_sec: float
    visual_query: str
    emotional_register: str = "mechanism_flow"  # drives CLIP hint
    pre_silence_ms: int = 0       # normally 0 in the master-beat structure
```

Pause timings are **not** applied by TTS. They are inserted as numpy silence arrays at the WAV level in s04, or as timeline gaps in s06.

### `vidmaxx/models/timeline.py`
```python
class ChapterSlice(BaseModel):
    ...
    music_duck: bool = False   # True for pattern interrupt chapters (volume=0 in filtergraph)
```

`music_duck` is still supported by s06 for legacy pattern-interrupt registers, but the current master-beat structure keeps `PATTERN_INTERRUPT_INDICES` empty and uses rhetorical ruptures instead of full-chapter music drops.

### `vidmaxx/state/project_state.py`
SQLite schema has two tables: `projects` and `factsheet_hashes`.

```python
state_mgr.set_factsheet_hash(slug, hash_str)  # written by s01c after SHA-256
state_mgr.get_factsheet_hash(slug)            # read by s02a for hash verification
```

`run_stage(slug, stage)` context manager:
- Validates project is at the expected stage before running
- On success: calls `advance(slug)`
- On exception: calls `fail(slug, str(exc))` then re-raises

### `vidmaxx/services/llm/client.py`
Three-layer JSON parsing in `complete_json()`:
1. Direct `json.loads(extracted)`
2. `_repair_json()` — closes unclosed brackets by walking the string
3. One API retry with the same prompt

`complete()` and `complete_json()` are cache-first: LLM calls are keyed by sha256(system + prompt) and stored in diskcache. `complete_json_with_thinking()` is NOT cached — the architect stage caches the parsed Outline by factsheet hash instead.

`_JSON_INSTRUCTION` appended to every `complete_json` prompt:
> "CRITICAL: Return ONLY valid, complete JSON. No markdown, no explanation. If you cannot fit all items, return fewer items but ALWAYS properly close every bracket and brace."

### `vidmaxx/services/beat_doctrine.py`
Loads `sample/master_beat_analysis.json` and compacts it for production agents.

Consumers:
- s02a Architect: receives the full master beat contract and uses it in the outline cache key.
- Chapter Writer: receives the specific beat contract for its chapter index.
- Critic/Optimizer: receive sentence, transition, delete-policy, and anti-pattern rules.
- Validate runtime repair: receives runtime rules so expansion adds evidence/mechanism, not padding.

If `sample/master_beat_analysis.json` is missing, the loader returns `{}` and the pipeline falls back to the baked-in constants/prompts.

---

## Stage Details

### s01a — Research
Two-pass Gemini grounding flow:

Pass 1 finder: Gemini 2.5 Pro with `google_search` grounding enabled. It writes raw findings only in `[URL] | [primary/secondary] | [finding]` format. No narrative brief. s01a parses these lines, extracts grounding metadata, and writes `discovery.json`.

Pass 2 structurer: Gemini 2.5 Flash with JSON mode and thinking disabled. It receives only the finder output and must return: `cold_fact`, `cold_fact_source`, `three_layers`, `three_layer_sources`, `supporting_facts`, `supporting_fact_sources`, `twist`, `dinner_fact`, `dinner_fact_source`.

Source repair: every source URL in the brief is checked against finder/grounding evidence. Vertex grounding metadata may contain `vertexaisearch.cloud.google.com/grounding-api-redirect/...` tracking URLs while the finder text contains the clean source URL. Treat that as grounded when the clean URL appears in parsed finder output and the grounding metadata contains Vertex redirects. If the structurer invents or recalls a URL, s01a replaces it with the closest parsed finder source and records the change in `source_repair_log` / `source_repair_count` inside `auto_brief.json`. This keeps narrative momentum while ensuring VERIFY gets a real source URL for each claim.

If the structurer returns malformed JSON, s01a builds a fallback brief directly from parsed finder findings. If the finder returns fewer than 8 parseable findings, s01a logs a warning and pads missing layer/support slots from available parsed findings. Only zero parseable findings is a research-stage hard stop.

If `brief.json` already exists in the project directory (manual override), it is copied to `auto_brief.json` and the Gemini call is skipped entirely.

Writes `discovery.json` and `auto_brief.json`.

### s01b — Verify
Reads `auto_brief.json` and builds 10 semantic claims: `cl_cold_fact`, `cl_dinner_fact`, `cl_layer_0..2`, `cl_support_0..4`. Each layer/support claim uses its matching source array, never a shared fallback unless a manual brief omitted the arrays.

Each call: Gemini 2.5 Flash + `google_search` grounding. The prompt is adversarial — it searches specifically for contradicting evidence, updated figures, and misleading context. Returns `CONFIRMED | PLAUSIBLE | CONTESTED | UNVERIFIED` + `source_type` (primary/secondary) + `contradiction_detail`.

Anchor repair: if `cl_cold_fact` is contested, s01b replaces `brief.cold_fact` with the best verified number-bearing fallback claim and records `cold_fact_replaced=True`. If `cl_dinner_fact` is not confirmed, s01b replaces it with the best confirmed/available fallback and sets `dinner_fact_id` to that claim. `CONTESTED` claims are retained with their low-trust label so downstream agents know not to present them as settled facts. `UNVERIFIED` claims are excluded unless every claim fails, in which case one contested fallback keeps the run moving.

Embeds the repaired brief content into `VerifiedFactSheet.brief` so the Architect receives narrative anchors without loading a separate file.

Writes `verified_fact_sheet.json` with `frozen=False`.

### s01c — Freeze
Preferred gates (warnings only; the run continues):
- `len(claims) >= 8`
- At least 1 CONFIRMED claim
- `dinner_fact_id` should point to the strongest available claim, preferably CONFIRMED

If preferred gates are missed, s01c records `freeze_guardrail_warnings` and `freeze_risk_level="elevated"` in `VerifiedFactSheet.brief`. Missing dinner fact IDs are repaired to the strongest available fallback claim.

SHA-256 computed over canonical JSON (`frozen=False, file_hash=""`) so the hash covers content only. Written to SQLite via `set_factsheet_hash()`, then `frozen=True` set in the JSON file.

### s02a — Architect
Reads `verified_fact_sheet.json`, verifies:
1. `fact_sheet.file_hash == state_mgr.get_factsheet_hash(slug)` — tamper check
2. `fact_sheet.frozen == True` — freeze gate passed

`architect.build()` in the prompt module also enforces `isinstance(fact_sheet, VerifiedFactSheet)` and `frozen=True` — defense in depth.

Architect LLM call uses extended thinking (`thinking_budget=8000`, `max_tokens=16384`). Output cached by `factsheet_hash + brief_hash`. Chapter writers run in parallel via `asyncio.gather`. Chapter prose saved as `chapter_prose/{idx:02d}.txt`.

Optional `brief.json` (loaded from `paths.brief_json`): anchors specific scenes with pre-researched content. Fields: `topic`, `cold_fact`, `false_belief`, `three_layers` (list of 3), `twist`, `dinner_fact`. Brief hash included in architect cache key.

### s02b — Critic
Two-pass evaluation per chapter:

**Pass 1 — Deterministic (no LLM):**
- `MAX_18_WORDS` → CRITICAL — `len(text.split()) > 18`
- `NO_RHETORICAL_Q` → CRITICAL — `text.strip().endswith("?")`
- `NO_FILLER_TRANSITION` → WARNING — regex for "as we discussed", "moving on", etc.
- `FOUR_PURPOSE` pre-flag → WARNING — sentence has none of: digit, two-word proper noun, ellipsis, contradiction word (but, except, however, actually, despite, contrary, wrong, not)

**Pass 2 — LLM (per chapter, batched):**
- `FOUR_PURPOSE` adjudication — LLM confirms or clears pre-flagged sentences
- `UNGROUNDED_CLAIM` → CRITICAL — factual assertion not traceable to VerifiedClaims

**CRITICAL continuation:** if `has_critical=True`, `CriticReport` is written to disk (full flag list available), then the pipeline advances to Optimizer with:
- All flagged sentence IDs
- Verbatim ungrounded claim text for each `UNGROUNDED_CLAIM` flag
- Structural violations (rule name + sentence ID)

This keeps the run moving while preserving the audit trail. Optimizer must remove or ground these sentences.

### s02c — Optimizer
Receives WARNING and CRITICAL flags. CRITICAL flags are repaired by removing unsupported factual assertions or replacing them with verified claims supplied in the optimizer prompt.

For chapters with flags, annotates prose inline with `[FIX: SEVERITY:RULE_NAME]` markers, then asks LLM to rewrite only marked sentences. Unflagged sentences are preserved verbatim. Chapters with no flags are copied unchanged to `chapter_prose_final/`.

### s02d — Validate
Runs decomposer on each chapter's final prose (same split-at-1000-chars logic). Sentence IDs are renumbered sequentially after any split merges.

**Dinner fact anchor check:** extracts numbers (regex `\b\d[\d,\.%$]*\b`) and named entities (regex `[A-Z][a-z]+ [A-Z][a-z]+`) from the `dinner_fact_id` claim. All extracted key terms should appear in Chapter 0 text. Missing terms are logged as warnings.

Writes `script.json`. VALIDATE → ASSETS.

### s03 — Assets
Visual query generation follows the Incongruity Principle in the decomposer: an everyday object, place, or document with one slightly wrong or alarming detail. Avoid generic graphs, sad people, and abstract metaphors. Examples: `paycheck with negative net pay`, `piggy bank leaking coins`, `tax form stamped denied`.

Visual query augmentation: register hint prepended to `sentence.visual_query` before search. Hint is chosen randomly from `VISUAL_REGISTER_QUERIES[register]`. The hint leads (not trails) because Pexels weights the start of the query string. No augmentation for `systemic_flow` (default register).

`CLIPRanker` loaded once for the whole stage, unloaded on exit (memory management for 16GB unified memory).

### s04 — TTS
Kokoro TTS via mlx-audio. Model stays warm across all sentences, freed after stage.

`pre_silence_ms > 0`: the sentence is passed to TTS with `pause_before_ms=0` (to avoid double-counting — TTS bakes `pause_before_ms` as numpy silence). Then `_prepend_silence(wav_path, silence_ms)` inserts the real silence separately using numpy + soundfile.

This prevents the double-counting bug: TTS would bake 350ms (pause_before_ms default) + stage would add 1000ms = 1350ms total instead of 1000ms.

### s05 — Alignment
WhisperX at `float32` on MPS. `float16` crashes on MPS — do not change this.

### s06 — Timeline
Pattern interrupt detection still exists for legacy scripts: `any(s.emotional_register in {"pattern_interrupt_1", "pattern_interrupt_2"} for s in chapter.sentences)`. The current master-beat structure does not emit those registers by default.

### s07 — Render
Pre-render validation before ProcessPoolExecutor:
1. First chapter index == 0 (`The Strategic Hook`)
2. Pattern-interrupt chapters are checked only if `PATTERN_INTERRUPT_INDICES` is non-empty
3. Total duration 900–1200s window
4. No single video clip > 4 seconds

Chapter renders parallelized via `ProcessPoolExecutor(max_workers=RENDER_MAX_WORKERS)` then concatenated.

FFmpeg filtergraph (`render/filtergraph.py`):
- `music_duck=False` (normal): existing duck_vol + afade behavior
- `music_duck=True` (pattern interrupt): volume=0, 0.5s fade-back at chapter end

### s08 — Shorts Select
Beat 1 boundary determined from `emotional_register in {"strategic_hook", "clinical_shock"}` sentences in the script, not from `timeline.chapters[0].end_sec`. Finds latest `word.end_sec` among hook-register sentence IDs in alignment data. Fallback to `timeline.chapters[0].end_sec` if register data unavailable.

### s09 — Publish
Reads `verified_fact_sheet.json` (not the old `fact_sheet.json`) for tags and description. Tags built from `key_entities` + topic words.

---

## Orchestrator and Run Log

`run_log.json` at project root. Initialized on first run, appended per stage:
```json
{
  "stage": "CRITIC",
  "started_at": "2026-05-19T10:23:41Z",
  "completed_at": "2026-05-19T10:24:15Z",
  "duration_sec": 34.2,
  "error": null,
  "critical_count": 0,
  "warning_count": 3
}
```

`critical_count` and `warning_count` are only present on CRITIC stage entries. CRITICAL flags are also written to `critic_report.json` and passed to Optimizer.

---

## CLI Commands

```bash
vidmaxx new "topic"                          # create + run full pipeline
vidmaxx new "topic" -b brief.json            # run with pre-researched brief
vidmaxx new "topic" --stop-before tts        # stop before TTS
vidmaxx run <slug>                           # resume from current stage
vidmaxx run <slug> --stop-before render
vidmaxx reset <slug> --to VERIFY            # force stage (uppercase) + clear error
vidmaxx list                                 # all projects + stages
vidmaxx status <slug>                        # detail + file existence checks
vidmaxx shorts <slug> --pick 1,3             # render chosen short candidates
```

Stage names for `--to` / `--stop-before` match `PipelineStage` values: `RESEARCH`, `VERIFY`, `FREEZE`, `ARCHITECT`, `CRITIC`, `OPTIMIZER`, `VALIDATE`, `ASSETS`, `TTS`, `ALIGNMENT`, `TIMELINE`, `RENDER`, `SHORTS_SELECT`, `SHORTS_RENDER`, `PUBLISH`, `DONE`.

---

## brief.json

Optional file placed in the project directory (or passed via `-b`). Anchors specific scenes with pre-researched content. If missing, the Architect produces generic content from research alone.

Required fields:
```json
{
  "topic": "string",
  "cold_fact": "string — opens Scene 0, must be verifiable",
  "false_belief": "string — what viewers currently believe (Scene 1)",
  "three_layers": ["surface layer", "mechanism layer", "consequence layer"],
  "twist": "string — one level deeper than the obvious conclusion (Scene 7)",
  "dinner_fact": "string — the claim viewers will repeat at dinner"
}
```

A template is at `brief_template.json` in the project root.

---

## Style Pack (`style_pack/`)

```
style_pack/
  voice_ref.wav              # required: ~15s, 24kHz mono, clean speech
  voice_ref_transcript.txt   # exact words in voice_ref.wav
  fonts/
  music/                     # background music tracks
  sfx/                       # sound effects
  colors.json
```

To prep voice_ref from a longer source:
```bash
ffmpeg -i source.wav -t 15 -ar 24000 -ac 1 style_pack/voice_ref.wav -y
```

---

## .env Keys

```
GCP_PROJECT_ID=         # required — Vertex AI project for Gemini grounding (s01a, s01b)
PEXELS_API_KEY=         # required
PIXABAY_API_KEY=        # required
GOOGLE_API_KEY=         # required — Gemini LLM calls (architect, chapter writer, critic, optimizer)
YOUTUBE_CLIENT_ID=      # optional — only for publish stage
YOUTUBE_CLIENT_SECRET=  # optional
YOUTUBE_REFRESH_TOKEN=  # optional
```

Auth for Vertex AI (grounding): `gcloud auth application-default login` — no API key needed, ADC handles it. `GCP_PROJECT_ID` must have the Vertex AI API enabled.

---

## Cache

`diskcache` at `.cache/diskcache/`. Key namespaces:
- `llm:{sha256(system+prompt)}` — all `complete()` and `complete_json()` calls
- `clip:{sha256(url)}` — CLIP embeddings by asset URL
- Architect outline: cached separately by `factsheet_hash:brief_hash` (not via LLMClient)

Note: Gemini grounding calls (s01a, s01b) are NOT cached — grounding fetches live pages and results must be fresh per run. All other LLM calls go through LLMClient and are disk-cached.

Clear with `rm -rf .cache/diskcache`. Required after any bad LLM response is cached. After clearing, the next run re-calls all LLMs from scratch.

---

## Sentence ID Format

`ch{chapter_index:02d}_s{sentence_index:02d}` — zero-padded, 0-based.
After decomposer splits (prose > 1000 chars), both halves are merged and renumbered sequentially so IDs are always continuous within a chapter.

---

## Memory Constraints (M2, 16GB unified)

- Stages load models sequentially, unload before next stage
- CLIPRanker: loaded at start of s03, `unload()` called in finally block
- WhisperX: loaded in s05, freed after
- Kokoro TTS: warm across all sentences in s04, freed after stage
- Chapter renders: `ProcessPoolExecutor` with `RENDER_MAX_WORKERS=4` (M2 perf + efficiency cores)
- Never load two heavy models simultaneously

---

## Known Bugs Fixed

| Issue | Fix |
|-------|-----|
| StrEnum not in Python 3.9 | Backport shim in `models/project.py` |
| `X \| Y` union syntax on 3.9 | `from __future__ import annotations` |
| Gemini JSON wrapped in markdown fences | `_extract_json()` finds outermost `{`/`[` boundaries |
| Gemini output truncated mid-JSON | `_repair_json()` closes brackets; prose split at 1000 chars; JSON instruction appended to every prompt |
| Pattern interrupt silence double-counted | `pre_silence_ms > 0` → zero `pause_before_ms` before TTS, then prepend separately (1350ms → 1000ms) |
| music_duck used hardcoded chapter indices | Now reads `emotional_register` from sentences — semantic, not positional |
| Scene 1 boundary in shorts used timeline timestamp | Now uses `strategic_hook`/legacy `clinical_shock` register sentences + alignment word timestamps |
| s09_publish read deleted fact_sheet.json | Updated to read `verified_fact_sheet.json` |
| CRITICAL flags killed scripting momentum | s02b now records criticals and advances; s02c removes/grounds flagged sentences |
| Dinner fact anchor used 60% word overlap | Replaced with key term extraction (numbers + named entities must be present verbatim) |
| getattr fallback allowed non-VerifiedFactSheet into Architect | `isinstance` + `frozen` check; hard TypeError on wrong type |
| Stale s01_research.py + s02_script.py in stages dir | Deleted — no longer imported or reachable |
