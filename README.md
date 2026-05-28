# VidMaxx

A fully local, end-to-end YouTube video production pipeline running entirely on Apple Silicon. Give it a topic. Get back a rendered, narrated, captioned video with animated data visualizations, source-verified facts, and a YouTube Shorts cut — no human editing required.

---

## What it does

VidMaxx takes a single topic string and runs it through 15 sequential pipeline stages to produce a finished `.mp4`. Every stage is deterministic and resumable: crash mid-run, fix the issue, re-run, and it picks up exactly where it left off.

```
vidmaxx new "the hidden economics of unpaid care work"
vidmaxx run my-project-slug
```

---

## Pipeline architecture

```
Topic
  │
  ▼
S01a  Research         Gemini 2.5 Pro + Google Search grounding (2-pass: Finder → Architect)
S01b  Verify           Tavily cross-verification of all source URLs
S01c  Freeze           SHA-256 fact sheet hash committed to SQLite — tamper detection
  │
  ▼
S02a  Architect        Gemini 2.5 Pro + extended thinking — 9-scene structure → chapter outlines
S02b  Critic           LLM self-review pass — flags factual drift, pacing issues, weak hooks
S02c  Optimizer        Rewrites flagged beats using critic annotations
S02d  Validate         Final schema validation + sentence-level decomposition with viz classification
  │
  ▼
S03   Assets           Multi-source fetch: Wikimedia → Archive.org → Pexels → Pixabay
S03b  VLM Select       CLIP ViT-B/32 semantic ranking — cosine similarity to visual_query per sentence
  │
  ▼
S04   TTS              Kokoro-82M-bf16 via mlx-audio — sequential, faster-than-realtime on M2
S05   Alignment        WhisperX forced alignment — word-level timestamps at float32/MPS
  │
  ▼
S06   Timeline         Per-sentence asset-to-audio mapping → ChapterSlice model
S07   Render           FFmpeg h264_videotoolbox, chapter-parallel via ProcessPoolExecutor
S08   Shorts           Hook extraction + candidate ranking → `vidmaxx shorts --pick`
S09   Publish          YouTube Data API v3 upload (main video + selected shorts)
```

---

## The agent layer

VidMaxx runs five specialized **Manim animation agents** in parallel via `asyncio.Semaphore(4)`, each responsible for a distinct visualization class:

| Agent | Triggers on | What it renders |
|---|---|---|
| `CountingAgent` | Specific dollar amounts, percentages, counts | Animated counter ticking up to the target number |
| `ComparisonAgent` | Side-by-side quantities or groups | Proportional bar chart with labeled deltas |
| `FlowAgent` | Money/labor/power moving through a chain | Animated arrow flow between labeled entities |
| `TimelineAgent` | Historical date sequences or time spans | Horizontal timeline with event markers |
| `ShrinkAgent` | Depletion sequences (taxes, fees, costs) | Stacked bar shrinking segment by segment |

Each agent receives the sentence text and target duration, generates a Manim scene via LLM prompt, executes it in a subprocess, and returns a `.mp4`. If Manim execution fails, the orchestrator silently falls back to stock footage — the pipeline never stalls on an animation failure.

The `VizOrchestrator` routes sentences based on the `visualization_type` field set during decomposition (S02d). The LLM classifier is calibrated to assign viz types to 20–35% of sentences, biased toward stat-heavy chapters.

Separately, high-impact "none" sentences are upgraded to full-screen **quote cards** (Pillow PNG) rather than stock footage — a lighter fallback that still reads as intentional design.

---

## Research integrity

The research stage is deliberately two-pass and constrained:

**Pass 1 — Finder** (Gemini 2.5 Pro + Google Search grounding): fetches real pages, returns structured `[URL | primary/secondary | verbatim stat]` tuples. No narrative, no inference, no hallucination — just raw retrieval.

**Pass 2 — Architect** (Gemini 2.5 Flash, no web access, thinking disabled): converts the finder output into a structured brief. It is explicitly forbidden from introducing any fact not in the Pass 1 output. Every claim must cite a URL from the findings list.

After the brief is built, a source repair pass verifies every URL against grounding metadata. Unverifiable sources are flagged with `grounding_verified: false` and replaced with the best token-matched finding. The `all_claim_sources_verified` field in `auto_brief.json` is the single boolean signal used downstream.

**Freeze** (S01c): the verified fact sheet is SHA-256 hashed and the digest written to SQLite. Every subsequent stage that touches facts re-verifies the hash. A modified fact sheet hard-fails the pipeline — you cannot silently alter a fact after it has been frozen.

---

## TTS

The current backend is **Kokoro-82M-bf16** via `mlx-audio`, running natively on Apple Silicon via MLX. It generates faster than realtime on M2 — a 10-minute narration script synthesizes in roughly 3–4 minutes.

The pipeline architecture treats TTS as atomic per sentence: each sentence is one `asyncio.to_thread` call, sequential, no parallelism. MLX owns the GPU and is not thread-safe; sequential is the only correct execution model.

Pause timing is injected at compile time, not by the TTS model:

| Punctuation | Silence injected |
|---|---|
| `.` period | 350 ms |
| `,` comma | 150 ms |
| `?` question | 400 ms |
| `!` exclamation | 380 ms |
| paragraph break | 700 ms |

This gives consistent pacing regardless of which TTS backend is in use.

---

## Visual rendering

**Ken Burns motion**: every image asset gets a slow zoom (`1.0 → 1.12` scale over clip duration) via FFmpeg `zoompan` filter. Currently at 12% — enough to feel alive without being distracting.

**Captions**: ASS subtitle format, grouped into ~5-word chunks per Dialogue event. Emphasis words render yellow via inline ASS colour override (`\c&H0000FFFF&`). Font: Arial 68pt, 4px outline, semi-transparent black background.

**Encoder**: `h264_videotoolbox` — Apple's hardware H.264 encoder. This is the single biggest render throughput improvement over software encoding. Chapter renders run in parallel via `ProcessPoolExecutor` and are concatenated at the end.

**Short-clip looping**: Pexels and Pixabay commonly return 10–15s clips. The renderer uses `-stream_loop -1` + `trim=duration=D` in the filtergraph so any clip can fill any sentence duration without black frames.

---

## Caching

Every expensive operation is cached:

- **LLM responses**: `diskcache` keyed by `SHA-256(model + system + prompt)`. A full re-run from a warm cache takes seconds.
- **CLIP embeddings**: keyed by image URL. First-run fetches images; subsequent runs are pure dot-product lookups.
- **Pipeline stage outputs**: each stage writes its outputs to disk before advancing state. Resumption reads from disk, not re-runs.
- **SQLite WAL**: project state stored in `projects/registry.db` with WAL mode. Stage transitions are atomic.

---

## Known bottlenecks

These are the real performance ceilings on M2:

| Bottleneck | Where | Why | Current mitigation |
|---|---|---|---|
| WhisperX alignment | S05 | Float16 is broken on MPS; forced to float32 | None — float32 is the fix |
| Manim execution | S03 | Subprocess cold-start per animation; Python import overhead | Semaphore(4) parallelism |
| CLIP image fetching | S03b | Network-bound; each image is fetched before embedding | `diskcache` for repeat runs |
| FFmpeg zoompan | S07 | CPU-bound filter on every image asset | VideoToolbox for encode; zoompan stays CPU |
| TTS (cold start) | S04 | Model load ~15s on first run | Model held warm for entire stage |

---

## Scaling plan

The current architecture is deliberately single-machine. These are the intended scale-out paths, in priority order:

**1. TTS parallelism via batching**
Kokoro supports batch inference. The current sequential loop is a correctness constraint (MLX thread safety), not a fundamental limit. A batch API wrapper would cut TTS time by 4–8x while keeping single-GPU ownership.

**2. Manim agent distributed execution**
Animation agents are already isolated subprocesses. Moving them to a task queue (e.g., Celery + Redis) would let multiple machines render chapters in parallel. The `VizOrchestrator` already returns `dict[sentence_id, Asset | None]` — the API doesn't need to change.

**3. Multi-topic pipeline fan-out**
Each project is a self-contained SQLite row + file tree. Running 10 projects simultaneously requires no shared state changes — just a process pool at the `vidmaxx run` level.

**4. Asset CDN layer**
CLIP-scored assets are currently fetched fresh per project. A shared asset cache keyed by `(query_hash, source)` would eliminate redundant downloads across projects on the same topics.

**5. Cloud research offload**
The Gemini grounding call is the only step that requires internet. Everything else — writing, TTS, rendering, alignment — can run air-gapped. A thin cloud function that runs S01a–S01c and writes to S3 would let the local machine focus entirely on generation.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, async-first (anyio / asyncio) |
| LLM | Gemini 2.5 Pro (architect, chapters, viz), Gemini 2.5 Flash (decomposer, critic) |
| TTS | Kokoro-82M-bf16 via mlx-audio (MLX, Apple Silicon) |
| Alignment | WhisperX, float32, MPS |
| Vision | OpenCLIP ViT-B/32, MPS |
| Animation | Manim Community (5 specialized agents) |
| Video | FFmpeg h264_videotoolbox, ASS subtitles |
| State | SQLite WAL, diskcache, Pydantic v2 |
| CLI | Typer + Rich |
| Logging | structlog |

---

## Setup

```bash
# Requires Python 3.11, uv, ffmpeg, and an M-series Mac
brew install ffmpeg
uv sync

# Required env vars
export GOOGLE_API_KEY=...          # Gemini API
export GOOGLE_APPLICATION_CREDENTIALS=...  # Vertex AI / grounding
export TAVILY_API_KEY=...          # Fact verification
export PEXELS_API_KEY=...
export PIXABAY_API_KEY=...
export YOUTUBE_CLIENT_SECRETS=...  # OAuth2 for publish stage

vidmaxx new "your topic here"
vidmaxx run <slug>
```

---

## Project status

Active development. The end-to-end pipeline is working. Current focus: animation quality (Manim prompts), caption timing, and Ken Burns motion tuning.
