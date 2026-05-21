"""
Hard-coded pipeline constants. Not overridable via env — if you need to change
these, change them here and treat it as a code change.
"""

# ---------------------------------------------------------------------------
# Pause timings (milliseconds) — injected at timeline compile time, not TTS
# ---------------------------------------------------------------------------

PAUSE_MS_PERIOD = 350
PAUSE_MS_COMMA = 150
PAUSE_MS_QUESTION = 400
PAUSE_MS_EXCLAMATION = 380
PAUSE_MS_PARAGRAPH_BREAK = 700

# ---------------------------------------------------------------------------
# Pattern interrupt silence (milliseconds) — real silence prepended to WAV
# ---------------------------------------------------------------------------

PATTERN_INTERRUPT_PRE_SILENCE_MS = 1000

# ---------------------------------------------------------------------------
# Asset acquisition
# ---------------------------------------------------------------------------

CLIP_SIMILARITY_THRESHOLD = 0.22       # minimum score to accept a candidate
CLIP_MODEL_NAME = "ViT-B-32"           # smaller/faster, sufficient for ranking
CLIP_PRETRAINED = "openai"
ASSET_CANDIDATES_PER_SOURCE = 3        # fetch top N from each source per query
ASSET_FETCH_CONCURRENCY = 8            # parallel httpx requests during Stage 3

# ---------------------------------------------------------------------------
# TTS sentence schema values (kept for decomposer prompt compatibility)
# These are written into script.json by the LLM but ignored at inference time.
# ---------------------------------------------------------------------------

TTS_DEFAULT_EXAGGERATION = 0.5
TTS_DEFAULT_CFG_WEIGHT = 0.5
TTS_DRAMATIC_EXAGGERATION = 0.7
TTS_FAST_CFG_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# WhisperX
# ---------------------------------------------------------------------------

WHISPERX_MODEL = "large-v3"
WHISPERX_DEVICE = "cpu"
WHISPERX_COMPUTE_TYPE = "float32"

# ---------------------------------------------------------------------------
# FFmpeg / render
# ---------------------------------------------------------------------------

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
VIDEO_BITRATE = "5M"
AUDIO_BITRATE = "192k"
VIDEO_ENCODER = "libx264"              # predictable file size; VideoToolbox ignores -b:v on motion-heavy content
AUDIO_ENCODER = "aac"

SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_MIN_DURATION_SEC = 30
SHORTS_MAX_DURATION_SEC = 60
SHORTS_COUNT = 3

# Sequential renders on 8GB — VideoToolbox is fast enough without parallelism
RENDER_MAX_WORKERS = 1

# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------

MUSIC_DEFAULT_VOLUME = 0.15
MUSIC_DUCK_VOLUME = 0.05               # volume when narration is present
MUSIC_FADE_DURATION_SEC = 1.5

# ---------------------------------------------------------------------------
# Motion effects
# ---------------------------------------------------------------------------

KEN_BURNS_SCALE_START = 1.0
KEN_BURNS_SCALE_END = 1.08             # 8% zoom over the clip duration

# ---------------------------------------------------------------------------
# Script / LLM
# ---------------------------------------------------------------------------

TARGET_VIDEO_DURATION_SEC = 1020       # 17:00 — sample-derived nine-beat structure sums to this

# Google GenAI SDK / Vertex AI direct calls.
# Used by s01a/s01b grounding flow, not LiteLLM.
GEMINI_GROUNDING_MODEL = "gemini-2.5-pro"
GEMINI_STRUCTURE_MODEL = "gemini-2.5-flash"
GEMINI_VLM_MODEL = "gemini-2.5-flash"          # vision stage — asset study + scene reassignment
GEMINI_GCP_LOCATION = "us-central1"

# Max concurrent Gemini vision calls — each sends an image frame, so keep low
VLM_CONCURRENCY = 4

# LiteLLM / Vertex AI calls used by the script pipeline.
SCRIPT_ARCHITECT_MODEL = "vertex_ai/gemini-2.5-pro"
SCRIPT_CHAPTER_MODEL = "vertex_ai/gemini-2.5-pro"
SCRIPT_DECOMPOSE_MODEL = "vertex_ai/gemini-2.5-flash"
RESEARCH_MODEL = "vertex_ai/gemini-2.5-pro"

# Approximate token budget for extended thinking in the architect pass
ARCHITECT_THINKING_BUDGET = 8000

# diskcache key prefix
CACHE_LLM_PREFIX = "llm"
CACHE_CLIP_PREFIX = "clip"
CACHE_VOICE_EMBED_PREFIX = "voice"

# ---------------------------------------------------------------------------
# Nine-scene structure — titles and per-scene metadata
# ---------------------------------------------------------------------------

# Exact scene titles the architect outputs (index = chapter index, 0-based)
SCENE_LABELS: list[str] = [
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

# 0-based chapter indices that are hard audio pattern interrupts.
# The sample-derived structure uses tonal/rhetorical ruptures instead of
# full-chapter silence/music-duck interrupts, so this is intentionally empty.
PATTERN_INTERRUPT_INDICES: frozenset[int] = frozenset()

# Fixed target durations per scene (seconds) — must sum to TARGET_VIDEO_DURATION_SEC
SCENE_DURATIONS_SEC: list[int] = [105, 105, 105, 105, 105, 195, 120, 105, 75]

# ---------------------------------------------------------------------------
# Visual emotional registers — eight categories mapped to stock query hints
# ---------------------------------------------------------------------------
# Used by decomposer to label each sentence; s03 appends a hint to visual_query
# so CLIP ranks within the correct emotional category.

VISUAL_REGISTER_QUERIES: dict[str, list[str]] = {
    # Beat 1: Strategic Hook — scale, power, immediate impact
    "strategic_hook": [
        "empty government corridor surveillance",
        "classified documents desk lamp",
        "server room blinking lights dark",
        "lone figure empty courtroom",
        "security camera close up night",
    ],
    # Beat 2: Mechanism Flow — process, technical supply chain
    "mechanism_flow": [
        "factory conveyor belt machinery",
        "control room screens operators",
        "shipping containers port cranes",
        "data center cables server racks",
        "industrial pipe network close up",
    ],
    # Beat 3: Incongruity Drop — ordinary thing with wrong detail
    "incongruity_drop": [
        "paycheck negative net pay",
        "piggy bank leaking coins",
        "office cubicle locked wallet",
        "tax form stamped denied",
        "empty grocery cart receipt",
    ],
    # Beat 4: Dependency Link — resources and vulnerability
    "dependency_link": [
        "supply chain warehouse conveyor",
        "power lines over industrial plant",
        "cargo ship containers aerial",
        "hands connecting network cables",
        "water pipe valve close up",
    ],
    # Beat 5: Human Anchor — analog life inside system
    "human_anchor": [
        "family dinner table smiling",
        "office worker smiling desk morning",
        "busy street market daytime",
        "students studying library sunlight",
        "commuters walking city sidewalk",
    ],
    # Beat 6: Causal Collapse — failure chain
    "causal_collapse": [
        "money transfer abstract network nodes",
        "aerial city highway traffic flow",
        "corporate boardroom empty chairs",
        "data visualization abstract blue",
        "supply chain warehouse conveyor",
    ],
    # Beat 7: Institutional Facade — official story versus behavior
    "institutional_facade": [
        "corporate lobby glass building",
        "press conference podium microphones",
        "government hearing room empty chairs",
        "executive boardroom polished table",
        "official document signature close up",
    ],
    # Beat 8: Shadow-Truth — darker pivot
    "shadow_truth": [
        "single door hallway dim light",
        "close up hand stopping clock",
        "empty parking garage echo",
        "screen going dark reflection",
        "one chair spotlight theater",
    ],
    # Beat 9: Synthesis — liminal conclusion
    "synthesis": [
        "chess pieces close up dramatic",
        "mirror reflection distorted hallway",
        "hands shuffling playing cards slow",
        "person slowly looking up realization",
        "two doors hallway choice perspective",
    ],
    # Legacy/fallback registers kept for old cached scripts.
    "clinical_shock": [
        "empty government corridor surveillance",
        "classified documents desk lamp",
        "server room blinking lights dark",
    ],
    "false_normal": [
        "family dinner table smiling",
        "office worker smiling desk morning",
        "busy street market daytime",
    ],
    "infrastructure": [
        "bank building exterior glass facade",
        "wall street trading floor screens",
        "server room cooling system cables",
    ],
    "systemic_flow": [
        "money transfer abstract network nodes",
        "aerial city highway traffic flow",
        "corporate boardroom empty chairs",
    ],
    "personal_consequence": [
        "person checking phone alone night",
        "bills stacked kitchen table close up",
        "hands holding wallet empty",
    ],
    "twist": [
        "chess pieces close up dramatic",
        "mirror reflection distorted hallway",
        "hands shuffling playing cards slow",
    ],
    "exit": [
        "city skyline dusk silhouette",
        "person walking away camera road",
        "closing door hallway receding",
        "road stretching empty horizon sunset",
        "window rain night city lights",
    ],
}

# CLIP ranking descriptions — richer visual language than keyword queries.
# Used as the ranking target so CLIP optimizes for the scene's emotional tone,
# not the compound search string.
VISUAL_REGISTER_DESCRIPTIONS: dict[str, str] = {
    "strategic_hook":      "ominous empty institutional space surveillance clinical cold stark",
    "mechanism_flow":      "industrial machinery process technical overhead systematic flow",
    "incongruity_drop":    "ordinary object wrong disturbing detail mundane contrast close up",
    "dependency_link":     "infrastructure network supply chain scale aerial industrial",
    "human_anchor":        "real person everyday life warmth analog natural daylight candid",
    "causal_collapse":     "abstract network failure cascade ripple dark visualization",
    "institutional_facade":"polished institutional official corporate government formal building",
    "shadow_truth":        "dim revelation hidden beneath surface single object silence dark",
    "synthesis":           "liminal threshold choice perspective wide contemplative",
    "clinical_shock":      "ominous empty institutional space surveillance clinical cold stark",
    "false_normal":        "cheerful family everyday life ordinary morning routine",
    "infrastructure":      "financial institution glass building trading floor screens",
    "systemic_flow":       "abstract system process flow nodes network aerial neutral",
    "personal_consequence":"person alone night bills wallet hands close up anxiety",
    "twist":               "chess game reflection distortion playing cards slow close up",
    "exit":                "city skyline dusk silhouette road horizon receding departure",
    "pattern_interrupt_1": "stark single object isolated silence clinical white space",
    "pattern_interrupt_2": "single dark object silence revelation contrast exposure",
}
