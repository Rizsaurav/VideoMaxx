# MEMORY.md

## 2026-05-03 — Architecture decisions locked
What was decided: Per-sentence atomic TTS units; Chatterbox on MPS with float32 fix; WhisperX large-v3 at float32; h264_videotoolbox encoder; diskcache for LLM + CLIP; sequential stages with explicit model unloads; parallel chapter renders via ProcessPoolExecutor.
Why: Maximize quality-per-compute on M2 16GB, enable per-sentence re-rolls, stay under memory ceiling.
What was rejected: Paragraph-level TTS, libx264, float16 on MPS, concurrent stage model loading.

## 2026-05-03 — Project structure finalized
What was decided: SRP-driven layout — models/, pipeline/stages/, services/ (llm/tts/vision/alignment/assets), render/, state/, utils/. One file per concern. CLI via typer. Config via pydantic-settings.
Why: Scalability, testability, and clean phase-by-phase implementation without cross-cutting concerns.
What was rejected: Monolithic pipeline file, God-class orchestrators.
