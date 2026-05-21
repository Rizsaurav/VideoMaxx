# Sample Beat Analysis Lab

This folder is a contained research lab for studying pacing DNA from reference
YouTube transcripts.

## Input

Put transcript JSON files in:

```bash
sample/json/
```

Each file should contain:

```json
{
  "title": "Video title",
  "creator": "Creator name",
  "transcript_data": [
    {"timestamp": "0:00", "text": "Transcript segment text"}
  ]
}
```

## Main Agent: LLM Deep Writing Analysis

Run:

```bash
python sample/agents/llm_synchronized_beat_analysis.py
```

The LLM agent performs:

- Normalization: maps every source transcript to 0-100%.
- Per-source deep writing analysis.
- Macro structure mapping.
- Meso-level scene, pivot, evidence, and tension analysis.
- Micro-level sentence craft analysis.
- Transition, voice, visual language, and ending analysis.
- Cross-source synthesis into a locked 17-minute, 9-beat writer contract.

It writes individual creator analyses to:

```bash
sample/analysis/
```

Set `VIDMAXX_SAMPLE_FORCE=1` to regenerate existing per-source analyses.
Set `VIDMAXX_SAMPLE_ANALYSIS_MODEL=...` to override the default model.

## Fallback Agent

There is also a deterministic fallback:

```bash
python sample/agents/synchronized_beat_analysis.py
```

That file is only a rough scaffold. The serious analysis path is the LLM agent.

## Output

Writes:

```bash
sample/master_beat_analysis.json
```

The `unified_pacing_map` is the contract for Writer agents. The observed
cross-transcript medians are evidence, not permission to drift from the locked
17-minute structure.
