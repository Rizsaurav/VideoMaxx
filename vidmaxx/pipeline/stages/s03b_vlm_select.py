"""
Stage 3b — VLM asset study and scene-level reassignment

Reads:   paths.script        (script.json)
         paths.assets_dir/   (downloaded assets from s03)
Writes:  paths.vlm_manifest_json  (vlm_manifest.json)
Advances: VLM_SELECT → TTS

For every downloaded asset, Gemini Vision studies a representative frame
and picks the sentence within the same scene it best matches.  Assets can
move to any sentence in their scene — not to other scenes — preserving
chapter coherence while improving within-scene narrative alignment.

Conflict resolution: when two assets claim the same sentence, the one with
the higher relevance_score wins; the loser keeps its original assignment.

Caching: each (asset_hash, scene_context_hash) pair is cached in the shared
pipeline diskcache so re-runs after partial failures cost nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import structlog
from google import genai
from google.genai import types

from vidmaxx.config.constants import (
    GEMINI_GCP_LOCATION,
    GEMINI_VLM_MODEL,
    VLM_CONCURRENCY,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Script
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_STUDY_PROMPT = """\
You are a documentary video editor reviewing a single visual asset.

Pick the sentence below this visual best illustrates. Score the fit (0.0=irrelevant, 1.0=perfect).

Scene sentences:
{scene_sentences}

Return ONLY valid JSON (no markdown, no prose):
{{"asset_description":"3-5 word description","mood":"clinical|ominous|warm|neutral|alarming|institutional","best_sentence_id":"chXX_sYY","relevance_score":0.0,"reason":"brief reason"}}"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> None:
    paths = state_mgr.paths(slug)
    script = Script.model_validate_json(paths.script.read_text())

    log.info("stage_vlm_select_start", slug=slug, sentences=len(script.sentences))

    with state_mgr.run_stage(slug, PipelineStage.VLM_SELECT) as paths:
        client = genai.Client(
            vertexai=True,
            **({} if not settings.gcp_project_id else {"project": settings.gcp_project_id}),
            location=GEMINI_GCP_LOCATION,
        )

        scenes = _group_by_scene(script.sentences)
        sem = asyncio.Semaphore(VLM_CONCURRENCY)

        all_manifest: list[dict] = []

        for scene_idx, scene_sentences in scenes.items():
            scene_assets: list[tuple[str, Path]] = []  # (sentence_id, asset_path)
            for s in scene_sentences:
                asset = _find_asset(paths.assets_dir, s.id)
                if asset:
                    scene_assets.append((s.id, asset))

            if not scene_assets:
                log.warning("vlm_no_assets_in_scene", scene=scene_idx)
                continue

            sentence_context = "\n".join(
                f"{s.id}: {s.text}" for s in scene_sentences
            )

            study_results = await asyncio.gather(*[
                _study_asset(client, asset_path, sentence_context, sem, cache)
                for _, asset_path in scene_assets
            ])

            # Attach original sentence_id to each result
            for (orig_sid, asset_path), result in zip(scene_assets, study_results):
                result["original_sentence_id"] = orig_sid
                result["asset_path"] = str(asset_path)

            assignments = _resolve_assignments(scene_sentences, scene_assets, study_results)
            all_manifest.extend(assignments)

        paths.vlm_manifest_json.write_text(json.dumps(all_manifest, indent=2))

        reassigned = sum(1 for m in all_manifest if m.get("reassigned"))
        low_score  = sum(1 for m in all_manifest if m.get("relevance_score", 1.0) < 0.4)
        log.info(
            "stage_vlm_select_done",
            slug=slug,
            total=len(all_manifest),
            reassigned=reassigned,
            low_score=low_score,
        )


# ---------------------------------------------------------------------------
# Asset study
# ---------------------------------------------------------------------------

async def _study_asset(
    client: genai.Client,
    asset_path: Path,
    sentence_context: str,
    sem: asyncio.Semaphore,
    cache: PipelineCache,
) -> dict:
    # Cache key: hash of file content + scene sentences (sentences may change if script is re-run)
    file_hash  = hashlib.md5(asset_path.read_bytes()).hexdigest()[:16]
    ctx_hash   = hashlib.md5(sentence_context.encode()).hexdigest()[:16]
    cache_key  = f"vlm:{file_hash}:{ctx_hash}"

    cached = cache.get(cache_key)
    if cached:
        return cached

    jpeg_bytes = _to_jpeg(asset_path)

    async with sem:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=GEMINI_VLM_MODEL,
                    contents=[
                        types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        _STUDY_PROMPT.format(scene_sentences=sentence_context),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        maxOutputTokens=1024,
                        thinkingConfig=types.ThinkingConfig(thinkingBudget=0),
                    ),
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            log.warning("vlm_timeout", asset=asset_path.name)
            return {
                "asset_description": "timeout",
                "mood": "neutral",
                "best_sentence_id": None,
                "relevance_score": 0.0,
                "reason": "vlm timeout",
            }

    try:
        raw = response.text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        # Strip single-quote wrapper (Vertex AI JSON mime type artefact)
        if raw.startswith("'"):
            raw = raw[1:]
        if raw.endswith("'"):
            raw = raw[:-1]
        # Try full parse first; fall back to regex extraction for truncated responses
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = _regex_extract(raw)
            if not result.get("best_sentence_id"):
                log.warning("vlm_parse_error", asset=asset_path.name, raw=repr(raw[:200]))
                return {
                    "asset_description": "parse error",
                    "mood": "neutral",
                    "best_sentence_id": None,
                    "relevance_score": 0.0,
                    "reason": "vlm parse failed",
                }
    except AttributeError as exc:
        log.warning("vlm_parse_error", asset=asset_path.name, error=str(exc))
        # Don't cache — let re-runs retry
        return {
            "asset_description": "parse error",
            "mood": "neutral",
            "best_sentence_id": None,
            "relevance_score": 0.0,
            "reason": "vlm parse failed",
        }

    cache.set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

def _resolve_assignments(
    scene_sentences: list,
    scene_assets: list[tuple[str, Path]],
    study_results: list[dict],
) -> list[dict]:
    original: dict[str, str] = {sid: str(p) for sid, p in scene_assets}
    valid_ids = {s.id for s in scene_sentences}

    # Group claims by target sentence_id; ignore claims outside this scene
    claims: dict[str, list[dict]] = {}
    for result in study_results:
        target = result.get("best_sentence_id")
        if target and target in valid_ids:
            claims.setdefault(target, []).append(result)

    final: list[dict] = []
    used_assets: set[str] = set()

    for sentence in scene_sentences:
        sid = sentence.id
        candidates = sorted(
            claims.get(sid, []),
            key=lambda x: x.get("relevance_score", 0.0),
            reverse=True,
        )

        winner = next(
            (c for c in candidates if c["asset_path"] not in used_assets),
            None,
        )

        if winner is None:
            # No valid claim — keep original assignment
            final.append({
                "sentence_id": sid,
                "asset_path": original.get(sid),
                "mood": "neutral",
                "relevance_score": 0.3,
                "reassigned": False,
                "reason": "no_vlm_claim",
            })
        else:
            used_assets.add(winner["asset_path"])
            reassigned = winner["asset_path"] != original.get(sid)
            final.append({
                "sentence_id": sid,
                "asset_path": winner["asset_path"],
                "mood": winner.get("mood", "neutral"),
                "relevance_score": winner.get("relevance_score", 0.0),
                "reassigned": reassigned,
                "reason": winner.get("reason", ""),
                "asset_description": winner.get("asset_description", ""),
            })

    return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regex_extract(raw: str) -> dict:
    """Extract key VLM fields from truncated JSON using regex."""
    def _str(field: str) -> str | None:
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', raw)
        return m.group(1) if m else None

    def _num(field: str) -> float:
        m = re.search(rf'"{field}"\s*:\s*([\d.]+)', raw)
        return float(m.group(1)) if m else 0.0

    return {
        "asset_description": _str("asset_description") or "",
        "mood": _str("mood") or "neutral",
        "best_sentence_id": _str("best_sentence_id"),
        "relevance_score": _num("relevance_score"),
        "reason": _str("reason") or "truncated",
    }


def _group_by_scene(sentences: list) -> dict[int, list]:
    scenes: dict[int, list] = {}
    for s in sentences:
        try:
            scene_idx = int(s.id.split("_")[0].replace("ch", ""))
        except (ValueError, IndexError):
            scene_idx = 0
        scenes.setdefault(scene_idx, []).append(s)
    return scenes


def _find_asset(assets_dir: Path, sentence_id: str) -> Path | None:
    for ext in (*_VIDEO_EXTS, *_IMAGE_EXTS):
        p = assets_dir / f"{sentence_id}{ext}"
        if p.exists():
            return p
    # glob fallback (handles uppercase extensions)
    matches = [
        f for f in assets_dir.glob(f"{sentence_id}.*")
        if f.suffix.lower() in _VIDEO_EXTS | _IMAGE_EXTS
    ]
    return matches[0] if matches else None


def _to_jpeg(asset_path: Path) -> bytes:
    """Return JPEG bytes for the asset — extracts a frame from video, reads image directly."""
    if asset_path.suffix.lower() in _VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", "2",
                "-i", str(asset_path),
                "-vframes", "1",
                "-q:v", "5",
                tmp_path,
            ],
            capture_output=True,
            check=True,
        )
        data = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
        return data

    # Image — re-encode to JPEG via ffmpeg to normalise format (webp, png → jpeg)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(asset_path), "-q:v", "5", tmp_path],
        capture_output=True,
        check=True,
    )
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data
