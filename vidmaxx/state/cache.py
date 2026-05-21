"""
diskcache wrappers for:
  - LLM responses  (keyed by hash of system + prompt)
  - CLIP embeddings (keyed by remote asset URL)
  - Voice reference embedding (keyed by hash of voice_ref.wav bytes)
  - Generic passthrough (architect outline, etc.)

Single shared Cache instance — separated by key prefix, shared eviction and
size limits. Call close() when the pipeline exits or use as a context manager.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
    return h.hexdigest()


class PipelineCache:
    def __init__(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(
            str(cache_dir),
            size_limit=int(10e9),   # 10 GB cap
            eviction_policy="least-recently-used",
        )

    # ------------------------------------------------------------------
    # LLM cache
    # ------------------------------------------------------------------

    def _llm_key(self, system: str, prompt: str) -> str:
        return f"llm:{_hash(system, prompt)}"

    def get_llm(self, system: str, prompt: str) -> str | None:
        return self._cache.get(self._llm_key(system, prompt))

    def set_llm(self, system: str, prompt: str, response: str) -> None:
        self._cache.set(self._llm_key(system, prompt), response)

    # ------------------------------------------------------------------
    # CLIP embedding cache  (remote_url → list[float])
    # ------------------------------------------------------------------

    def _clip_key(self, remote_url: str) -> str:
        return f"clip:{_hash(remote_url)}"

    def get_clip(self, remote_url: str) -> list[float] | None:
        return self._cache.get(self._clip_key(remote_url))

    def set_clip(self, remote_url: str, embedding: list[float]) -> None:
        self._cache.set(self._clip_key(remote_url), embedding)

    # ------------------------------------------------------------------
    # Voice reference embedding cache
    # ------------------------------------------------------------------

    def _voice_key(self, voice_ref_path: Path) -> str:
        digest = hashlib.sha256(voice_ref_path.read_bytes()).hexdigest()
        return f"voice:{digest}"

    def get_voice_embed(self, voice_ref_path: Path) -> Any | None:
        return self._cache.get(self._voice_key(voice_ref_path))

    def set_voice_embed(self, voice_ref_path: Path, embedding: Any) -> None:
        self._cache.set(self._voice_key(voice_ref_path), embedding)

    # ------------------------------------------------------------------
    # Generic passthrough (for anything else that needs caching)
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache.set(key, value)

    def invalidate(self, key: str) -> None:
        self._cache.delete(key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> "PipelineCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()
