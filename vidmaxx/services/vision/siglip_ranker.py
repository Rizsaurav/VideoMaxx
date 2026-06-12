"""
Remote SigLIP asset ranker — calls the GPU-hosted ranking server via ngrok.

API contract (discovered from /openapi.json):
  POST /v1/assets/upload   multipart files[] → {"job_id": "...", "files": [...]}
  POST /v1/assets/rank     ?sentence=&job_id= → {"ranked_assets": [["name", score], ...]}

Flow per rank() call:
  1. Fetch all candidate images concurrently (httpx).
  2. Upload as multipart to /v1/assets/upload — filenames are positional ("0.jpg" etc.)
     so scores can be mapped back to candidates by index.
  3. Call /v1/assets/rank with the query sentence.
  4. Normalize scores within the batch (divide by max) so the best candidate is always
     1.0 — raw SigLIP scores are tiny and would fail the pipeline's CLIP threshold.
  5. Return candidates sorted by score descending.

Caching: rank results cached by (query_hash, sorted_urls_hash) in PipelineCache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import structlog

from vidmaxx.models.asset import AssetCandidate
from vidmaxx.state.cache import PipelineCache

log = structlog.get_logger(__name__)

_FETCH_TIMEOUT  = httpx.Timeout(15.0)
_UPLOAD_TIMEOUT = httpx.Timeout(60.0)


def _cache_key(query: str, urls: list[str]) -> str:
    h = hashlib.sha256()
    h.update(query.encode())
    for u in sorted(urls):
        h.update(u.encode())
    return f"siglip:{h.hexdigest()[:20]}"


class RemoteSigLIPRanker:
    def __init__(self, base_url: str, cache: PipelineCache | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._cache = cache
        log.info("siglip_ranker_ready", url=self._base_url)

    def unload(self) -> None:
        pass  # stateless HTTP client — nothing to free

    def __enter__(self) -> "RemoteSigLIPRanker":
        return self

    def __exit__(self, *_) -> None:
        pass

    async def rank(
        self,
        query: str,
        candidates: list[AssetCandidate],
    ) -> list[AssetCandidate]:
        if not candidates:
            return []

        urls = [c.clip_url for c in candidates]

        # Cache check
        if self._cache is not None:
            key = _cache_key(query, urls)
            cached = self._cache.get(key)
            if cached is not None:
                score_map: dict[int, float] = json.loads(cached)
                scored = [
                    c.model_copy(update={"clip_score": score_map.get(str(i), 0.0)})
                    for i, c in enumerate(candidates)
                ]
                return sorted(scored, key=lambda c: c.clip_score, reverse=True)

        # 1. Fetch all images concurrently
        image_data: list[tuple[int, bytes]] = []
        async with httpx.AsyncClient(follow_redirects=True, timeout=_FETCH_TIMEOUT) as client:
            import asyncio
            async def _fetch(i: int, url: str) -> tuple[int, bytes | None]:
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                    return i, r.content
                except Exception as exc:
                    log.warning("siglip_image_fetch_failed", url=url, error=str(exc))
                    return i, None

            results = await asyncio.gather(*[_fetch(i, u) for i, u in enumerate(urls)])

        image_data = [(i, data) for i, data in results if data is not None]
        if not image_data:
            log.warning("siglip_all_fetches_failed", query=query)
            return candidates  # return unscored rather than empty

        # 2. Upload images as multipart
        files = [
            ("files", (f"{i}.jpg", data, "image/jpeg"))
            for i, data in image_data
        ]
        async with httpx.AsyncClient(follow_redirects=True, timeout=_UPLOAD_TIMEOUT) as client:
            upload_resp = await client.post(
                f"{self._base_url}/v1/assets/upload",
                files=files,
            )
            upload_resp.raise_for_status()
            job_id = upload_resp.json()["job_id"]

            # 3. Rank
            rank_resp = await client.post(
                f"{self._base_url}/v1/assets/rank",
                params={"sentence": query, "job_id": job_id},
            )
            rank_resp.raise_for_status()
            ranked_assets: list[list] = rank_resp.json()["ranked_assets"]

        # 4. Map filename → index → score
        raw_scores: dict[int, float] = {}
        for filename, score in ranked_assets:
            try:
                idx = int(Path(filename).stem)
                raw_scores[idx] = float(score)
            except (ValueError, AttributeError):
                pass

        # Normalize within batch so best candidate = 1.0
        max_score = max(raw_scores.values(), default=1.0)
        if max_score <= 0:
            max_score = 1.0

        score_map_str: dict[str, float] = {}
        scored: list[AssetCandidate] = []
        for i, c in enumerate(candidates):
            raw = raw_scores.get(i, 0.0)
            norm = raw / max_score
            score_map_str[str(i)] = norm
            scored.append(c.model_copy(update={"clip_score": norm}))

        log.debug(
            "siglip_ranked",
            query=query[:60],
            candidates=len(candidates),
            uploaded=len(image_data),
            best_score=round(max_score, 6),
        )

        # Cache
        if self._cache is not None:
            self._cache.set(_cache_key(query, urls), json.dumps(score_map_str))

        return sorted(scored, key=lambda c: c.clip_score, reverse=True)
