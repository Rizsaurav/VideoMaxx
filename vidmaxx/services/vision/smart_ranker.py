"""
SmartRanker — tries RemoteSigLIPRanker first, falls back to CLIPRanker.

SigLIP is prioritized because it is a larger model (ViT-L/16 vs ViT-B/32)
and runs on a dedicated GPU. CLIP acts as a local safety net when the
remote server is unreachable or returns an error.

Both rankers share the same rank(query, candidates) interface.
"""

from __future__ import annotations

import structlog

from vidmaxx.models.asset import AssetCandidate
from vidmaxx.services.vision.clip_ranker import CLIPRanker
from vidmaxx.services.vision.siglip_ranker import RemoteSigLIPRanker
from vidmaxx.state.cache import PipelineCache

log = structlog.get_logger(__name__)


class SmartRanker:
    def __init__(
        self,
        siglip_url: str,
        clip_device: str = "cpu",
        cache: PipelineCache | None = None,
    ) -> None:
        self._siglip = RemoteSigLIPRanker(base_url=siglip_url, cache=cache)
        self._clip: CLIPRanker | None = None          # lazy — only loaded on fallback
        self._clip_device = clip_device
        self._cache = cache

    def _get_clip(self) -> CLIPRanker:
        if self._clip is None:
            log.info("smart_ranker_loading_clip_fallback")
            self._clip = CLIPRanker(device=self._clip_device, cache=self._cache)
        return self._clip

    def unload(self) -> None:
        if self._clip is not None:
            self._clip.unload()
            self._clip = None

    def __enter__(self) -> "SmartRanker":
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    async def rank(
        self,
        query: str,
        candidates: list[AssetCandidate],
    ) -> list[AssetCandidate]:
        try:
            result = await self._siglip.rank(query, candidates)
            log.debug("smart_ranker_used_siglip", query=query[:50])
            return result
        except Exception as exc:
            log.warning("smart_ranker_siglip_failed_using_clip", error=str(exc))
            return await self._get_clip().rank(query, candidates)
