"""
Asset fetcher.

Per-sentence flow:
  1. Search all three sources in parallel (Pexels, Pixabay, Wikimedia).
  2. Pool all candidates.
  3. CLIP-rank them against visual_query.
  4. Take the highest-scoring candidate above CLIP_SIMILARITY_THRESHOLD.
  5. Download the winner to assets_dir/{sentence_id}.{ext}.
  6. Return an Asset; append credit to credits list.

If no candidate clears the threshold, we take the best available rather
than returning nothing — a low-score asset is better than a black frame.
"""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

from vidmaxx.config.constants import (
    ASSET_CANDIDATES_PER_SOURCE,
    CLIP_SIMILARITY_THRESHOLD,
)
from vidmaxx.models.asset import Asset, AssetCandidate, AssetSource, AssetType
from vidmaxx.models.sentence import Sentence
from vidmaxx.services.assets.base import AssetSourceBase
from vidmaxx.services.assets.internet_archive import InternetArchiveSource
from vidmaxx.services.assets.pexels import PexelsSource
from vidmaxx.services.assets.pixabay import PixabaySource
from vidmaxx.services.assets.wikimedia import WikimediaSource
from vidmaxx.services.vision.clip_ranker import CLIPRanker

log = structlog.get_logger(__name__)

_DOWNLOAD_TIMEOUT = httpx.Timeout(60.0)


class AssetFetcher:
    def __init__(
        self,
        pexels_key: str,
        pixabay_key: str,
        ranker: CLIPRanker,
    ) -> None:
        self._sources: list[AssetSourceBase] = [
            PexelsSource(pexels_key),
            PixabaySource(pixabay_key),
            WikimediaSource(),
            InternetArchiveSource(),
        ]
        self._ranker = ranker

    async def fetch(
        self,
        sentence: Sentence,
        assets_dir: Path,
        clip_query: str | None = None,
        prefer_type: AssetType | None = None,
    ) -> Asset | None:
        query = sentence.visual_query or sentence.text[:60]
        rank_target = clip_query or query

        # 1. Search all sources in parallel
        images_only = prefer_type == AssetType.IMAGE
        all_candidates = await self._search_all(query, images_only=images_only)
        if not all_candidates:
            # Fallback: retry without image restriction
            if images_only:
                all_candidates = await self._search_all(query, images_only=False)
            if not all_candidates:
                log.warning("asset_no_candidates", id=sentence.id, query=query)
                return None

        # 2. CLIP-rank
        ranked = await self._ranker.rank(rank_target, all_candidates)

        if not ranked:
            return None

        winner = ranked[0]
        if winner.clip_score < CLIP_SIMILARITY_THRESHOLD:
            log.debug(
                "asset_below_threshold",
                id=sentence.id,
                score=winner.clip_score,
                threshold=CLIP_SIMILARITY_THRESHOLD,
            )
            return None
        log.debug("asset_winner", id=sentence.id, score=winner.clip_score, source=winner.source)

        # 3. Download
        local_path = await self._download(winner, sentence.id, assets_dir)
        if local_path is None:
            return None

        return Asset(
            sentence_id=sentence.id,
            source=winner.source,
            asset_type=winner.asset_type,
            local_path=local_path,
            remote_url=winner.remote_url,
            page_url=winner.page_url,
            license=winner.license,
            attribution_required=winner.attribution_required,
            attribution_text=winner.attribution_text,
            clip_score=winner.clip_score,
            width=winner.width,
            height=winner.height,
            duration_sec=winner.duration_sec,
        )

    async def _search_all(self, query: str, images_only: bool = False) -> list[AssetCandidate]:
        results = await asyncio.gather(
            *[src.search(query, ASSET_CANDIDATES_PER_SOURCE, images_only=images_only) for src in self._sources],
            return_exceptions=True,
        )
        candidates: list[AssetCandidate] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("asset_source_failed", error=str(r))
            else:
                candidates.extend(r)
        return candidates

    async def _download(
        self,
        candidate: AssetCandidate,
        sentence_id: str,
        assets_dir: Path,
    ) -> Path | None:
        ext = _ext_from_url(candidate.remote_url, candidate.asset_type)
        out_path = assets_dir / f"{sentence_id}{ext}"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT
            ) as client:
                async with client.stream("GET", candidate.remote_url) as r:
                    r.raise_for_status()
                    with tmp_path.open("wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
            tmp_path.rename(out_path)  # atomic: only exists if fully written
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)  # discard partial download
            log.warning("asset_download_failed", id=sentence_id, error=str(exc))
            return None

        log.debug("asset_downloaded", id=sentence_id, path=str(out_path))
        return out_path


def _ext_from_url(url: str, asset_type: AssetType) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return ".mp4" if asset_type == AssetType.VIDEO else ".jpg"
