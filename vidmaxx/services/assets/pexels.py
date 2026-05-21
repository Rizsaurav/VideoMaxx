"""
Pexels asset source.

Videos are preferred (B-roll motion reads better on screen). Photos are
fetched as fallback when the video search returns fewer than n results.

API docs:
  Videos: GET https://api.pexels.com/videos/search
  Photos: GET https://api.pexels.com/v1/search
"""

import random

import httpx
import structlog

from vidmaxx.models.asset import AssetCandidate, AssetSource, AssetType
from vidmaxx.services.assets.base import AssetSourceBase

log = structlog.get_logger(__name__)

_VIDEO_URL = "https://api.pexels.com/videos/search"
_PHOTO_URL = "https://api.pexels.com/v1/search"
_TIMEOUT = httpx.Timeout(15.0)


class PexelsSource(AssetSourceBase):
    def __init__(self, api_key: str) -> None:
        self._headers = {"Authorization": api_key}

    async def search(self, query: str, n: int, images_only: bool = False) -> list[AssetCandidate]:
        async with httpx.AsyncClient(headers=self._headers, timeout=_TIMEOUT) as client:
            if images_only:
                return await self._search_photos(client, query, n)
            videos = await self._search_videos(client, query, n)
            if len(videos) < n:
                photos = await self._search_photos(client, query, n - len(videos))
                return videos + photos
            return videos

    async def _search_videos(
        self, client: httpx.AsyncClient, query: str, n: int
    ) -> list[AssetCandidate]:
        try:
            r = await client.get(
                _VIDEO_URL,
                params={"query": query, "per_page": n, "orientation": "landscape", "page": random.randint(1, 4)},
            )
            r.raise_for_status()
        except Exception as exc:
            log.warning("pexels_video_search_failed", query=query, error=str(exc))
            return []

        candidates = []
        for v in r.json().get("videos", []):
            # Pick the highest-quality landscape file
            files = sorted(
                v.get("video_files", []),
                key=lambda f: f.get("width", 0),
                reverse=True,
            )
            best = next(
                (f for f in files if f.get("file_type") == "video/mp4"),
                files[0] if files else None,
            )
            if not best:
                continue
            candidates.append(
                AssetCandidate(
                    source=AssetSource.PEXELS,
                    asset_type=AssetType.VIDEO,
                    remote_url=best["link"],
                    thumbnail_url=v.get("image", ""),
                    page_url=v.get("url", ""),
                    license="Pexels License",
                    attribution_required=False,
                    width=best.get("width", 0),
                    height=best.get("height", 0),
                    duration_sec=float(v.get("duration", 0)),
                )
            )
        return candidates

    async def _search_photos(
        self, client: httpx.AsyncClient, query: str, n: int
    ) -> list[AssetCandidate]:
        try:
            r = await client.get(
                _PHOTO_URL,
                params={"query": query, "per_page": n, "orientation": "landscape", "page": random.randint(1, 4)},
            )
            r.raise_for_status()
        except Exception as exc:
            log.warning("pexels_photo_search_failed", query=query, error=str(exc))
            return []

        candidates = []
        for p in r.json().get("photos", []):
            src = p.get("src", {})
            url = src.get("original") or src.get("large2x") or src.get("large", "")
            if not url:
                continue
            candidates.append(
                AssetCandidate(
                    source=AssetSource.PEXELS,
                    asset_type=AssetType.IMAGE,
                    remote_url=url,
                    page_url=p.get("url", ""),
                    license="Pexels License",
                    attribution_required=False,
                    width=p.get("width", 0),
                    height=p.get("height", 0),
                )
            )
        return candidates
