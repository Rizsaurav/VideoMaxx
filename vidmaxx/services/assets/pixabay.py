"""
Pixabay asset source.

Videos preferred; images as fallback (same pattern as Pexels).

API docs:
  Videos: GET https://pixabay.com/api/videos/
  Images: GET https://pixabay.com/api/
"""

import random

import httpx
import structlog

from vidmaxx.models.asset import AssetCandidate, AssetSource, AssetType
from vidmaxx.services.assets.base import AssetSourceBase

log = structlog.get_logger(__name__)

_VIDEO_URL = "https://pixabay.com/api/videos/"
_IMAGE_URL = "https://pixabay.com/api/"
_TIMEOUT = httpx.Timeout(15.0)


class PixabaySource(AssetSourceBase):
    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def search(self, query: str, n: int, images_only: bool = False) -> list[AssetCandidate]:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            if images_only:
                return await self._search_images(client, query, n)
            videos = await self._search_videos(client, query, n)
            if len(videos) < n:
                images = await self._search_images(client, query, n - len(videos))
                return videos + images
            return videos

    async def _search_videos(
        self, client: httpx.AsyncClient, query: str, n: int
    ) -> list[AssetCandidate]:
        try:
            r = await client.get(
                _VIDEO_URL,
                params={
                    "key": self._key,
                    "q": query,
                    "per_page": n,
                    "video_type": "film",
                    "page": random.randint(1, 4),
                },
            )
            r.raise_for_status()
        except Exception as exc:
            log.warning("pixabay_video_search_failed", query=query, error=str(exc))
            return []

        candidates = []
        for hit in r.json().get("hits", []):
            # Prefer "large" quality; fall back to "medium"
            videos = hit.get("videos", {})
            file_info = videos.get("large") or videos.get("medium") or {}
            url = file_info.get("url", "")
            if not url:
                continue
            # Pixabay video API doesn't return a thumbnail URL directly;
            # use the picture_id to construct the thumbnail
            picture_id = hit.get("picture_id", "")
            thumbnail_url = (
                f"https://i.vimeocdn.com/video/{picture_id}_640x360.jpg"
                if picture_id
                else ""
            )
            candidates.append(
                AssetCandidate(
                    source=AssetSource.PIXABAY,
                    asset_type=AssetType.VIDEO,
                    remote_url=url,
                    thumbnail_url=thumbnail_url,
                    page_url=hit.get("pageURL", ""),
                    license="Pixabay License",
                    attribution_required=False,
                    width=file_info.get("width", 0),
                    height=file_info.get("height", 0),
                    duration_sec=float(hit.get("duration", 0)),
                )
            )
        return candidates

    async def _search_images(
        self, client: httpx.AsyncClient, query: str, n: int
    ) -> list[AssetCandidate]:
        try:
            r = await client.get(
                _IMAGE_URL,
                params={
                    "key": self._key,
                    "q": query,
                    "per_page": n,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "page": random.randint(1, 4),
                },
            )
            r.raise_for_status()
        except Exception as exc:
            log.warning("pixabay_image_search_failed", query=query, error=str(exc))
            return []

        candidates = []
        for hit in r.json().get("hits", []):
            url = hit.get("largeImageURL") or hit.get("webformatURL", "")
            if not url:
                continue
            candidates.append(
                AssetCandidate(
                    source=AssetSource.PIXABAY,
                    asset_type=AssetType.IMAGE,
                    remote_url=url,
                    page_url=hit.get("pageURL", ""),
                    license="Pixabay License",
                    attribution_required=False,
                    width=hit.get("imageWidth", 0),
                    height=hit.get("imageHeight", 0),
                )
            )
        return candidates
