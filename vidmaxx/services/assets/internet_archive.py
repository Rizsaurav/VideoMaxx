"""
Internet Archive source — public domain and CC-licensed video clips.

Best for: congressional hearings, government press releases, news broadcasts,
documentary footage, historical records. All returned content is public domain
or CC licensed.

No API key required. Uses the Archive.org advancedsearch and metadata APIs.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog

from vidmaxx.models.asset import AssetCandidate, AssetSource, AssetType
from vidmaxx.services.assets.base import AssetSourceBase

log = structlog.get_logger(__name__)

_SEARCH_URL = "https://archive.org/advancedsearch.php"
_METADATA_URL = "https://archive.org/metadata/{identifier}/files"
_DETAILS_URL = "https://archive.org/details/{identifier}"
_DOWNLOAD_BASE = "https://archive.org/download"
_TIMEOUT = httpx.Timeout(20.0)
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB — IA files can be enormous


class InternetArchiveSource(AssetSourceBase):

    async def search(self, query: str, max_results: int = 5, images_only: bool = False) -> list[AssetCandidate]:
        params = {
            "q": (
                f"({query}) AND mediatype:movies "
                "AND (licenseurl:(*creativecommons*) "
                "OR licenseurl:(*publicdomain*) "
                "OR NOT licenseurl:[* TO *])"
            ),
            "fl": "identifier,title,licenseurl",
            "sort": "downloads desc",
            "rows": max_results * 3,
            "output": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_SEARCH_URL, params=params)
                resp.raise_for_status()
                docs = resp.json().get("response", {}).get("docs", [])
        except Exception as exc:
            log.warning("archive_search_failed", query=query, error=str(exc))
            return []

        if not docs:
            return []

        candidates = await asyncio.gather(*[
            self._get_candidate(doc)
            for doc in docs[:max_results * 2]
        ])
        return [c for c in candidates if c is not None][:max_results]

    async def _get_candidate(self, doc: dict) -> AssetCandidate | None:
        identifier = doc.get("identifier", "")
        if not identifier:
            return None

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_METADATA_URL.format(identifier=identifier))
                resp.raise_for_status()
                files = resp.json().get("result", [])
        except Exception:
            return None

        # Prefer derivative (compressed) mp4s — smaller, faster download
        mp4s = [
            f for f in files
            if f.get("name", "").endswith(".mp4")
            and f.get("source") == "derivative"
        ]
        if not mp4s:
            # Fall back to any mp4
            mp4s = [f for f in files if f.get("name", "").endswith(".mp4")]
        if not mp4s:
            return None

        mp4s = [f for f in mp4s if int(f.get("size", 999_999_999)) <= _MAX_FILE_BYTES]
        if not mp4s:
            return None
        mp4s.sort(key=lambda f: int(f.get("size", 999_999_999)))
        chosen = mp4s[0]
        license_url = doc.get("licenseurl", "")

        return AssetCandidate(
            source=AssetSource.INTERNET_ARCHIVE,
            asset_type=AssetType.VIDEO,
            remote_url=f"{_DOWNLOAD_BASE}/{identifier}/{chosen['name']}",
            page_url=_DETAILS_URL.format(identifier=identifier),
            license=license_url or "public_domain",
            attribution_required=bool(license_url),
            attribution_text=(
                f"{doc.get('title', identifier)} — Internet Archive"
                if license_url else ""
            ),
            width=1280,
            height=720,
            duration_sec=0.0,
            clip_score=0.0,
        )
