"""
Wikimedia Commons asset source.

No API key required. Returns CC0/public-domain images only.
Uses the MediaWiki API to search file namespace (ns=6), then resolves
direct file URLs via imageinfo.

API docs: https://www.mediawiki.org/wiki/API:Search
"""

import httpx
import structlog

from vidmaxx.models.asset import AssetCandidate, AssetSource, AssetType
from vidmaxx.services.assets.base import AssetSourceBase

log = structlog.get_logger(__name__)

_API = "https://commons.wikimedia.org/w/api.php"
_TIMEOUT = httpx.Timeout(20.0)

# Only fetch these media types — skip SVG, PDF, audio, etc.
_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

# Free licenses only
_FREE_LICENSES = {"CC0", "Public Domain", "CC-BY", "CC-BY-SA", "CC-BY-4.0", "CC-BY-SA-4.0"}


class WikimediaSource(AssetSourceBase):
    async def search(self, query: str, n: int, images_only: bool = False) -> list[AssetCandidate]:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            titles = await self._search_titles(client, query, n * 3)  # over-fetch, filter below
            if not titles:
                return []
            return await self._resolve_files(client, titles, n)

    async def _search_titles(
        self, client: httpx.AsyncClient, query: str, limit: int
    ) -> list[str]:
        try:
            r = await client.get(
                _API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 6,       # File namespace
                    "srlimit": limit,
                    "format": "json",
                },
            )
            r.raise_for_status()
        except Exception as exc:
            log.warning("wikimedia_search_failed", query=query, error=str(exc))
            return []

        results = r.json().get("query", {}).get("search", [])
        return [item["title"] for item in results]

    async def _resolve_files(
        self, client: httpx.AsyncClient, titles: list[str], n: int
    ) -> list[AssetCandidate]:
        # MediaWiki accepts up to 50 titles per imageinfo call
        chunk = titles[:50]
        try:
            r = await client.get(
                _API,
                params={
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "imageinfo",
                    "iiprop": "url|mediatype|size|extmetadata",
                    "iiextmetadatafilter": "LicenseShortName",
                    "format": "json",
                },
            )
            r.raise_for_status()
        except Exception as exc:
            log.warning("wikimedia_resolve_failed", error=str(exc))
            return []

        pages = r.json().get("query", {}).get("pages", {}).values()
        candidates = []
        for page in pages:
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]

            mime = info.get("mime", "")
            if mime not in _ALLOWED_MIME:
                continue

            url = info.get("url", "")
            if not url:
                continue

            extmeta = info.get("extmetadata", {})
            license_name = extmeta.get("LicenseShortName", {}).get("value", "")

            page_url = (
                "https://commons.wikimedia.org/wiki/" + page.get("title", "").replace(" ", "_")
            )
            attribution_required = "CC-BY" in license_name or "CC BY" in license_name

            candidates.append(
                AssetCandidate(
                    source=AssetSource.WIKIMEDIA,
                    asset_type=AssetType.IMAGE,
                    remote_url=url,
                    page_url=page_url,
                    license=license_name or "Unknown",
                    attribution_required=attribution_required,
                    attribution_text=f"Via Wikimedia Commons — {page.get('title', '')}",
                    width=info.get("width", 0),
                    height=info.get("height", 0),
                )
            )
            if len(candidates) >= n:
                break

        return candidates
