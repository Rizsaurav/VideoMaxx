"""
Credits ledger — read/write helpers for projects/{slug}/credits.json.

credits.json is a flat list of attribution records, one per fetched asset.
Only assets with attribution_required=True are legally mandatory, but we
log everything for provenance.
"""

import json
from pathlib import Path

from vidmaxx.models.asset import Asset


def write_credits(assets: list[Asset], path: Path) -> None:
    records = [
        {
            "sentence_id": a.sentence_id,
            "source": str(a.source),
            "license": a.license,
            "attribution_required": a.attribution_required,
            "attribution_text": a.attribution_text,
            "page_url": a.page_url,
            "remote_url": a.remote_url,
        }
        for a in assets
    ]
    path.write_text(json.dumps(records, indent=2))


def read_credits(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def required_attributions(path: Path) -> list[dict]:
    """Return only the records that legally require attribution."""
    return [r for r in read_credits(path) if r.get("attribution_required")]
