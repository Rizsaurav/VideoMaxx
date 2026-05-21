"""
Stage 9 — Publish

Reads:   paths.final_video       (out/video.mp4)
         paths.verified_fact_sheet (verified_fact_sheet.json — tags + description summary)
         paths.credits_json      (credits.json — required attributions for description)
         paths.script            (script.json — topic / chapter titles)
Writes:  paths.publish_json      (publish.json — YouTube video IDs + URLs)
Advances: PUBLISH → DONE

If YOUTUBE_CLIENT_ID / CLIENT_SECRET / REFRESH_TOKEN are not set in .env,
the stage is a no-op: it logs a notice and advances to DONE immediately.

YouTube API
-----------
Uses the YouTube Data API v3 resumable upload (no google-api-python-client
dependency).  OAuth2 credentials are supplied as a pre-obtained refresh
token; the stage refreshes the access token on each run.

Upload flow:
  1. POST oauth2.googleapis.com/token  → access_token
  2. POST googleapis.com/upload/youtube/v3/videos?uploadType=resumable
         with JSON metadata + X-Upload-Content-Length header  → upload URI
  3. PUT  {upload_uri} in 10 MB chunks until 200/201  → video_id

The default privacy is "private".  Change YOUTUBE_PRIVACY in .env to
"unlisted" or "public" once you've reviewed the video.

Shorts are uploaded as separate videos with '#Shorts' appended to the title.
"""

import json
from pathlib import Path

import httpx
import structlog
import typer

from vidmaxx.config.settings import Settings
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.factsheet import VerifiedFactSheet
from vidmaxx.models.script import Script
from vidmaxx.state.project_state import ProjectStateManager
from vidmaxx.utils.credits import required_attributions

log = structlog.get_logger(__name__)

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
_CATEGORY_EDUCATION = "27"
_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB — must be a multiple of 256 KB


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def _build_description(summary: str, credits_path: Path) -> str:
    parts: list[str] = []

    if summary:
        parts.append(summary[:2000])

    required = required_attributions(credits_path)
    if required:
        parts.append("\n\nSources & Credits:")
        for r in required:
            line = r.get("attribution_text") or r.get("page_url", "")
            if line:
                parts.append(f"  {line}")

    parts.append("\n\nMade with VidMaxx · https://github.com/vidmaxx/vidmaxx")
    return "\n".join(parts)[:5000]  # YouTube description hard limit


def _build_tags(fact_sheet: VerifiedFactSheet) -> list[str]:
    raw = list(fact_sheet.key_entities) + fact_sheet.topic.split()
    seen: dict[str, None] = {}
    for t in raw:
        key = t.lower().strip()
        if key and key not in seen:
            seen[key] = None

    tags: list[str] = []
    total_chars = 0
    for t in seen:
        if len(t) > 30:
            continue  # YouTube per-tag limit
        if total_chars + len(t) + 1 > 490:
            break
        tags.append(t)
        total_chars += len(t) + 1
    return tags


def _video_metadata(title: str, description: str, tags: list[str], privacy: str) -> dict:
    return {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": _CATEGORY_EDUCATION,
            "defaultLanguage": "en",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }


# ---------------------------------------------------------------------------
# OAuth2 token refresh
# ---------------------------------------------------------------------------

async def _refresh_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    http: httpx.AsyncClient,
) -> str:
    resp = await http.post(
        _TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError(f"Token refresh returned no access_token: {resp.text}")
    log.info("youtube_token_refreshed")
    return token


# ---------------------------------------------------------------------------
# Resumable upload
# ---------------------------------------------------------------------------

async def _initiate_upload(
    video_path: Path,
    metadata: dict,
    access_token: str,
    http: httpx.AsyncClient,
) -> str:
    """Send the initiation request; return the upload URI."""
    file_size = video_path.stat().st_size
    resp = await http.post(
        _UPLOAD_URL,
        params={"uploadType": "resumable", "part": "snippet,status"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(file_size),
            "Content-Type": "application/json; charset=UTF-8",
        },
        content=json.dumps(metadata).encode(),
    )
    resp.raise_for_status()
    upload_uri = resp.headers.get("location")
    if not upload_uri:
        raise RuntimeError("YouTube initiation response had no Location header")
    log.info("youtube_upload_initiated", size_mb=round(file_size / 1e6, 1))
    return upload_uri


async def _upload_chunks(
    video_path: Path,
    upload_uri: str,
    http: httpx.AsyncClient,
) -> str:
    """Stream the video file in 10 MB chunks. Returns the YouTube video_id."""
    file_size = video_path.stat().st_size
    uploaded = 0

    with open(video_path, "rb") as fh:
        while uploaded < file_size:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break

            end = uploaded + len(chunk) - 1
            resp = await http.put(
                upload_uri,
                headers={
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {uploaded}-{end}/{file_size}",
                },
                content=chunk,
            )

            if resp.status_code in (200, 201):
                video_id = resp.json().get("id")
                if not video_id:
                    raise RuntimeError(f"Upload complete but no video ID in response: {resp.text}")
                log.info("youtube_upload_complete", video_id=video_id)
                return video_id

            if resp.status_code == 308:  # Resume Incomplete — server ACKed this chunk
                range_header = resp.headers.get("range", "")
                if range_header:
                    # "bytes=0-10485759"
                    uploaded = int(range_header.split("-")[1]) + 1
                else:
                    uploaded = end + 1

                pct = round(uploaded / file_size * 100)
                log.info("youtube_upload_progress", pct=pct, mb=round(uploaded / 1e6, 1))
                typer.echo(f"    upload {pct}%", nl=False)
                typer.echo("\r", nl=False)
                continue

            resp.raise_for_status()

    raise RuntimeError("Upload loop exhausted without receiving a final 200/201")


async def _upload_one(
    video_path: Path,
    metadata: dict,
    access_token: str,
    http: httpx.AsyncClient,
) -> str:
    """Initiate + stream one video. Returns video_id."""
    uri = await _initiate_upload(video_path, metadata, access_token, http)
    return await _upload_chunks(video_path, uri, http)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    settings: Settings,
) -> None:
    """Upload the main video and any rendered shorts to YouTube.

    If YouTube credentials are not configured, advances to DONE silently.
    """
    if not settings.youtube_configured():
        log.info("publish_skipped_no_credentials", slug=slug)
        typer.echo(
            "  YouTube credentials not set — skipping publish.\n"
            "  Add YOUTUBE_CLIENT_ID / CLIENT_SECRET / REFRESH_TOKEN to .env to enable."
        )
        with state_mgr.run_stage(slug, PipelineStage.PUBLISH):
            pass  # just advance to DONE
        return

    paths = state_mgr.paths(slug)

    fact_sheet = VerifiedFactSheet.model_validate_json(paths.verified_fact_sheet.read_text())
    script = Script.model_validate_json(paths.script.read_text())
    privacy = getattr(settings, "youtube_privacy", "private")

    description = _build_description(fact_sheet.raw_summary, paths.credits_json)
    tags = _build_tags(fact_sheet)
    base_title = script.topic[:96]  # leave room for ' #Shorts' suffix

    log.info("stage_publish_start", slug=slug, privacy=privacy)

    # httpx client with no timeout on reads/writes (uploads can be slow).
    timeout = httpx.Timeout(connect=30.0, read=None, write=None, pool=30.0)

    published: dict[str, str] = {}  # label → YouTube URL

    with state_mgr.run_stage(slug, PipelineStage.PUBLISH) as paths:
        async with httpx.AsyncClient(timeout=timeout) as http:
            access_token = await _refresh_token(
                settings.youtube_client_id,
                settings.youtube_client_secret,
                settings.youtube_refresh_token,
                http,
            )

            # --- Main video ---
            if not paths.final_video.exists():
                raise RuntimeError(
                    f"final_video not found: {paths.final_video}\n"
                    "Run the render stage first."
                )

            typer.echo(f"  Uploading main video ({round(paths.final_video.stat().st_size / 1e6, 0):.0f} MB)...")
            main_meta = _video_metadata(base_title, description, tags, privacy)
            main_id = await _upload_one(paths.final_video, main_meta, access_token, http)
            main_url = f"https://youtu.be/{main_id}"
            published["main"] = main_url
            typer.echo(f"  Main video: {main_url}")

            # --- Shorts ---
            shorts = sorted(paths.shorts_dir.glob("short_*.mp4")) if paths.shorts_dir.exists() else []
            for i, short_path in enumerate(shorts, start=1):
                size_mb = round(short_path.stat().st_size / 1e6, 0)
                typer.echo(f"  Uploading short {i}/{len(shorts)} ({size_mb:.0f} MB)...")
                short_title = f"{base_title} #Shorts"[:100]
                short_meta = _video_metadata(short_title, description, tags + ["Shorts"], privacy)
                short_id = await _upload_one(short_path, short_meta, access_token, http)
                short_url = f"https://youtu.be/{short_id}"
                published[f"short_{i}"] = short_url
                typer.echo(f"  Short {i}: {short_url}")

        # Write publish.json for reference.
        paths.publish_json.write_text(json.dumps(published, indent=2))

    typer.echo(f"\n  Published {len(published)} video(s) as '{privacy}'.")
    if privacy != "public":
        typer.echo("  Set YOUTUBE_PRIVACY=public in .env before re-running to make them public.")
