"""
Cloudflare R2 storage client (S3-compatible).

Key layout in the bucket:
  projects/{slug}/assets/{filename}   — ranked assets (3 GB cap across all projects)
  projects/{slug}/final.mp4           — rendered video

The 3 GB asset cap is enforced per upload call: if adding the next file
would push total asset bytes over the limit, it is skipped and logged.
The cap is checked by summing sizes of objects under projects/*/assets/.

Upload is best-effort — a failure logs a warning but never crashes the pipeline.
Local files are NOT deleted after upload; the render stage still needs them.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError

log = structlog.get_logger(__name__)

_ASSET_CAP_BYTES = 3 * 1024 ** 3   # 3 GB


class R2Client:
    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
    ) -> None:
        self._bucket = bucket
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        log.info("r2_ready", bucket=bucket)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_asset(self, local_path: Path, slug: str) -> str | None:
        """
        Upload a single asset file to projects/{slug}/assets/{filename}.
        Skips the upload if it would push total asset usage over 3 GB.
        Returns the R2 object key on success, None on skip/failure.
        """
        size = local_path.stat().st_size
        used = self._asset_bytes_used()
        if used + size > _ASSET_CAP_BYTES:
            log.warning(
                "r2_asset_cap_reached",
                cap_gb=3,
                used_gb=round(used / 1024 ** 3, 2),
                file=local_path.name,
                size_mb=round(size / 1024 ** 2, 1),
            )
            return None

        key = f"projects/{slug}/assets/{local_path.name}"
        return self._upload(local_path, key)

    def upload_final(self, local_path: Path, slug: str) -> str | None:
        """Upload the final rendered video to projects/{slug}/final.mp4."""
        key = f"projects/{slug}/final.mp4"
        return self._upload(local_path, key)

    def asset_bytes_used(self) -> int:
        return self._asset_bytes_used()

    def public_url(self, key: str) -> str:
        """Returns the R2 public URL for a key (requires public bucket or dev URL)."""
        return f"{self._s3.meta.endpoint_url}/{self._bucket}/{key}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upload(self, local_path: Path, key: str) -> str | None:
        content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        size_mb = round(local_path.stat().st_size / 1024 ** 2, 1)
        try:
            self._s3.upload_file(
                str(local_path),
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            log.info("r2_uploaded", key=key, size_mb=size_mb)
            return key
        except (BotoCoreError, ClientError) as exc:
            log.warning("r2_upload_failed", key=key, error=str(exc))
            return None

    def _asset_bytes_used(self) -> int:
        total = 0
        paginator = self._s3.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self._bucket, Prefix="projects/"):
                for obj in page.get("Contents", []):
                    if "/assets/" in obj["Key"]:
                        total += obj["Size"]
        except (BotoCoreError, ClientError) as exc:
            log.warning("r2_usage_check_failed", error=str(exc))
        return total
