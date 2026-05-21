"""
CLIP-based asset ranker.

Loads ViT-B-32 once, keeps it warm across the entire asset stage,
then explicitly unloads. Import of open_clip is deferred to _load()
so the rest of the codebase doesn't require it at import time.

Ranking flow per sentence:
  1. Encode the visual_query as a text embedding.
  2. For each candidate: check diskcache by clip_url; fetch + encode if miss.
  3. Cosine similarity between text and image embeddings → clip_score.
  4. Return candidates sorted by clip_score descending.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import numpy as np
import structlog
import torch

from vidmaxx.config.constants import CLIP_MODEL_NAME, CLIP_PRETRAINED
from vidmaxx.models.asset import AssetCandidate
from vidmaxx.state.cache import PipelineCache

log = structlog.get_logger(__name__)

_FETCH_TIMEOUT = httpx.Timeout(10.0)


class CLIPRanker:
    def __init__(self, device: str = "mps", cache: PipelineCache | None = None) -> None:
        self._device = device
        self._cache = cache
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is not installed. Run: uv pip install open-clip-torch"
            ) from exc

        effective_device = (
            self._device
            if self._device == "mps" and torch.backends.mps.is_available()
            else "cpu"
        )

        log.info("clip_loading", model=CLIP_MODEL_NAME, device=effective_device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model = model.to(effective_device).eval()

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self._effective_device = effective_device
        log.info("clip_ready")

    def unload(self) -> None:
        if self._model is not None:
            del self._model, self._preprocess, self._tokenizer
            self._model = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            log.info("clip_unloaded")

    def __enter__(self) -> "CLIPRanker":
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Text embedding
    # ------------------------------------------------------------------

    def _encode_text(self, query: str) -> np.ndarray:
        tokens = self._tokenizer([query]).to(self._effective_device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()[0]

    # ------------------------------------------------------------------
    # Image embedding
    # ------------------------------------------------------------------

    def _encode_image_bytes(self, data: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self._effective_device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()[0]

    async def _get_image_embedding(
        self, url: str, client: httpx.AsyncClient
    ) -> np.ndarray | None:
        if self._cache is not None:
            cached = self._cache.get_clip(url)
            if cached is not None:
                return np.array(cached, dtype=np.float32)

        try:
            r = await client.get(url, timeout=_FETCH_TIMEOUT)
            r.raise_for_status()
            embedding = self._encode_image_bytes(r.content)
        except Exception as exc:
            log.warning("clip_image_fetch_failed", url=url, error=str(exc))
            return None

        if self._cache is not None:
            self._cache.set_clip(url, embedding.tolist())

        return embedding

    # ------------------------------------------------------------------
    # Public ranking API
    # ------------------------------------------------------------------

    async def rank(
        self,
        query: str,
        candidates: list[AssetCandidate],
    ) -> list[AssetCandidate]:
        """
        Fills in clip_score on each candidate and returns them sorted
        highest-score-first. Candidates with failed image fetches get score 0.
        """
        if not candidates:
            return []

        text_emb = self._encode_text(query)

        scored: list[AssetCandidate] = []
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for candidate in candidates:
                img_emb = await self._get_image_embedding(candidate.clip_url, client)
                if img_emb is not None:
                    score = float(np.dot(text_emb, img_emb))
                    # Clamp to [0, 1] — cosine similarity for normalised vectors
                    # is in [-1, 1]; negative scores mean anti-correlated content
                    candidate = candidate.model_copy(
                        update={"clip_score": max(0.0, min(1.0, score))}
                    )
                scored.append(candidate)

        return sorted(scored, key=lambda c: c.clip_score, reverse=True)
