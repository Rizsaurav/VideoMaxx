"""
Stage 3 — Asset acquisition

Reads:   paths.script  (script.json)
Writes:  paths.assets_dir/{sentence_id}.{ext}  (one file per sentence)
         paths.credits_json                      (credits.json)
Advances: ASSETS → TTS

All sentences are fetched in parallel, bounded by ASSET_FETCH_CONCURRENCY
so we don't flood the source APIs. CLIPRanker is loaded once for the whole
stage and unloaded on exit.

credits.json is a flat list of Asset records (JSON). Only sentences where
attribution_required=True are strictly needed for compliance, but we log
all of them for provenance.
"""

from __future__ import annotations

import asyncio
import json

import structlog

from vidmaxx.config.constants import (
    ASSET_FETCH_CONCURRENCY,
    VISUAL_REGISTER_DESCRIPTIONS,
    VISUAL_REGISTER_QUERIES,
)
from vidmaxx.config.settings import Settings
from vidmaxx.models.asset import Asset, AssetType
from vidmaxx.models.project import PipelineStage
from vidmaxx.models.script import Script
from vidmaxx.services.assets.fetcher import AssetFetcher
from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.vision.clip_ranker import CLIPRanker
from vidmaxx.services.visualization.orchestrator import VizOrchestrator
from vidmaxx.services.visualization.quote_card import (
    generate_quote_card,
    select_quote_card_sentences,
)
from vidmaxx.state.cache import PipelineCache
from vidmaxx.state.project_state import ProjectStateManager

log = structlog.get_logger(__name__)


async def run(
    slug: str,
    state_mgr: ProjectStateManager,
    cache: PipelineCache,
    settings: Settings,
) -> list[Asset]:
    paths = state_mgr.paths(slug)
    script = Script.model_validate_json(paths.script.read_text())
    sentences = script.sentences
    log.info("stage_assets_start", slug=slug, sentences=len(sentences))

    with state_mgr.run_stage(slug, PipelineStage.ASSETS) as paths:
        paths.assets_dir.mkdir(parents=True, exist_ok=True)

        # Split sentences: numerical ones go to viz agents, rest to stock footage
        viz_sentences = [s for s in sentences if s.visualization_type != "none"]
        stock_sentences = [s for s in sentences if s.visualization_type == "none"]

        # High-exaggeration "none" sentences get full-screen quote cards instead of stock.
        quote_ids = select_quote_card_sentences(stock_sentences)
        quote_sentences  = [s for s in stock_sentences if s.id in quote_ids]
        stock_sentences  = [s for s in stock_sentences if s.id not in quote_ids]
        log.info(
            "stage_assets_split",
            viz=len(viz_sentences),
            quote_cards=len(quote_sentences),
            stock=len(stock_sentences),
        )

        # Generate quote cards synchronously (fast — just Pillow PNG writes)
        quote_assets: list[Asset] = []
        for s in quote_sentences:
            try:
                png_path = generate_quote_card(s.id, s.text, paths.assets_dir)
                quote_assets.append(_asset_from_existing(s.id, png_path))
            except Exception:
                log.exception("quote_card_failed", id=s.id)

        ranker = CLIPRanker(device=settings.tts_device, cache=cache)
        fetcher = AssetFetcher(
            pexels_key=settings.pexels_api_key,
            pixabay_key=settings.pixabay_api_key,
            ranker=ranker,
        )

        try:
            if viz_sentences:
                viz_orchestrator = VizOrchestrator(
                    LLMClient(api_key=settings.google_api_key, cache=cache)
                )
                viz_task = asyncio.create_task(
                    viz_orchestrator.generate_all(viz_sentences, paths.assets_dir)
                )
            else:
                viz_task = None

            stock_assets = await _fetch_all(fetcher, stock_sentences, paths.assets_dir, settings)
        finally:
            ranker.unload()

        viz_results: dict = await viz_task if viz_task else {}

        # Merge: viz and quote cards take priority over stock
        asset_map: dict[str, Asset] = {a.sentence_id: a for a in stock_assets if a}
        for a in quote_assets:
            asset_map[a.sentence_id] = a
        for sid, asset in viz_results.items():
            if asset:
                asset_map[sid] = asset

        assets = list(asset_map.values())

        _write_credits(assets, paths.credits_json)
        log.info(
            "stage_assets_done",
            slug=slug,
            viz_generated=len([a for a in viz_results.values() if a]),
            quote_cards=len(quote_assets),
            stock_fetched=len([a for a in stock_assets if a]),
            total=len(assets),
            missing=len(sentences) - len(assets),
        )

    return assets


_ASSETS_SIZE_CAP_BYTES = 3 * 1024 ** 3  # 3 GB


def _dir_size(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


async def _fetch_all(fetcher, sentences, assets_dir, settings) -> list[Asset]:
    semaphore = asyncio.Semaphore(settings.max_parallel_assets)
    results: list[Asset | None] = [None] * len(sentences)
    cap_hit = False

    async def fetch_one(i, sentence):
        nonlocal cap_hit
        async with semaphore:
            # Always load existing files — cap only blocks NEW downloads
            existing = list(assets_dir.glob(f"{sentence.id}.*"))
            if existing:
                log.debug("asset_cache_hit", id=sentence.id, path=str(existing[0]))
                results[i] = _asset_from_existing(sentence.id, existing[0])
                return

            if cap_hit:
                return

            if _dir_size(assets_dir) >= _ASSETS_SIZE_CAP_BYTES:
                cap_hit = True
                log.warning("assets_size_cap_hit", cap_gb=2, remaining=len(sentences) - i)
                return

            register = getattr(sentence, "emotional_register", "systemic_flow")
            register_queries = VISUAL_REGISTER_QUERIES.get(register, [])
            clip_query = VISUAL_REGISTER_DESCRIPTIONS.get(register)

            if register_queries:
                # Deterministic rotation — same sentence always gets same hint
                parts = sentence.id.split("_s")
                sentence_idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else i
                hint = register_queries[sentence_idx % len(register_queries)]
                sentence = sentence.model_copy(
                    update={"visual_query": f"{hint} {sentence.visual_query}"}
                )
            # Every 5th sentence prefers an image (20%) — images are ~40x smaller
            # than video, so 20% images saves ~500MB and adds visual variety.
            prefer_type = AssetType.IMAGE if i % 5 == 0 else AssetType.VIDEO
            asset = await fetcher.fetch(sentence, assets_dir, clip_query=clip_query, prefer_type=prefer_type)
            results[i] = asset
            if (i + 1) % 10 == 0:
                done = sum(1 for r in results if r is not None)
                log.info("assets_progress", done=done, total=len(sentences))

    await asyncio.gather(*[fetch_one(i, s) for i, s in enumerate(sentences)])

    missing_ids = [s.id for i, s in enumerate(sentences) if results[i] is None]
    if missing_ids:
        log.warning(
            "assets_missing",
            count=len(missing_ids),
            sentence_ids=missing_ids,
            note="These sentences will render as black frames.",
        )

    return [r for r in results if r is not None]


def _asset_from_existing(sentence_id: str, path) -> Asset:
    from vidmaxx.models.asset import AssetSource, AssetType
    ext = path.suffix.lower()
    asset_type = AssetType.VIDEO if ext in {".mp4", ".mov", ".webm"} else AssetType.IMAGE
    return Asset(
        sentence_id=sentence_id,
        source=AssetSource.PEXELS,  # source unknown on cache hit, placeholder
        asset_type=asset_type,
        local_path=path,
        remote_url="",
        page_url="",
        license="unknown",
        attribution_required=False,
        attribution_text="",
        clip_score=0.0,
        width=0,
        height=0,
        duration_sec=0.0,
    )


def _write_credits(assets: list[Asset], out_path) -> None:
    credits = [
        {
            "sentence_id": a.sentence_id,
            "source": a.source,
            "license": a.license,
            "attribution_required": a.attribution_required,
            "attribution_text": a.attribution_text,
            "page_url": a.page_url,
        }
        for a in assets
    ]
    out_path.write_text(json.dumps(credits, indent=2))
