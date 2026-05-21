"""
VizOrchestrator — routes sentences to Manim agents, runs in parallel,
falls back to matplotlib then stock footage if all attempts fail.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from vidmaxx.models.asset import Asset, AssetSource, AssetType
from vidmaxx.models.sentence import Sentence
from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.visualization.agents.comparison_agent import ComparisonAgent
from vidmaxx.services.visualization.agents.counting_agent import CountingAgent
from vidmaxx.services.visualization.agents.flow_agent import FlowAgent
from vidmaxx.services.visualization.agents.matplotlib_agent import MatplotlibAgent
from vidmaxx.services.visualization.agents.shrink_agent import ShrinkAgent
from vidmaxx.services.visualization.agents.timeline_agent import TimelineAgent
from vidmaxx.services.visualization.executor import ManimExecutor

log = structlog.get_logger(__name__)


class VizOrchestrator:

    def __init__(self, llm: LLMClient) -> None:
        executor = ManimExecutor()
        self._agents = {
            "counting_number": CountingAgent(llm, executor),
            "comparison":      ComparisonAgent(llm, executor),
            "flow":            FlowAgent(llm, executor),
            "timeline":        TimelineAgent(llm, executor),
            "shrink":          ShrinkAgent(llm, executor),
        }
        self._matplotlib = MatplotlibAgent(llm)

    async def generate_all(
        self,
        sentences: list[Sentence],
        assets_dir: Path,
        concurrency: int = 4,
    ) -> dict[str, Asset | None]:
        sem = asyncio.Semaphore(concurrency)
        results: dict[str, Asset | None] = {}

        async def process(sentence: Sentence) -> None:
            async with sem:
                results[sentence.id] = await self._process(sentence, assets_dir)

        await asyncio.gather(*[process(s) for s in sentences])
        return results

    async def _process(self, sentence: Sentence, assets_dir: Path) -> Asset | None:
        viz_type = sentence.visualization_type

        output_path = assets_dir / f"{sentence.id}_viz.mp4"
        if output_path.exists():
            log.debug("viz_cache_hit", id=sentence.id)
            return _make_asset(sentence.id, output_path)

        agent = self._agents.get(viz_type)
        if not agent:
            log.warning("viz_unknown_type", id=sentence.id, viz_type=viz_type)
            return None

        result = await agent.run(
            sentence_text=sentence.text,
            duration_sec=sentence.expected_duration_sec,
            output_path=output_path,
        )
        if result:
            return _make_asset(sentence.id, result)

        # Fallback 1: matplotlib static image
        log.warning("viz_manim_exhausted_trying_matplotlib", id=sentence.id, viz_type=viz_type)
        static_path = assets_dir / f"{sentence.id}_viz.png"
        static = await self._matplotlib.run(
            sentence_text=sentence.text,
            viz_type=viz_type,
            output_path=static_path,
        )
        if static:
            return _make_asset(sentence.id, static)

        # Fallback 2: None → s03 pool-cycles stock footage
        log.error("viz_all_failed_using_stock", id=sentence.id, viz_type=viz_type)
        return None


def _make_asset(sentence_id: str, path: Path) -> Asset:
    is_video = path.suffix.lower() == ".mp4"
    return Asset(
        sentence_id=sentence_id,
        source=AssetSource.GENERATED,
        asset_type=AssetType.VIDEO if is_video else AssetType.IMAGE,
        local_path=path,
        remote_url="",
        page_url="",
        license="generated",
        attribution_required=False,
        attribution_text="",
        clip_score=1.0,
        width=1920,
        height=1080,
        duration_sec=0.0,
    )
