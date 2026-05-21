from __future__ import annotations

import asyncio
import json
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
import structlog
from pydantic import BaseModel, ValidationError

from vidmaxx.services.llm.client import LLMClient
from vidmaxx.services.visualization.executor import ManimExecutor

log = structlog.get_logger(__name__)

MAX_ATTEMPTS = 5

_EXTRACT_SYSTEM = (
    "You are a data extraction assistant. "
    "Read the sentence and return only the requested Python variable assignments. "
    "No explanation. No markdown. No extra lines."
)

_DEBUG_SYSTEM = (
    "You are a Manim CE v0.19+ config debugger. "
    "The template code is fixed and correct. Only the config variables at the top can be wrong. "
    "Return only the corrected config variable lines."
)

# Per-type vision checks — structural, not aesthetic.
_VISION_PROMPTS: dict[str, str] = {
    "counting_number": """\
Sentence: "{sentence}"
Frame: 65% of animation duration.

Is there a large number clearly visible on screen?
Is the number non-zero and plausibly related to the sentence?
Is the background dark?

Return JSON: {{"numbers_visible": [...], "verdict": "PASS or FAIL", "reason": "one sentence"}}
FAIL if: main number is 0, 0%, $0, or screen is mostly blank.
PASS if: a non-zero value is clearly readable.""",

    "comparison": """\
Sentence: "{sentence}"
Frame: 65% of animation duration.

Are there two vertical bars of different heights?
Are labels visible on or near each bar?

Return JSON: {{"numbers_visible": [...], "verdict": "PASS or FAIL", "reason": "one sentence"}}
FAIL if: only one bar, both bars same height, or no bars at all.
PASS if: two bars with different heights and labels are visible.""",

    "flow": """\
Sentence: "{sentence}"
Frame: 65% of animation duration.

Are there rectangular boxes with text inside them?
Are there arrows connecting the boxes?

Return JSON: {{"numbers_visible": [...], "verdict": "PASS or FAIL", "reason": "one sentence"}}
FAIL if: no boxes, no arrows, or boxes have no readable text.
PASS if: labelled boxes connected by arrows are visible.""",

    "timeline": """\
Sentence: "{sentence}"
Frame: 65% of animation duration.

Is there a horizontal line visible?
Are there dots and date labels on the line?

Return JSON: {{"numbers_visible": [...], "verdict": "PASS or FAIL", "reason": "one sentence"}}
FAIL if: no horizontal line, no dots, or no date labels.
PASS if: a timeline with dates and at least one event label is visible.""",

    "shrink": """\
Sentence: "{sentence}"
Frame: 65% of animation duration.

Is there a horizontal bar that has been partially depleted?
Are there labels identifying what was removed?

Return JSON: {{"numbers_visible": [...], "verdict": "PASS or FAIL", "reason": "one sentence"}}
FAIL if: bar is still full width with no depletion, or no bar at all.
PASS if: a partially depleted bar with at least one deduction label is visible.""",
}

_VISION_FALLBACK = """\
Sentence: "{sentence}"
Frame: 65% of animation duration.

Does this frame contain meaningful animated data visualization content?

Return JSON: {{"numbers_visible": [...], "verdict": "PASS or FAIL", "reason": "one sentence"}}
FAIL if: screen is blank/black with no readable content.
PASS if: any relevant visual content is clearly visible."""


class BaseVizAgent(ABC):

    def __init__(self, llm: LLMClient, executor: ManimExecutor) -> None:
        self._llm = llm
        self._executor = executor

    @property
    @abstractmethod
    def viz_type(self) -> str: ...

    @property
    @abstractmethod
    def template(self) -> str:
        """Fixed Manim scene code. The CONFIG block prepended by the agent."""
        ...

    @property
    @abstractmethod
    def extract_prompt(self) -> str:
        """Prompt that asks the LLM to return only CONFIG variable assignments."""
        ...

    @property
    @abstractmethod
    def debug_prompt(self) -> str:
        """Prompt that asks the LLM to fix CONFIG variables given an error."""
        ...

    @property
    @abstractmethod
    def config_model(self) -> type[BaseModel]:
        """Pydantic model that validates the LLM-extracted JSON config."""
        ...

    @staticmethod
    def _config_to_py(model: BaseModel) -> str:
        """Serialize a validated Pydantic config to safe Python variable assignments."""
        lines = []
        for field_name, value in model.model_dump().items():
            lines.append(f"{field_name} = {repr(value)}")
        return "\n".join(lines)

    def _build_script(self, config: BaseModel) -> str:
        return self._config_to_py(config) + "\n\n" + self.template

    def _parse_config(self, raw: str) -> tuple[BaseModel | None, str]:
        """Parse and validate LLM JSON output. Returns (model, error_msg)."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            data = json.loads(cleaned)
            model = self.config_model(**data)
            return model, ""
        except json.JSONDecodeError as exc:
            return None, f"JSON parse error: {exc}"
        except ValidationError as exc:
            return None, f"Pydantic validation error: {exc}"

    async def run(
        self,
        sentence_text: str,
        duration_sec: float,
        output_path: Path,
        data_hints: dict | None = None,
    ) -> Path | None:
        data_hints = data_hints or {}
        raw_config: str | None = None
        error: str | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            log.info("viz_agent_attempt", agent=self.viz_type, attempt=attempt,
                     sentence=sentence_text[:60])

            if attempt == 1:
                raw_config = await self._extract_config(sentence_text, duration_sec, data_hints)
            else:
                raw_config = await self._fix_config(
                    sentence_text, raw_config, error, attempt, duration_sec
                )

            if not raw_config:
                continue

            parsed, parse_error = self._parse_config(raw_config)
            if parsed is None:
                error = f"Config validation failed — {parse_error}"
                log.warning("viz_config_invalid", agent=self.viz_type,
                            attempt=attempt, error=parse_error[:200])
                continue

            full_script = self._build_script(parsed)
            success, error, path = await asyncio.to_thread(
                self._executor.execute, full_script, output_path, duration_sec
            )

            if success:
                ok, vision_feedback = await self._vision_review(
                    path, sentence_text, duration_sec, self.viz_type, attempt
                )
                if ok:
                    log.info("viz_agent_success", agent=self.viz_type, attempt=attempt)
                    return path
                error = (
                    f"VISION_REVIEW_FAILURE — code ran without error but frame was visually wrong.\n"
                    f"Reason: {vision_feedback}\n"
                    f"This is a CONFIG logic error, not a syntax error.\n"
                    f"Check: values match the sentence, non-zero targets, correct prefix/suffix."
                )
                path.unlink(missing_ok=True)
                log.warning("viz_agent_vision_rejected", agent=self.viz_type,
                            attempt=attempt, feedback=vision_feedback[:200])

            log.warning("viz_agent_attempt_failed", agent=self.viz_type,
                        attempt=attempt, error=(error or "")[:200])

        log.error("viz_agent_exhausted", agent=self.viz_type, sentence=sentence_text[:60])
        return None

    async def _vision_review(
        self,
        video_path: Path,
        sentence_text: str,
        duration_sec: float,
        viz_type: str,
        attempt: int,
    ) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "review_frame.png"
            extracted = await asyncio.to_thread(
                self._executor.extract_frame, video_path, duration_sec * 0.65, frame
            )
            if not extracted:
                log.warning("viz_review_frame_extract_failed",
                            agent=viz_type, attempt=attempt, path=str(video_path),
                            note="failing open")
                return True, ""

            template = _VISION_PROMPTS.get(viz_type, _VISION_FALLBACK)
            prompt = template.format(sentence=sentence_text)

            try:
                raw = await self._llm.complete_with_image(
                    model="vertex_ai/gemini-2.5-flash",
                    system="You are a visualization quality reviewer. Return only JSON.",
                    prompt=prompt,
                    image_path=frame,
                    max_tokens=256,
                )
            except Exception as exc:
                log.warning("viz_review_llm_error",
                            agent=viz_type, attempt=attempt, error=str(exc),
                            note="failing open")
                return True, ""

        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            result = json.loads(cleaned)
            verdict = result.get("verdict", "PASS").upper()
            reason = result.get("reason", "")
            numbers = result.get("numbers_visible", [])
            if verdict == "FAIL":
                return False, f"{reason} Numbers visible: {numbers}"
        except Exception as exc:
            log.warning("viz_review_parse_error",
                        agent=viz_type, attempt=attempt, error=str(exc),
                        raw=raw[:200], note="failing open")

        return True, ""

    async def _extract_config(
        self, sentence_text: str, duration_sec: float, data_hints: dict
    ) -> str | None:
        prompt = self.extract_prompt.format(
            sentence_text=sentence_text,
            duration_sec=duration_sec,
            **data_hints,
        )
        return await self._llm.complete(
            model="vertex_ai/gemini-2.5-flash",
            system=_EXTRACT_SYSTEM,
            prompt=prompt,
            max_tokens=512,
        )

    async def _fix_config(
        self,
        sentence_text: str,
        broken_config: str | None,
        error: str | None,
        attempt: int,
        duration_sec: float,
    ) -> str | None:
        prompt = self.debug_prompt.format(
            sentence_text=sentence_text,
            broken_config=broken_config or "",
            error_output=error or "No error captured",
            attempt=attempt,
            max_attempts=MAX_ATTEMPTS,
            duration_sec=duration_sec,
        )
        return await self._llm.complete(
            model="vertex_ai/gemini-2.5-flash",
            system=_DEBUG_SYSTEM,
            prompt=prompt,
            max_tokens=512,
        )
