"""
LLM client backed by litellm.acompletion.

Responsibilities:
  - Cache-first: every call checks PipelineCache before hitting the API.
  - Retry on rate limits (tenacity, exponential backoff).
  - Three call modes: plain text, JSON extraction, extended thinking.
  - No prompt logic lives here — callers pass (system, prompt) strings.

Model routing is done by the caller via the `model` argument:
  - "vertex_ai/gemini-2.5-pro"   → SCRIPT_ARCHITECT_MODEL, SCRIPT_CHAPTER_MODEL, RESEARCH_MODEL
  - "vertex_ai/gemini-2.5-flash" → SCRIPT_DECOMPOSE_MODEL
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vidmaxx.state.cache import PipelineCache

log = structlog.get_logger(__name__)

litellm.suppress_debug_info = True  # keep logs clean

# Strips ```json ... ``` or ``` ... ``` wrappers that models sometimes emit
_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    start = min(
        text.find("{") if "{" in text else len(text),
        text.find("[") if "[" in text else len(text),
    )
    end = max(text.rfind("}"), text.rfind("]")) + 1
    if start < end:
        return text[start:end]
    return text.strip()


def _text_from_response(response) -> str:
    """Extract plain text from a litellm response object.

    Handles both string content (standard) and list-of-blocks content
    (some providers return thinking + text as separate blocks).
    """
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    return content or ""


_JSON_INSTRUCTION = (
    "\n\nCRITICAL: Return ONLY valid, complete JSON. No markdown, no explanation. "
    "If you cannot fit all items, return fewer items but ALWAYS properly close every "
    "bracket and brace. Incomplete JSON is unacceptable."
)


class LLMClient:
    def __init__(self, api_key: str, cache: PipelineCache) -> None:
        self._api_key = api_key
        self._cache = cache

    # ------------------------------------------------------------------
    # Internal: raw API call with retry — no cache logic here
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(litellm.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _call(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "api_key": self._api_key,
        }
        if thinking_budget is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        log.debug("llm_call", model=model, prompt_chars=len(prompt))
        response = await litellm.acompletion(**kwargs)
        return _text_from_response(response)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _repair_json(self, text: str) -> str:
        last_valid = max(text.rfind("}"), text.rfind("]"))
        if last_valid == -1:
            return text
        text = text[: last_valid + 1]
        opens: list[str] = []
        for ch in text:
            if ch in "{[":
                opens.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if opens:
                    opens.pop()
        return text + "".join(reversed(opens))

    # ------------------------------------------------------------------
    # Public API — signatures 
    # ------------------------------------------------------------------

    async def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int = 8192,
    ) -> str:
        cached = self._cache.get_llm(system, prompt)
        if cached is not None:
            log.debug("llm_cache_hit", model=model)
            return cached

        result = await self._call(
            model=model, system=system, prompt=prompt, max_tokens=max_tokens
        )
        self._cache.set_llm(system, prompt, result)
        return result

    async def complete_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int = 8192,
    ) -> dict | list:
        augmented_prompt = prompt + _JSON_INSTRUCTION
        raw = await self.complete(
            model=model, system=system, prompt=augmented_prompt, max_tokens=max_tokens
        )
        extracted = _extract_json(raw)
        log.debug("llm_json_raw", raw=raw[:2000], extracted=extracted[:2000])

        # Layer 1: direct parse
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        # Layer 2: structural repair — zero extra API calls
        try:
            repaired = self._repair_json(extracted)
            log.debug("llm_json_repaired", repaired=repaired[:500])
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Layer 3: one retry with the same prompt
        log.warning("llm_json_retry", model=model)
        raw = await self._call(
            model=model, system=system, prompt=augmented_prompt, max_tokens=max_tokens
        )
        return json.loads(_extract_json(raw))

    async def complete_with_image(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        image_path: Path,
        max_tokens: int = 512,
    ) -> str:
        import base64
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        log.debug("llm_vision_call", model=model)
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            api_key=self._api_key,
        )
        return _text_from_response(response)

    async def complete_with_thinking(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        thinking_budget: int,
        max_tokens: int,
    ) -> str:
        # Not cached — see comment in complete_json_with_thinking below.
        return await self._call(
            model=model,
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )

    async def complete_json_with_thinking(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        thinking_budget: int,
        max_tokens: int,
    ) -> dict | list:
        # Extended thinking output is not cached here — the architect stage
        # caches the parsed Outline by fact_sheet hash, which is cheaper.
        raw = await self.complete_with_thinking(
            model=model,
            system=system,
            prompt=prompt,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
        )
        return json.loads(_extract_json(raw))
