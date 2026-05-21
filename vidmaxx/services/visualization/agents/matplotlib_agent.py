from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import structlog

from vidmaxx.services.llm.client import LLMClient

log = structlog.get_logger(__name__)

MAX_ATTEMPTS = 3

_GENERATE_PROMPT = """
Write a complete Python matplotlib script to visualize this data.

Script line: "{sentence_text}"
Visualization type: "{viz_type}"
Output path: "{output_path}"

STYLE:
- Figure size: (19.2, 10.8) at 100 dpi = 1920x1080
- Background: #000000 black
- Text: #FFFFFF white
- Accent: #E63946 red
- No titles, no borders, minimal axes
- Save as PNG to exactly: {output_path}
- DPI: 100

Use plt.style.use('dark_background') as base.
Remove all spines. Remove tick marks. Keep only essential labels.
The visualization must be clean and cinematic.

Extract the actual numbers and entities from the script line.
Do not use placeholder data.

Return only Python code. No markdown.
"""

_DEBUG_PROMPT = """
Fix this broken matplotlib script.

Script line: "{sentence_text}"
Attempt: {attempt}/{max_attempts}

BROKEN CODE:
{broken_code}

ERROR:
{error_output}

Fix only the error. Return only Python code.
"""


class MatplotlibAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(
        self,
        sentence_text: str,
        viz_type: str,
        output_path: Path,
    ) -> Path | None:
        code: str | None = None
        error: str | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt == 1:
                code = await self._llm.complete(
                    model="vertex_ai/gemini-2.5-flash",
                    system="Python matplotlib expert. Return only code.",
                    prompt=_GENERATE_PROMPT.format(
                        sentence_text=sentence_text,
                        viz_type=viz_type,
                        output_path=str(output_path),
                    ),
                    max_tokens=2048,
                )
            else:
                code = await self._llm.complete(
                    model="vertex_ai/gemini-2.5-flash",
                    system="Python matplotlib debugger. Fix only the error.",
                    prompt=_DEBUG_PROMPT.format(
                        sentence_text=sentence_text,
                        broken_code=code or "",
                        error_output=error or "unknown",
                        attempt=attempt,
                        max_attempts=MAX_ATTEMPTS,
                    ),
                    max_tokens=2048,
                )

            if not code:
                continue

            success, error = await asyncio.to_thread(self._execute, code, output_path)
            if success:
                log.info("matplotlib_agent_success", attempt=attempt)
                return output_path

            log.warning("matplotlib_agent_failed", attempt=attempt, error=(error or "")[:200])

        return None

    def _execute(self, code: str, output_path: Path) -> tuple[bool, str | None]:
        clean = code.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            for part in parts:
                s = part.lstrip("python").strip()
                if "import matplotlib" in s or "import plt" in s or "plt." in s:
                    clean = s
                    break

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(clean)
            script = Path(f.name)

        try:
            result = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return False, result.stderr[-1000:]
            if not output_path.exists():
                return False, "Script ran but no output file created"
            return True, None
        except Exception as exc:
            return False, str(exc)
        finally:
            script.unlink(missing_ok=True)
