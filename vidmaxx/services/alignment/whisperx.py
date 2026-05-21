"""
Alignment service — duration-based, no model required.

Word timestamps are derived directly from per-sentence WAV durations and
character positions. No transcription, no GPU, instant execution.

Words are distributed proportionally within the speech portion of each
WAV (full duration minus the baked-in pause_before/pause_after padding).

Output is a list[CaptionSegment], one per sentence, each with
CaptionWord entries carrying start/end times relative to the full
narration and is_emphasis=True for words in sentence.emphasis_words.
"""

from pathlib import Path

import soundfile as sf
import structlog

from vidmaxx.models.script import Script
from vidmaxx.models.timeline import CaptionSegment, CaptionWord
from vidmaxx.utils.timing import sentence_start_times

log = structlog.get_logger(__name__)


class AlignmentService:
    def __init__(self, **_kwargs) -> None:
        log.info("alignment_ready", mode="duration_based")

    def unload(self) -> None:
        pass

    def __enter__(self) -> "AlignmentService":
        return self

    def __exit__(self, *_) -> None:
        pass

    def run(
        self,
        narration_path: Path,
        script: Script,
        audio_dir: Path,
    ) -> list[CaptionSegment]:
        sentences = script.sentences
        log.info("alignment_start", sentences=len(sentences))

        durations = [
            sf.info(str(audio_dir / f"{s.id}.wav")).duration
            for s in sentences
        ]
        starts = sentence_start_times(durations)

        result = []
        for i, sentence in enumerate(sentences):
            sentence_start = starts[i]
            full_duration = durations[i]

            # Exclude baked-in silence so words land in the speech window
            pause_before = sentence.pause_before_ms / 1000.0
            pause_after = sentence.pause_after_ms / 1000.0
            speech_start = sentence_start + pause_before
            speech_duration = max(full_duration - pause_before - pause_after, 0.05)

            emphasis_set = {w.lower() for w in sentence.emphasis_words}
            words = sentence.text.split()
            total_chars = max(sum(len(w) for w in words), 1)

            caption_words: list[CaptionWord] = []
            char_cursor = 0
            for word in words:
                word_start = speech_start + (char_cursor / total_chars) * speech_duration
                word_end = speech_start + ((char_cursor + len(word)) / total_chars) * speech_duration
                caption_words.append(
                    CaptionWord(
                        word=word,
                        start_sec=round(word_start, 3),
                        end_sec=round(word_end, 3),
                        is_emphasis=word.lower().rstrip(".,!?") in emphasis_set,
                    )
                )
                char_cursor += len(word)

            result.append(CaptionSegment(sentence_id=sentence.id, words=caption_words))

        log.info("alignment_done", sentences=len(result))
        return result
