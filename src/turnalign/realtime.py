from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from threading import Event

from .audio import AudioTimeline
from .models import AudioChunk, TranscriptEvent
from .plugins import AsrBackend, OnlineDiarizationBackend
from .session import transcribe_events


@dataclass(slots=True)
class RealtimePipeline:
    """Low-latency first pass with stable event identities."""

    backend: AsrBackend
    vad_threshold: float = 0.012
    silence_seconds: float = 0.7
    max_utterance_seconds: float = 20.0
    partial_seconds: float = 2.0
    online_diarizer: OnlineDiarizationBackend | None = None

    def events(
        self,
        chunks: Iterable[AudioChunk],
        *,
        recorded_timeline: AudioTimeline | None = None,
        cancel_event: Event | None = None,
        emit_end: bool = True,
        close_backend: bool = True,
    ) -> Iterator[TranscriptEvent]:
        yield from transcribe_events(
            chunks,
            self.backend,
            live=True,
            vad_threshold=self.vad_threshold,
            silence_seconds=self.silence_seconds,
            max_utterance_seconds=self.max_utterance_seconds,
            partial_seconds=self.partial_seconds,
            recorded_timeline=recorded_timeline,
            cancel_event=cancel_event,
            close_backend=close_backend,
            emit_end=emit_end,
            online_diarizer=self.online_diarizer,
        )
