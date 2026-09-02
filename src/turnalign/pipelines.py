from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .audio import AudioTimeline
from .models import AudioChunk, TranscriptEvent
from .offline import OfflineRefinementPipeline
from .realtime import RealtimePipeline
from .resources import close_resources

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TwoPassPipeline:
    """Record once, emit a realtime draft, then refine the same segments."""

    realtime: RealtimePipeline
    offline: OfflineRefinementPipeline

    def events(self, chunks: Iterable[AudioChunk]) -> Iterator[TranscriptEvent]:
        commits: list[TranscriptEvent] = []
        refinement_error: BaseException | None = None
        refinement_started = False
        try:
            with AudioTimeline() as timeline:
                for event in self.realtime.events(
                    chunks,
                    recorded_timeline=timeline,
                    emit_end=False,
                ):
                    if event.kind == "commit":
                        commits.append(event)
                    yield event
                try:
                    refinement_started = True
                    yield from self.offline.refine(timeline, commits)
                except Exception as error:  # noqa: BLE001 - refinement is fail-open by design
                    refinement_error = error
                yield TranscriptEvent(
                    kind="end",
                    segment_id="session",
                    revision=1,
                    start=timeline.end,
                    end=timeline.end,
                    metadata={
                        "segments": len(commits),
                        "passes": 2,
                        "recording_storage": "disk-timeline",
                        "refinement_status": (
                            "failed" if refinement_error is not None else "complete"
                        ),
                        "refinement_error_type": (
                            type(refinement_error).__name__
                            if refinement_error is not None else None
                        ),
                    },
                )
        finally:
            if not refinement_started:
                close_resources(
                    (self.offline.backend, self.offline.aligner, self.offline.diarizer),
                    logger=LOGGER,
                    reason="unused refinement shutdown",
                )
