from __future__ import annotations

from dataclasses import dataclass, field

from .models import TranscriptEvent


@dataclass(slots=True)
class EventStreamValidator:
    revisions: dict[str, int] = field(default_factory=dict)
    ended: bool = False
    last_commit_start: float = 0.0
    max_end: float = 0.0

    def accept(self, event: TranscriptEvent) -> None:
        if self.ended:
            raise ValueError("event received after session end")
        previous = self.revisions.get(event.segment_id, 0)
        if event.revision <= previous:
            raise ValueError(f"non-increasing revision for {event.segment_id}")
        if event.kind == "replace" and previous == 0:
            raise ValueError(f"replace references unknown segment {event.segment_id}")
        if event.kind == "commit" and previous != 0:
            raise ValueError(f"commit reuses segment {event.segment_id}")
        if event.kind == "commit" and event.start < self.last_commit_start:
            raise ValueError("commit events are not chronological")
        if event.kind == "end":
            if event.start < self.max_end:
                raise ValueError("end event precedes transcript content")
            self.ended = True
        if event.kind == "commit":
            self.last_commit_start = event.start
        self.max_end = max(self.max_end, event.end)
        self.revisions[event.segment_id] = event.revision
