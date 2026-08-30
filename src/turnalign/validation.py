from __future__ import annotations

from dataclasses import dataclass, field

from .models import TranscriptEvent

_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"partial", "commit"}),
    "partial": frozenset({"partial", "commit"}),
    "commit": frozenset({"replace"}),
    "replace": frozenset({"replace"}),
}


@dataclass(slots=True)
class EventStreamValidator:
    revisions: dict[str, int] = field(default_factory=dict)
    states: dict[str, str] = field(default_factory=dict)
    ended: bool = False
    last_commit_start: float = 0.0
    max_end: float = 0.0
    protocol_version: int | None = None
    session_id: str | None = None
    last_sequence: int | None = None
    last_acknowledged_sequence: int | None = None

    def accept(self, event: TranscriptEvent) -> None:
        if self.ended:
            raise ValueError("event received after session end")
        if self.protocol_version is None:
            self.protocol_version = event.protocol_version
        elif event.protocol_version != self.protocol_version:
            raise ValueError("protocol version changed during session")
        if event.session_id is not None:
            if self.session_id is None:
                self.session_id = event.session_id
            elif event.session_id != self.session_id:
                raise ValueError("session_id changed during session")
        if event.sequence is not None:
            if self.last_sequence is not None and event.sequence != self.last_sequence + 1:
                raise ValueError("event sequence is not contiguous")
            self.last_sequence = event.sequence
        if event.acknowledged_sequence is not None:
            if (
                self.last_acknowledged_sequence is not None
                and event.acknowledged_sequence < self.last_acknowledged_sequence
            ):
                raise ValueError("acknowledged sequence moved backwards")
            self.last_acknowledged_sequence = event.acknowledged_sequence
        if event.kind == "end":
            open_segments = [
                segment_id
                for segment_id, state in self.states.items()
                if state == "partial"
            ]
            if open_segments:
                raise ValueError(
                    f"end received with open partial segment {open_segments[0]}"
                )
            if event.start < self.max_end:
                raise ValueError("end event precedes transcript content")
            self.ended = True
            self.revisions[event.segment_id] = event.revision
            return

        previous = self.revisions.get(event.segment_id, 0)
        if event.revision <= previous:
            raise ValueError(f"non-increasing revision for {event.segment_id}")

        if event.kind == "speaker_merge":
            source = event.metadata.get("from_speaker")
            target = event.metadata.get("to_speaker")
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(target, str)
                or not target
                or source == target
            ):
                raise ValueError("speaker_merge requires distinct from_speaker and to_speaker")
            self.revisions[event.segment_id] = event.revision
            self.max_end = max(self.max_end, event.end)
            return

        previous_state = self.states.get(event.segment_id)
        if event.kind not in _TRANSITIONS.get(previous_state, frozenset()):
            if event.kind == "replace" and previous_state is None:
                raise ValueError(f"replace references unknown segment {event.segment_id}")
            raise ValueError(
                f"invalid {previous_state or 'new'} -> {event.kind} transition "
                f"for {event.segment_id}"
            )
        if event.kind == "commit" and event.start < self.last_commit_start:
            raise ValueError("commit events are not chronological")
        if event.kind == "commit":
            self.last_commit_start = event.start
        self.max_end = max(self.max_end, event.end)
        self.revisions[event.segment_id] = event.revision
        self.states[event.segment_id] = event.kind
