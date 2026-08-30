from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from .audio import AudioTimeline
from .models import AudioChunk, TranscriptEvent


@dataclass(slots=True)
class RecoverySession:
    session_id: str
    config_key: str
    timeline: AudioTimeline = field(default_factory=AudioTimeline)
    next_audio_sequence: int = 0
    next_event_sequence: int = 0
    next_segment_index: int = 0
    transcribed_through: float = 0.0
    events: list[dict[str, object]] = field(default_factory=list)
    active: bool = True
    completed: bool = False
    updated_at: float = field(default_factory=monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


class RecoveryStore:
    """In-process disconnect recovery with disk-backed accepted audio."""

    def __init__(self, max_sessions: int = 128) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._sessions: dict[str, RecoverySession] = {}
        self._lock = threading.RLock()

    def open(
        self,
        config_key: str,
        requested_session_id: str | None = None,
    ) -> tuple[RecoverySession, bool]:
        with self._lock:
            if requested_session_id is not None:
                session = self._sessions.get(requested_session_id)
                if session is None:
                    raise LookupError("recovery session was not found")
                if session.config_key != config_key:
                    raise ValueError("recovery session configuration changed")
                if session.completed:
                    raise ValueError("completed session cannot be resumed")
                if session.active:
                    raise RuntimeError("recovery session is still active")
                self._finalize_open_partials(session)
                session.active = True
                session.updated_at = monotonic()
                return session, True

            self._evict_if_needed()
            session = RecoverySession(str(uuid4()), config_key)
            self._sessions[session.session_id] = session
            return session, False

    def _finalize_open_partials(self, session: RecoverySession) -> None:
        with session.lock:
            latest: dict[str, dict[str, object]] = {}
            for event in session.events:
                if event.get("kind") in {"partial", "commit", "replace"}:
                    latest[str(event["segment_id"])] = event
            for segment_id, event in latest.items():
                if event.get("kind") != "partial":
                    continue
                recovered = dict(event)
                recovered["kind"] = "commit"
                recovered["revision"] = int(event["revision"]) + 1
                recovered["sequence"] = session.next_event_sequence
                metadata = dict(event.get("metadata") or {})
                metadata["recovery_committed"] = True
                recovered["metadata"] = metadata
                session.next_event_sequence += 1
                session.events.append(recovered)
                session.transcribed_through = max(
                    session.transcribed_through,
                    float(recovered["end"]),
                )
                if segment_id.startswith("seg-"):
                    try:
                        index = int(segment_id.removeprefix("seg-"))
                    except ValueError:
                        continue
                    session.next_segment_index = max(session.next_segment_index, index + 1)

    def _evict_if_needed(self) -> None:
        if len(self._sessions) < self.max_sessions:
            return
        candidates = [
            session for session in self._sessions.values()
            if not session.active
        ]
        if not candidates:
            raise RuntimeError("recovery session capacity reached")
        oldest = min(candidates, key=lambda item: item.updated_at)
        oldest.timeline.close()
        self._sessions.pop(oldest.session_id, None)

    def append_audio(self, session: RecoverySession, chunk: AudioChunk) -> int:
        with session.lock:
            sequence = session.next_audio_sequence
            session.timeline.append(chunk)
            session.next_audio_sequence += 1
            session.updated_at = monotonic()
            return sequence

    def append_event(
        self,
        session: RecoverySession,
        event: TranscriptEvent,
    ) -> dict[str, object]:
        with session.lock:
            event.session_id = session.session_id
            event.sequence = session.next_event_sequence
            event.acknowledged_sequence = (
                session.next_audio_sequence - 1
                if session.next_audio_sequence else None
            )
            session.next_event_sequence += 1
            if event.segment_id.startswith("seg-"):
                try:
                    index = int(event.segment_id.removeprefix("seg-"))
                except ValueError:
                    pass
                else:
                    session.next_segment_index = max(session.next_segment_index, index + 1)
            if event.kind in {"commit", "replace"}:
                session.transcribed_through = max(session.transcribed_through, event.end)
            payload = event.to_dict()
            session.events.append(payload)
            session.updated_at = monotonic()
            return payload

    def replay_after(
        self,
        session: RecoverySession,
        acknowledged_event_sequence: int,
    ) -> list[dict[str, object]]:
        with session.lock:
            return [
                dict(event)
                for event in session.events
                if isinstance(event.get("sequence"), int)
                and int(event["sequence"]) > acknowledged_event_sequence
            ]

    def release(self, session: RecoverySession, *, completed: bool = False) -> None:
        with self._lock:
            session.completed = session.completed or completed
            session.active = False
            session.updated_at = monotonic()

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.timeline.close()
