from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from .audio import AudioTimeline
from .models import AudioChunk, TranscriptEvent


class RecoveryConflictError(RuntimeError):
    pass


class RecoveryCapacityError(RuntimeError):
    pass


class RecoveryEventLimitError(RuntimeError):
    pass


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
    event_sizes: list[int] = field(default_factory=list)
    retained_event_bytes: int = 0
    first_retained_event_sequence: int = 0
    active: bool = True
    completed: bool = False
    updated_at: float = field(default_factory=monotonic)
    audio_bytes: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class RecoveryStore:
    """In-process disconnect recovery with disk-backed accepted audio."""

    def __init__(
        self,
        max_sessions: int = 128,
        max_events_per_session: int = 2_048,
        max_event_bytes: int = 512 * 1024,
        max_event_bytes_per_session: int = 8 * 1024 * 1024,
        max_audio_bytes_per_session: int = 512 * 1024 * 1024,
        max_total_audio_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_events_per_session <= 0:
            raise ValueError("max_events_per_session must be positive")
        if max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be positive")
        if max_event_bytes_per_session <= 0:
            raise ValueError("max_event_bytes_per_session must be positive")
        if max_event_bytes > max_event_bytes_per_session:
            raise ValueError("max_event_bytes cannot exceed max_event_bytes_per_session")
        if max_audio_bytes_per_session <= 0:
            raise ValueError("max_audio_bytes_per_session must be positive")
        if max_total_audio_bytes <= 0:
            raise ValueError("max_total_audio_bytes must be positive")
        if max_audio_bytes_per_session > max_total_audio_bytes:
            raise ValueError(
                "max_audio_bytes_per_session cannot exceed max_total_audio_bytes"
            )
        self.max_sessions = max_sessions
        self.max_events_per_session = max_events_per_session
        self.max_event_bytes = max_event_bytes
        self.max_event_bytes_per_session = max_event_bytes_per_session
        self.max_audio_bytes_per_session = max_audio_bytes_per_session
        self.max_total_audio_bytes = max_total_audio_bytes
        self._audio_bytes = 0
        self._sessions: dict[str, RecoverySession] = {}
        self._lock = threading.RLock()
        self._closed = False

    def open(
        self,
        config_key: str,
        requested_session_id: str | None = None,
    ) -> tuple[RecoverySession, bool]:
        with self._lock:
            if self._closed:
                raise RuntimeError("recovery store is closed")
            if requested_session_id is not None:
                session = self._sessions.get(requested_session_id)
                if session is None:
                    raise LookupError("recovery session was not found")
                if session.config_key != config_key:
                    raise ValueError("recovery session configuration changed")
                if session.completed:
                    raise ValueError("completed session cannot be resumed")
                if session.active:
                    raise RecoveryConflictError("recovery session is still active")
                self._finalize_open_partials(session)
                session.active = True
                session.updated_at = monotonic()
                return session, True

            self._evict_if_needed()
            session = RecoverySession(str(uuid4()), config_key)
            self._sessions[session.session_id] = session
            return session, False

    def prune_expired(
        self,
        max_idle_seconds: float,
        *,
        now: float | None = None,
    ) -> int:
        """Remove inactive recovery sessions whose resume window has expired."""
        if not math.isfinite(max_idle_seconds) or max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be finite and positive")
        current = monotonic() if now is None else now
        with self._lock:
            if self._closed:
                return 0
            expired = [
                session
                for session in self._sessions.values()
                if not session.active
                and current - session.updated_at >= max_idle_seconds
            ]
            for session in expired:
                self._sessions.pop(session.session_id, None)
                self._audio_bytes = max(0, self._audio_bytes - session.audio_bytes)
                session.audio_bytes = 0
        for session in expired:
            with session.lock:
                session.timeline.close()
        return len(expired)

    def _finalize_open_partials(self, session: RecoverySession) -> None:
        with session.lock:
            latest: dict[str, dict[str, object]] = {}
            for event in session.events:
                if event.get("kind") in {"partial", "commit", "replace"}:
                    latest[str(event["segment_id"])] = event
            for segment_id, event in latest.items():
                if event.get("kind") != "partial":
                    continue
                revision = event.get("revision")
                end = event.get("end")
                metadata_value = event.get("metadata")
                if isinstance(revision, bool) or not isinstance(revision, int):
                    raise TypeError("retained event has an invalid revision")
                if isinstance(end, bool) or not isinstance(end, (int, float)):
                    raise TypeError("retained event has an invalid end timestamp")
                if metadata_value is not None and not isinstance(metadata_value, dict):
                    raise TypeError("retained event has invalid metadata")
                recovered = dict(event)
                recovered["kind"] = "commit"
                recovered["revision"] = revision + 1
                recovered["sequence"] = session.next_event_sequence
                metadata = dict(metadata_value or {})
                metadata["recovery_committed"] = True
                recovered["metadata"] = metadata
                self._retain_event(session, recovered)
                session.next_event_sequence += 1
                session.transcribed_through = max(
                    session.transcribed_through,
                    float(end),
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
            raise RecoveryCapacityError("recovery session capacity reached")
        oldest = min(candidates, key=lambda item: item.updated_at)
        oldest.timeline.close()
        self._audio_bytes -= oldest.audio_bytes
        self._sessions.pop(oldest.session_id, None)

    def append_audio(self, session: RecoverySession, chunk: AudioChunk) -> int:
        size = len(chunk.pcm_s16le)
        with self._lock:
            if self._closed:
                raise RuntimeError("recovery store is closed")
            if session.audio_bytes + size > self.max_audio_bytes_per_session:
                raise RecoveryCapacityError("recovery session audio capacity reached")
            if self._audio_bytes + size > self.max_total_audio_bytes:
                raise RecoveryCapacityError("recovery store audio capacity reached")
            session.audio_bytes += size
            self._audio_bytes += size
        try:
            with session.lock:
                sequence = session.next_audio_sequence
                session.timeline.append(chunk)
                session.next_audio_sequence += 1
                session.updated_at = monotonic()
                return sequence
        except BaseException:
            with self._lock:
                if session.audio_bytes >= size:
                    session.audio_bytes -= size
                    self._audio_bytes = max(0, self._audio_bytes - size)
            raise

    def _retain_event(
        self,
        session: RecoverySession,
        payload: dict[str, object],
        payload_bytes: int | None = None,
    ) -> None:
        if payload_bytes is None:
            payload_bytes = len(json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
        if payload_bytes > self.max_event_bytes:
            raise RecoveryEventLimitError(
                f"event payload exceeds {self.max_event_bytes} UTF-8 bytes"
            )
        session.events.append(payload)
        session.event_sizes.append(payload_bytes)
        session.retained_event_bytes += payload_bytes
        while (
            len(session.events) > self.max_events_per_session
            or session.retained_event_bytes > self.max_event_bytes_per_session
        ):
            session.events.pop(0)
            session.retained_event_bytes -= session.event_sizes.pop(0)
        if session.events:
            first_sequence = session.events[0].get("sequence")
            if isinstance(first_sequence, int):
                session.first_retained_event_sequence = first_sequence
        else:
            session.first_retained_event_sequence = session.next_event_sequence

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
            payload = event.to_dict()
            payload_bytes = len(json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
            self._retain_event(session, payload, payload_bytes)
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
            session.updated_at = monotonic()
            return payload

    def replay_after(
        self,
        session: RecoverySession,
        acknowledged_event_sequence: int,
    ) -> list[dict[str, object]]:
        with session.lock:
            if acknowledged_event_sequence < session.first_retained_event_sequence - 1:
                raise ValueError(
                    "acknowledged event is older than the retained recovery window"
                )
            replay = []
            for event in session.events:
                sequence = event.get("sequence")
                if (
                    isinstance(sequence, int)
                    and not isinstance(sequence, bool)
                    and sequence > acknowledged_event_sequence
                ):
                    replay.append(dict(event))
            return replay

    def release(self, session: RecoverySession, *, completed: bool = False) -> None:
        close_timeline = False
        with self._lock:
            session.completed = session.completed or completed
            session.active = False
            session.updated_at = monotonic()
            if session.completed and session.audio_bytes:
                self._audio_bytes -= session.audio_bytes
                session.audio_bytes = 0
            close_timeline = session.completed
        if close_timeline:
            session.timeline.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._audio_bytes = 0
            for session in sessions:
                session.audio_bytes = 0
        for session in sessions:
            with session.lock:
                session.timeline.close()
