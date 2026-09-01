from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import stat
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter
from typing import cast
from urllib.parse import urlsplit

from .jsonutil import strict_json_object
from .models import TranscriptEvent
from .validation import EventStreamValidator

_MAX_PROBE_AUDIO_FILE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _ProbeAudioMaterial:
    pcm: bytes
    artifact_sha256: str | None
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class WebSocketSessionResult:
    session: int
    passed: bool
    backend: str | None = None
    backend_implementation: str | None = None
    model: str | None = None
    model_revision: str | None = None
    device: str | None = None
    language: str | None = None
    compute_type: str | None = None
    loaded_models: tuple[dict[str, object], ...] = ()
    ready_seconds: float | None = None
    total_seconds: float | None = None
    events: int = 0
    partials: int = 0
    commits: int = 0
    audio_acks: int = 0
    last_acknowledged_sequence: int | None = None
    final_buffered_bytes: int | None = None
    backpressure_pauses: int = 0
    dropped_partials: int = 0
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class WebSocketRecoveryResult:
    passed: bool
    backend: str | None = None
    backend_implementation: str | None = None
    model: str | None = None
    model_revision: str | None = None
    device: str | None = None
    language: str | None = None
    compute_type: str | None = None
    loaded_models: tuple[dict[str, object], ...] = ()
    disconnected_audio_seconds: float | None = None
    first_last_acknowledged_sequence: int | None = None
    resumed_next_audio_sequence: int | None = None
    final_acknowledged_sequence: int | None = None
    final_buffered_bytes: int | None = None
    events: int = 0
    commits: int = 0
    audio_acks: int = 0
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class WebSocketGateReport:
    status: str
    source_commit: str | None
    created_at: str
    validity_seconds: float
    uri: str
    identity_consistent: bool
    backend: str | None
    backend_implementation: str | None
    model: str | None
    model_revision: str | None
    device: str | None
    language: str | None
    compute_type: str | None
    loaded_models: tuple[dict[str, object], ...]
    probe_audio_sha256: str | None
    probe_audio_bytes: int
    probe_audio_rms: float
    sessions: int
    passed_sessions: int
    failed_sessions: int
    audio_seconds_per_session: float
    realtime_pacing: bool
    min_commits_per_session: int
    min_audio_acks_per_session: int
    max_dropped_partials_per_session: int | None
    max_backpressure_pauses_per_session: int | None
    max_ready_seconds: float | None
    max_total_seconds: float | None
    ready_seconds_p95: float | None
    total_seconds_p95: float | None
    events: int
    commits: int
    audio_acks: int
    backpressure_pauses: int
    dropped_partials: int
    recovery_probe_required: bool
    recovery_probe: WebSocketRecoveryResult | None
    results: tuple[WebSocketSessionResult, ...]

    @property
    def passed(self) -> bool:
        sessions_passed = (
            self.failed_sessions == 0 and self.passed_sessions == self.sessions
        )
        recovery_passed = (
            not self.recovery_probe_required
            or self.recovery_probe is not None and self.recovery_probe.passed
        )
        return sessions_passed and recovery_passed and self.identity_consistent

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ReadyIdentity:
    backend: str
    backend_implementation: str
    model: str | None
    model_revision: str | None
    device: str | None
    language: str | None
    compute_type: str | None
    loaded_models: tuple[dict[str, object], ...]


def _valid_identity(value: object, *, required: bool = False) -> bool:
    if value is None:
        return not required
    return (
        isinstance(value, str)
        and bool(value)
        and value.strip() == value
        and len(value) <= 512
        and not any(
            ord(character) < 32 or ord(character) == 127 for character in value
        )
    )


def _loaded_model_entry(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        return None
    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    if (
        not isinstance(path, str)
        or not path.startswith("/var/lib/turnalign/models/")
        or not isinstance(digest, str)
        or len(digest) != 64
        or not all(character in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        return None
    return {
        "path": path,
        "sha256": digest,
        "bytes": size,
    }


def _loaded_models(value: object) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("server ready response has invalid loaded_models")
    entries = []
    for item in value:
        entry = _loaded_model_entry(item)
        if entry is None:
            raise ValueError("server ready response has an invalid loaded model entry")
        entries.append(entry)
    identities = {
        (entry["path"], entry["sha256"], entry["bytes"])
        for entry in entries
    }
    if len(identities) != len(entries):
        raise ValueError("server ready response contains duplicate loaded model entries")
    return tuple(entries)


def _ready_identity(
    payload: dict[str, object],
    *,
    requested_backend: str | None,
    requested_model: str | None,
    requested_language: str | None,
    requested_compute_type: str | None,
) -> _ReadyIdentity:
    backend = payload.get("backend")
    backend_implementation = payload.get("backend_implementation")
    config = payload.get("config")
    revision = payload.get("model_revision")
    if not _valid_identity(backend, required=True):
        raise ValueError("server ready response has no valid backend identity")
    if not _valid_identity(backend_implementation, required=True):
        raise ValueError(
            "server ready response has no valid backend implementation identity"
        )
    if not isinstance(config, dict):
        raise TypeError("server ready response has no configuration identity")
    values: dict[str, str | None] = {}
    for key in ("model", "device", "language", "compute_type"):
        value = config.get(key)
        if not _valid_identity(value):
            raise ValueError(f"server ready response has an invalid {key} identity")
        values[key] = cast(str | None, value)
    if not _valid_identity(revision):
        raise ValueError("server ready response has an invalid model revision")
    loaded_models = _loaded_models(payload.get("loaded_models"))
    for label, requested, observed in (
        ("backend", requested_backend, backend),
        ("model", requested_model, values["model"]),
        ("language", requested_language, values["language"]),
        ("compute_type", requested_compute_type, values["compute_type"]),
    ):
        if requested is not None and observed != requested:
            raise ValueError(f"server ready response changed the requested {label}")
    return _ReadyIdentity(
        backend=cast(str, backend),
        backend_implementation=cast(str, backend_implementation),
        model=values["model"],
        model_revision=cast(str | None, revision),
        device=values["device"],
        language=values["language"],
        compute_type=values["compute_type"],
        loaded_models=loaded_models,
    )


def _parse_server_response(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise TypeError("server returned an unexpected binary frame")
    return strict_json_object(raw, label="server response")


def _validate_options(
    *,
    sessions: int,
    audio_seconds: float,
    sample_rate: int,
    channels: int,
    frame_ms: int,
    timeout: float,
    min_commits: int,
    min_audio_acks: int,
    max_dropped_partials: int | None,
    max_backpressure_pauses: int | None,
    max_ready_seconds: float | None,
    max_total_seconds: float | None,
    verify_recovery: bool,
    recovery_resume_timeout: float,
) -> None:
    for name, value in (
        ("sessions", sessions),
        ("sample_rate", sample_rate),
        ("channels", channels),
        ("frame_ms", frame_ms),
        ("min_commits", min_commits),
        ("min_audio_acks", min_audio_acks),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    if not isfinite(audio_seconds) or audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive and finite")
    if not 8_000 <= sample_rate <= 96_000:
        raise ValueError("sample_rate must be between 8000 and 96000")
    if not 1 <= channels <= 8:
        raise ValueError("channels must be between 1 and 8")
    if not 20 <= frame_ms <= 10_000:
        raise ValueError("frame_ms must be between 20 and 10000")
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if min_commits < 0:
        raise ValueError("min_commits must be non-negative")
    if min_audio_acks < 0:
        raise ValueError("min_audio_acks must be non-negative")
    for name, optional_integer in (
        ("max_dropped_partials", max_dropped_partials),
        ("max_backpressure_pauses", max_backpressure_pauses),
    ):
        if optional_integer is not None and (
            isinstance(optional_integer, bool)
            or not isinstance(optional_integer, int)
            or optional_integer < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or omitted")
    for name, optional_seconds in (
        ("max_ready_seconds", max_ready_seconds),
        ("max_total_seconds", max_total_seconds),
    ):
        if optional_seconds is not None and (
            not isfinite(optional_seconds) or optional_seconds <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(verify_recovery, bool):
        raise TypeError("verify_recovery must be a boolean")
    if not isfinite(recovery_resume_timeout) or recovery_resume_timeout <= 0:
        raise ValueError("recovery_resume_timeout must be positive and finite")
    if verify_recovery:
        total_samples = round(sample_rate * audio_seconds)
        frame_samples = max(1, round(sample_rate * frame_ms / 1000))
        if total_samples < frame_samples * 2:
            raise ValueError(
                "recovery verification requires at least two audio frames"
            )


def _synthetic_probe_audio(
    total_samples: int,
    sample_rate: int,
    channels: int,
) -> bytes:
    frame = bytearray()
    amplitude = 0.25
    for index in range(total_samples):
        value = int(
            amplitude * 32767 * math.sin(
                2 * math.pi * 440 * index / sample_rate
            )
        )
        encoded = value.to_bytes(2, "little", signed=True)
        for _ in range(channels):
            frame.extend(encoded)
    return bytes(frame)


def _rms_s16le(pcm: bytes) -> float:
    if not pcm or len(pcm) % 2:
        return 0.0
    values = (
        int.from_bytes(pcm[index:index + 2], "little", signed=True)
        for index in range(0, len(pcm), 2)
    )
    count = 0
    total = 0
    for value in values:
        total += value * value
        count += 1
    return math.sqrt(total / count) if count else 0.0


def _probe_audio_material(
    *,
    audio_seconds: float,
    sample_rate: int,
    channels: int,
    probe_audio_path: Path | None,
) -> _ProbeAudioMaterial:
    total_samples = max(1, round(sample_rate * audio_seconds))
    frame_bytes = channels * 2
    if probe_audio_path is None:
        pcm = _synthetic_probe_audio(total_samples, sample_rate, channels)
        return _ProbeAudioMaterial(pcm, None, len(pcm))
    if sample_rate != 16_000 or channels != 1:
        raise ValueError("probe-audio must be a 16 kHz mono PCM16 WAV file")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        probe_audio_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
    )
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_size <= 0
            or initial.st_size > _MAX_PROBE_AUDIO_FILE_BYTES
        ):
            raise ValueError("probe-audio must be a bounded regular file")
        with os.fdopen(descriptor, "rb") as opened:
            descriptor = -1
            artifact = opened.read(_MAX_PROBE_AUDIO_FILE_BYTES + 1)
            final = os.fstat(opened.fileno())
        current = os.lstat(probe_audio_path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        len(artifact) != initial.st_size
        or len(artifact) > _MAX_PROBE_AUDIO_FILE_BYTES
        or (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        or (final.st_dev, final.st_ino) != (current.st_dev, current.st_ino)
        or stat.S_ISLNK(current.st_mode)
    ):
        raise ValueError("probe-audio changed while it was being read")
    with wave.open(io.BytesIO(artifact), "rb") as source:
        if (
            source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
            or source.getframerate() != 16_000
            or source.getnchannels() != 1
        ):
            raise ValueError(
                "probe-audio must be uncompressed signed 16-bit 16 kHz mono PCM"
            )
        data = source.readframes(total_samples)
    if len(data) < total_samples * frame_bytes:
        raise ValueError(
            "probe-audio is shorter than the requested per-session duration"
        )
    return _ProbeAudioMaterial(
        data[: total_samples * frame_bytes],
        hashlib.sha256(artifact).hexdigest(),
        len(artifact),
    )


def _validate_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("uri must be an absolute ws:// or wss:// endpoint")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("uri contains an invalid port") from error
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "uri credentials, query strings and fragments are forbidden; "
            "use --auth-token-file or --auth-token-env"
        )


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * 0.95) - 1)]


async def _run_session(
    uri: str,
    session: int,
    *,
    audio_seconds: float,
    sample_rate: int,
    channels: int,
    frame_ms: int,
    timeout: float,
    min_commits: int,
    min_audio_acks: int,
    max_dropped_partials: int | None,
    max_backpressure_pauses: int | None,
    max_ready_seconds: float | None,
    max_total_seconds: float | None,
    realtime: bool,
    backend: str | None,
    model: str | None,
    language: str | None,
    compute_type: str | None,
    auth_token: str | None,
    probe_audio: bytes,
) -> WebSocketSessionResult:
    try:
        from websockets.asyncio.client import connect
    except ImportError as error:
        raise RuntimeError(
            "WebSocket gate requires: pip install 'turnalign[server]'"
        ) from error

    async def execute() -> WebSocketSessionResult:
        started = perf_counter()
        async with connect(uri, max_size=20 * 1024 * 1024) as websocket:
            request: dict[str, object] = {
                "type": "start",
                "sample_rate": sample_rate,
                "channels": channels,
            }
            for key, value in (
                ("backend", backend),
                ("model", model),
                ("language", language),
                ("compute_type", compute_type),
                ("auth", auth_token),
            ):
                if value is not None:
                    request[key] = value
            await websocket.send(json.dumps(request, ensure_ascii=False))
            ready = _parse_server_response(await websocket.recv())
            if ready.get("type") == "error":
                raise RuntimeError(
                    f"server rejected session: {ready.get('code', 'unknown_error')}: "
                    f"{ready.get('message', 'request failed')}"
                )
            if ready.get("type") != "ready" or ready.get("protocol_version") != 1:
                raise ValueError("server did not return a protocol v1 ready response")
            session_id = ready.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("server ready response is missing a session_id")
            identity = _ready_identity(
                ready,
                requested_backend=backend,
                requested_model=model,
                requested_language=language,
                requested_compute_type=compute_type,
            )
            ready_seconds = perf_counter() - started

            flow_allowed = asyncio.Event()
            flow_allowed.set()
            validator = EventStreamValidator()
            counters = {
                "events": 0,
                "partials": 0,
                "commits": 0,
                "audio_acks": 0,
                "backpressure_pauses": 0,
                "dropped_partials": 0,
            }
            last_acknowledged_sequence: int | None = None
            final_buffered_bytes: int | None = None

            async def receive() -> None:
                nonlocal final_buffered_bytes, last_acknowledged_sequence
                while True:
                    payload = _parse_server_response(await websocket.recv())
                    message_type = payload.get("type")
                    if message_type == "error":
                        raise RuntimeError(
                            f"server session failed: {payload.get('code', 'unknown_error')}: "
                            f"{payload.get('message', 'request failed')}"
                        )
                    if message_type == "audio_ack":
                        if payload.get("session_id") != session_id:
                            raise ValueError("audio acknowledgement session_id changed")
                        buffered_bytes = payload.get("buffered_bytes")
                        if (
                            isinstance(buffered_bytes, bool)
                            or not isinstance(buffered_bytes, int)
                            or buffered_bytes < 0
                        ):
                            raise ValueError(
                                "audio acknowledgement has invalid buffered_bytes"
                            )
                        final_buffered_bytes = buffered_bytes
                        acknowledged = payload.get("acknowledged_sequence")
                        if acknowledged is not None:
                            if (
                                isinstance(acknowledged, bool)
                                or not isinstance(acknowledged, int)
                                or acknowledged < 0
                            ):
                                raise ValueError(
                                    "audio acknowledgement has invalid sequence"
                                )
                            if (
                                last_acknowledged_sequence is not None
                                and acknowledged < last_acknowledged_sequence
                            ):
                                raise ValueError(
                                    "audio acknowledgement sequence moved backwards"
                                )
                            last_acknowledged_sequence = acknowledged
                        counters["audio_acks"] += 1
                        continue
                    if message_type == "flow_control":
                        action = payload.get("action")
                        if action == "pause":
                            counters["backpressure_pauses"] += 1
                            flow_allowed.clear()
                        elif action == "resume":
                            flow_allowed.set()
                        else:
                            raise ValueError("invalid flow-control action")
                        continue
                    if "kind" not in payload:
                        continue
                    event = TranscriptEvent.from_dict(payload)
                    if event.session_id != session_id:
                        raise ValueError("event session_id does not match ready response")
                    validator.accept(event)
                    counters["events"] += 1
                    counters["partials"] += event.kind == "partial"
                    counters["commits"] += event.kind == "commit"
                    if event.kind == "end":
                        counters["dropped_partials"] = int(
                            event.metadata.get("dropped_partials", 0)
                        )
                        return

            receiver = asyncio.create_task(receive())
            try:
                frame_samples = max(1, round(sample_rate * frame_ms / 1000))
                total_samples = max(1, round(sample_rate * audio_seconds))
                expected_bytes = total_samples * channels * 2
                if len(probe_audio) != expected_bytes:
                    raise ValueError("probe audio byte length does not match session duration")
                sent_samples = 0
                sent_bytes = 0
                while sent_samples < total_samples:
                    if receiver.done():
                        await receiver
                    await flow_allowed.wait()
                    samples = min(frame_samples, total_samples - sent_samples)
                    block = probe_audio[sent_bytes:sent_bytes + samples * channels * 2]
                    await websocket.send(block)
                    sent_samples += samples
                    sent_bytes += len(block)
                    if realtime:
                        await asyncio.sleep(samples / sample_rate)
                    else:
                        await asyncio.sleep(0)
                await websocket.send(json.dumps({"type": "end"}))
                await receiver
            finally:
                if not receiver.done():
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)

            if not validator.ended:
                raise ValueError("server closed without a terminal end event")
            total_seconds = perf_counter() - started
            failures = []
            if counters["commits"] < min_commits:
                failures.append(
                    f"commit count {counters['commits']} is below minimum {min_commits}"
                )
            if counters["audio_acks"] < min_audio_acks:
                failures.append(
                    f"audio acknowledgement count {counters['audio_acks']} is below "
                    f"minimum {min_audio_acks}"
                )
            if last_acknowledged_sequence is None:
                failures.append("no accepted audio sequence was acknowledged")
            if final_buffered_bytes != 0:
                failures.append(
                    "final audio acknowledgement did not drain the frame buffer"
                )
            if (
                max_dropped_partials is not None
                and counters["dropped_partials"] > max_dropped_partials
            ):
                failures.append(
                    f"dropped partial count {counters['dropped_partials']} exceeds "
                    f"maximum {max_dropped_partials}"
                )
            if (
                max_backpressure_pauses is not None
                and counters["backpressure_pauses"] > max_backpressure_pauses
            ):
                failures.append(
                    f"backpressure pause count {counters['backpressure_pauses']} exceeds "
                    f"maximum {max_backpressure_pauses}"
                )
            if max_ready_seconds is not None and ready_seconds > max_ready_seconds:
                failures.append(
                    f"ready latency {ready_seconds:.3f}s exceeds {max_ready_seconds:.3f}s"
                )
            if max_total_seconds is not None and total_seconds > max_total_seconds:
                failures.append(
                    f"total latency {total_seconds:.3f}s exceeds {max_total_seconds:.3f}s"
                )
            failure = "; ".join(failures) or None
            return WebSocketSessionResult(
                session=session,
                passed=failure is None,
                backend=identity.backend,
                backend_implementation=identity.backend_implementation,
                model=identity.model,
                model_revision=identity.model_revision,
                device=identity.device,
                language=identity.language,
                compute_type=identity.compute_type,
                loaded_models=identity.loaded_models,
                ready_seconds=ready_seconds,
                total_seconds=total_seconds,
                events=counters["events"],
                partials=counters["partials"],
                commits=counters["commits"],
                audio_acks=counters["audio_acks"],
                last_acknowledged_sequence=last_acknowledged_sequence,
                final_buffered_bytes=final_buffered_bytes,
                backpressure_pauses=counters["backpressure_pauses"],
                dropped_partials=counters["dropped_partials"],
                failure=failure,
            )

    try:
        return await asyncio.wait_for(execute(), timeout=timeout)
    except asyncio.TimeoutError:
        return WebSocketSessionResult(
            session=session,
            passed=False,
            failure=f"session exceeded {timeout:.3f}s timeout",
        )
    except Exception as error:  # noqa: BLE001 - deployment boundary
        return WebSocketSessionResult(
            session=session,
            passed=False,
            failure=f"{type(error).__name__}: {error}",
        )


async def _run_recovery_probe(
    uri: str,
    *,
    audio_seconds: float,
    sample_rate: int,
    channels: int,
    frame_ms: int,
    timeout: float,
    recovery_resume_timeout: float,
    realtime: bool,
    backend: str | None,
    model: str | None,
    language: str | None,
    compute_type: str | None,
    auth_token: str | None,
    probe_audio: bytes,
) -> WebSocketRecoveryResult:
    try:
        from websockets.asyncio.client import connect
    except ImportError as error:
        raise RuntimeError(
            "WebSocket gate requires: pip install 'turnalign[server]'"
        ) from error

    async def execute() -> WebSocketRecoveryResult:
        validator = EventStreamValidator()
        flow_allowed = asyncio.Event()
        flow_allowed.set()
        counters = {"events": 0, "commits": 0, "audio_acks": 0}
        session_id = ""
        resume_token: str | None = None
        last_acknowledged_sequence: int | None = None
        final_buffered_bytes: int | None = None
        highest_event_sequence = -1
        first_identity: _ReadyIdentity | None = None

        def start_request(
            *,
            resume_session_id: str | None = None,
        ) -> dict[str, object]:
            request: dict[str, object] = {
                "type": "start",
                "sample_rate": sample_rate,
                "channels": channels,
            }
            for key, value in (
                ("backend", backend),
                ("model", model),
                ("language", language),
                ("compute_type", compute_type),
                ("auth", auth_token),
            ):
                if value is not None:
                    request[key] = value
            if resume_session_id is not None:
                if resume_token is None:
                    raise RuntimeError("recovery probe has no resume token")
                request["resume_session_id"] = resume_session_id
                request["resume_token"] = resume_token
                request["acknowledged_event_sequence"] = highest_event_sequence
            return request

        async def receive_json(websocket) -> dict[str, object]:
            return _parse_server_response(await websocket.recv())

        def validate_ready(
            payload: dict[str, object],
            *,
            expected_session_id: str | None = None,
            expected_resume_token: str | None = None,
        ) -> tuple[str, int, str, _ReadyIdentity]:
            if payload.get("type") == "error":
                raise RuntimeError(
                    f"server rejected recovery probe: "
                    f"{payload.get('code', 'unknown_error')}: "
                    f"{payload.get('message', 'request failed')}"
                )
            if payload.get("type") != "ready" or payload.get("protocol_version") != 1:
                raise ValueError("server did not return a protocol v1 ready response")
            ready_session_id = payload.get("session_id")
            if not isinstance(ready_session_id, str) or not ready_session_id:
                raise ValueError("server ready response is missing a session_id")
            if expected_session_id is not None and ready_session_id != expected_session_id:
                raise ValueError("recovered session_id changed")
            ready_resume_token = payload.get("resume_token")
            if not isinstance(ready_resume_token, str) or not ready_resume_token:
                raise ValueError("server ready response is missing a resume_token")
            if (
                expected_resume_token is not None
                and ready_resume_token != expected_resume_token
            ):
                raise ValueError("recovered resume_token changed")
            next_audio_sequence = payload.get("next_audio_sequence")
            if (
                isinstance(next_audio_sequence, bool)
                or not isinstance(next_audio_sequence, int)
                or next_audio_sequence < 0
            ):
                raise ValueError("ready response has invalid next_audio_sequence")
            identity = _ready_identity(
                payload,
                requested_backend=backend,
                requested_model=model,
                requested_language=language,
                requested_compute_type=compute_type,
            )
            return ready_session_id, next_audio_sequence, ready_resume_token, identity

        async def consume_message(websocket) -> None:
            nonlocal final_buffered_bytes, highest_event_sequence
            nonlocal last_acknowledged_sequence
            payload = await receive_json(websocket)
            message_type = payload.get("type")
            if message_type == "error":
                raise RuntimeError(
                    f"server recovery session failed: "
                    f"{payload.get('code', 'unknown_error')}: "
                    f"{payload.get('message', 'request failed')}"
                )
            if message_type == "audio_ack":
                if payload.get("session_id") != session_id:
                    raise ValueError("audio acknowledgement session_id changed")
                buffered_bytes = payload.get("buffered_bytes")
                if (
                    isinstance(buffered_bytes, bool)
                    or not isinstance(buffered_bytes, int)
                    or buffered_bytes < 0
                ):
                    raise ValueError("audio acknowledgement has invalid buffered_bytes")
                final_buffered_bytes = buffered_bytes
                acknowledged = payload.get("acknowledged_sequence")
                if acknowledged is not None:
                    if (
                        isinstance(acknowledged, bool)
                        or not isinstance(acknowledged, int)
                        or acknowledged < 0
                    ):
                        raise ValueError("audio acknowledgement has invalid sequence")
                    if (
                        last_acknowledged_sequence is not None
                        and acknowledged < last_acknowledged_sequence
                    ):
                        raise ValueError("audio acknowledgement sequence moved backwards")
                    last_acknowledged_sequence = acknowledged
                counters["audio_acks"] += 1
                return
            if message_type == "flow_control":
                action = payload.get("action")
                if action == "pause":
                    flow_allowed.clear()
                elif action == "resume":
                    flow_allowed.set()
                else:
                    raise ValueError("invalid flow-control action")
                return
            if "kind" not in payload:
                return
            event = TranscriptEvent.from_dict(payload)
            if event.session_id != session_id:
                raise ValueError("event session_id does not match recovery session")
            validator.accept(event)
            counters["events"] += 1
            counters["commits"] += event.kind == "commit"
            if event.sequence is not None:
                highest_event_sequence = max(highest_event_sequence, event.sequence)

        async def wait_until(websocket, predicate) -> None:
            while not predicate():
                await consume_message(websocket)

        frame_samples = max(1, round(sample_rate * frame_ms / 1000))
        total_samples = max(1, round(sample_rate * audio_seconds))
        disconnect_target = total_samples // 2
        sent_samples = 0
        sent_bytes = 0
        expected_bytes = total_samples * channels * 2
        if len(probe_audio) != expected_bytes:
            raise ValueError("probe audio byte length does not match recovery duration")

        first = await connect(uri, max_size=20 * 1024 * 1024)
        try:
            await first.send(json.dumps(start_request(), ensure_ascii=False))
            ready = await receive_json(first)
            session_id, _, resume_token, first_identity = validate_ready(ready)
            while (
                sent_samples < disconnect_target
                or last_acknowledged_sequence is None
                or final_buffered_bytes != 0
            ):
                while not flow_allowed.is_set():
                    await consume_message(first)
                if total_samples - sent_samples <= frame_samples:
                    raise ValueError(
                        "audio sample is too short to disconnect on an accepted frame "
                        "boundary and retain post-resume audio"
                    )
                samples = min(frame_samples, total_samples - sent_samples)
                previous_acks = counters["audio_acks"]
                block = probe_audio[sent_bytes:sent_bytes + samples * channels * 2]
                await first.send(block)
                sent_samples += samples
                sent_bytes += len(block)
                if realtime:
                    await asyncio.sleep(samples / sample_rate)
                await wait_until(
                    first,
                    lambda previous_acks=previous_acks: (
                        counters["audio_acks"] > previous_acks
                    ),
                )
        finally:
            await first.close()

        first_last_ack = last_acknowledged_sequence
        disconnected_audio_seconds = sent_samples / sample_rate
        if first_last_ack is None or final_buffered_bytes != 0:
            raise ValueError("disconnect boundary was not durably acknowledged")

        resume_deadline = perf_counter() + recovery_resume_timeout
        resumed = None
        resumed_next_audio_sequence = None
        while resumed is None:
            candidate = await connect(uri, max_size=20 * 1024 * 1024)
            try:
                await candidate.send(json.dumps(
                    start_request(resume_session_id=session_id),
                    ensure_ascii=False,
                ))
                response = await receive_json(candidate)
            except BaseException:
                await candidate.close()
                raise
            if response.get("type") == "error" and response.get("code") == "session_conflict":
                await candidate.close()
                if perf_counter() >= resume_deadline:
                    raise TimeoutError("recovery session remained active after disconnect")
                await asyncio.sleep(0.05)
                continue
            try:
                (
                    recovered_session_id,
                    resumed_next_audio_sequence,
                    _recovered_resume_token,
                    recovered_identity,
                ) = validate_ready(
                    response,
                    expected_session_id=session_id,
                    expected_resume_token=resume_token,
                )
                if recovered_session_id != session_id or response.get("resumed") is not True:
                    raise ValueError("server did not confirm recovery resume")
                if recovered_identity != first_identity:
                    raise ValueError("recovered server identity changed")
                if resumed_next_audio_sequence != first_last_ack + 1:
                    raise ValueError(
                        "recovered next_audio_sequence does not continue the first connection"
                    )
            except BaseException:
                await candidate.close()
                raise
            resumed = candidate
            flow_allowed.set()

        try:
            while sent_samples < total_samples:
                while not flow_allowed.is_set():
                    await consume_message(resumed)
                samples = min(frame_samples, total_samples - sent_samples)
                previous_acks = counters["audio_acks"]
                block = probe_audio[sent_bytes:sent_bytes + samples * channels * 2]
                await resumed.send(block)
                sent_samples += samples
                sent_bytes += len(block)
                if realtime:
                    await asyncio.sleep(samples / sample_rate)
                await wait_until(
                    resumed,
                    lambda previous_acks=previous_acks: (
                        counters["audio_acks"] > previous_acks
                    ),
                )
            await resumed.send(json.dumps({"type": "end"}))
            await wait_until(resumed, lambda: validator.ended)
        finally:
            await resumed.close()

        failures = []
        if final_buffered_bytes != 0:
            failures.append("final audio acknowledgement did not drain the frame buffer")
        if (
            last_acknowledged_sequence is None
            or last_acknowledged_sequence <= first_last_ack
        ):
            failures.append("audio acknowledgement sequence did not advance after resume")
        if not validator.ended:
            failures.append("recovered session did not emit a terminal end event")
        failure = "; ".join(failures) or None
        if first_identity is None:
            raise RuntimeError("recovery probe completed without a server identity")
        return WebSocketRecoveryResult(
            passed=failure is None,
            backend=first_identity.backend,
            backend_implementation=first_identity.backend_implementation,
            model=first_identity.model,
            model_revision=first_identity.model_revision,
            device=first_identity.device,
            language=first_identity.language,
            compute_type=first_identity.compute_type,
            loaded_models=first_identity.loaded_models,
            disconnected_audio_seconds=disconnected_audio_seconds,
            first_last_acknowledged_sequence=first_last_ack,
            resumed_next_audio_sequence=resumed_next_audio_sequence,
            final_acknowledged_sequence=last_acknowledged_sequence,
            final_buffered_bytes=final_buffered_bytes,
            events=counters["events"],
            commits=counters["commits"],
            audio_acks=counters["audio_acks"],
            failure=failure,
        )

    try:
        return await asyncio.wait_for(execute(), timeout=timeout)
    except asyncio.TimeoutError:
        return WebSocketRecoveryResult(
            passed=False,
            failure=f"recovery probe exceeded {timeout:.3f}s timeout",
        )
    except Exception as error:  # noqa: BLE001 - deployment boundary
        return WebSocketRecoveryResult(
            passed=False,
            failure=f"{type(error).__name__}: {error}",
        )


async def run_websocket_gate(
    uri: str,
    *,
    sessions: int = 4,
    audio_seconds: float = 5.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    frame_ms: int = 100,
    timeout: float = 120.0,
    min_commits: int = 0,
    min_audio_acks: int = 1,
    max_dropped_partials: int | None = 0,
    max_backpressure_pauses: int | None = None,
    max_ready_seconds: float | None = None,
    max_total_seconds: float | None = None,
    realtime: bool = False,
    backend: str | None = None,
    model: str | None = None,
    language: str | None = None,
    compute_type: str | None = None,
    auth_token: str | None = None,
    verify_recovery: bool = False,
    recovery_resume_timeout: float = 5.0,
    source_commit: str | None = None,
    probe_audio_path: Path | None = None,
    validity_seconds: float = 86400.0,
) -> WebSocketGateReport:
    """Exercise a deployed WebSocket endpoint without retaining transcript text."""

    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, (int, float))
        or not isfinite(validity_seconds)
        or validity_seconds <= 0
    ):
        raise ValueError("validity_seconds must be finite and positive")
    _validate_uri(uri)
    _validate_options(
        sessions=sessions,
        audio_seconds=audio_seconds,
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
        timeout=timeout,
        min_commits=min_commits,
        min_audio_acks=min_audio_acks,
        max_dropped_partials=max_dropped_partials,
        max_backpressure_pauses=max_backpressure_pauses,
        max_ready_seconds=max_ready_seconds,
        max_total_seconds=max_total_seconds,
        verify_recovery=verify_recovery,
        recovery_resume_timeout=recovery_resume_timeout,
    )
    probe_material = _probe_audio_material(
        audio_seconds=audio_seconds,
        sample_rate=sample_rate,
        channels=channels,
        probe_audio_path=probe_audio_path,
    )
    probe_audio = probe_material.pcm
    results = tuple(await asyncio.gather(*(
        _run_session(
            uri,
            session,
            audio_seconds=audio_seconds,
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
            timeout=timeout,
            min_commits=min_commits,
            min_audio_acks=min_audio_acks,
            max_dropped_partials=max_dropped_partials,
            max_backpressure_pauses=max_backpressure_pauses,
            max_ready_seconds=max_ready_seconds,
            max_total_seconds=max_total_seconds,
            realtime=realtime,
            backend=backend,
            model=model,
            language=language,
            compute_type=compute_type,
            auth_token=auth_token,
            probe_audio=probe_audio,
        )
        for session in range(1, sessions + 1)
    )))
    passed = sum(result.passed for result in results)
    recovery_probe = None
    if verify_recovery:
        recovery_probe = await _run_recovery_probe(
            uri,
            audio_seconds=audio_seconds,
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
            timeout=timeout,
            recovery_resume_timeout=recovery_resume_timeout,
            realtime=realtime,
            backend=backend,
            model=model,
            language=language,
            compute_type=compute_type,
            auth_token=auth_token,
            probe_audio=probe_audio,
        )
    observed_identities = [
        (
            result.backend,
            result.backend_implementation,
            result.model,
            result.model_revision,
            result.device,
            result.language,
            result.compute_type,
            json.dumps(
                result.loaded_models,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for result in results
        if result.passed and result.backend is not None
    ]
    if (
        recovery_probe is not None
        and recovery_probe.passed
        and recovery_probe.backend is not None
    ):
        observed_identities.append((
            recovery_probe.backend,
            recovery_probe.backend_implementation,
            recovery_probe.model,
            recovery_probe.model_revision,
            recovery_probe.device,
            recovery_probe.language,
            recovery_probe.compute_type,
            json.dumps(
                recovery_probe.loaded_models,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ))
    expected_identity_count = sessions + int(verify_recovery)
    identity_consistent = (
        len(observed_identities) == expected_identity_count
        and len(set(observed_identities)) == 1
    )
    observed_identity = (
        observed_identities[0]
        if identity_consistent
        else (None, None, None, None, None, None, None, "")
    )
    loaded_models = (
        results[0].loaded_models
        if results and identity_consistent and results[0].passed
        else ()
    )
    ready = [result.ready_seconds for result in results if result.ready_seconds is not None]
    total = [result.total_seconds for result in results if result.total_seconds is not None]
    created_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return WebSocketGateReport(
        status=(
            "passed"
            if passed == sessions
            and (recovery_probe is None or recovery_probe.passed)
            and identity_consistent
            else "failed"
        ),
        source_commit=source_commit,
        created_at=created_at,
        validity_seconds=round(validity_seconds, 3),
        uri=uri,
        identity_consistent=identity_consistent,
        backend=observed_identity[0],
        backend_implementation=observed_identity[1],
        model=observed_identity[2],
        model_revision=observed_identity[3],
        device=observed_identity[4],
        language=observed_identity[5],
        compute_type=observed_identity[6],
        loaded_models=loaded_models,
        probe_audio_sha256=probe_material.artifact_sha256,
        probe_audio_bytes=probe_material.artifact_bytes,
        probe_audio_rms=round(_rms_s16le(probe_audio), 3),
        sessions=sessions,
        passed_sessions=passed,
        failed_sessions=sessions - passed,
        audio_seconds_per_session=audio_seconds,
        realtime_pacing=realtime,
        min_commits_per_session=min_commits,
        min_audio_acks_per_session=min_audio_acks,
        max_dropped_partials_per_session=max_dropped_partials,
        max_backpressure_pauses_per_session=max_backpressure_pauses,
        max_ready_seconds=max_ready_seconds,
        max_total_seconds=max_total_seconds,
        ready_seconds_p95=_percentile_95(ready),
        total_seconds_p95=_percentile_95(total),
        events=sum(result.events for result in results),
        commits=sum(result.commits for result in results),
        audio_acks=sum(result.audio_acks for result in results),
        backpressure_pauses=sum(
            result.backpressure_pauses for result in results
        ),
        dropped_partials=sum(result.dropped_partials for result in results),
        recovery_probe_required=verify_recovery,
        recovery_probe=recovery_probe,
        results=results,
    )
