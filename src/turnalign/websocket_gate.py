from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from time import perf_counter
from urllib.parse import urlsplit

from .jsonutil import strict_json_object
from .models import TranscriptEvent
from .validation import EventStreamValidator


@dataclass(frozen=True, slots=True)
class WebSocketSessionResult:
    session: int
    passed: bool
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
    uri: str
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
        return sessions_passed and recovery_passed

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
                sent_samples = 0
                while sent_samples < total_samples:
                    if receiver.done():
                        await receiver
                    await flow_allowed.wait()
                    samples = min(frame_samples, total_samples - sent_samples)
                    await websocket.send(bytes(samples * channels * 2))
                    sent_samples += samples
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
        ) -> tuple[str, int, str]:
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
            return ready_session_id, next_audio_sequence, ready_resume_token

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

        first = await connect(uri, max_size=20 * 1024 * 1024)
        try:
            await first.send(json.dumps(start_request(), ensure_ascii=False))
            ready = await receive_json(first)
            session_id, _, resume_token = validate_ready(ready)
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
                await first.send(bytes(samples * channels * 2))
                sent_samples += samples
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
                ) = validate_ready(
                    response,
                    expected_session_id=session_id,
                    expected_resume_token=resume_token,
                )
                if recovered_session_id != session_id or response.get("resumed") is not True:
                    raise ValueError("server did not confirm recovery resume")
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
                await resumed.send(bytes(samples * channels * 2))
                sent_samples += samples
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
        return WebSocketRecoveryResult(
            passed=failure is None,
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
) -> WebSocketGateReport:
    """Exercise a deployed WebSocket endpoint without retaining transcript text."""

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
        )
    ready = [result.ready_seconds for result in results if result.ready_seconds is not None]
    total = [result.total_seconds for result in results if result.total_seconds is not None]
    return WebSocketGateReport(
        status=(
            "passed"
            if passed == sessions and (recovery_probe is None or recovery_probe.passed)
            else "failed"
        ),
        source_commit=source_commit,
        uri=uri,
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
