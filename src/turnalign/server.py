from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import queue
import signal
import threading
from dataclasses import asdict, replace
from http import HTTPStatus
from pathlib import Path
from time import monotonic
from typing import Any, TypeVar

from .audio import file_chunks
from .hints import AsrHints
from .model_pool import BackendPool, BackendPoolCapacityError
from .models import AudioChunk
from .plugins import AsrConfig
from .policy import ServerBusyError, ServerPolicy
from .recovery import (
    RecoveryCapacityError,
    RecoveryConflictError,
    RecoverySession,
    RecoveryStore,
)
from .registry import (
    create_asr,
    create_component,
    supports_session_hints,
    validate_asr_hints,
)
from .resources import close_resources, require_immutable_model_revision
from .session import transcribe_events

LOGGER = logging.getLogger(__name__)
_QueueItem = TypeVar("_QueueItem")


class _OutputBackpressureError(TimeoutError):
    pass


def _queue_output(
    outgoing: queue.Queue[dict[str, object] | None],
    item: dict[str, object] | None,
    cancel_event: threading.Event,
    transport_stats: dict[str, int],
    timeout: float,
) -> bool:
    """Queue an event without allowing a slow peer to stall inference forever."""
    if item is not None and item.get("kind") == "partial" and outgoing.full():
        transport_stats["dropped_partials"] += 1
        return False
    deadline = monotonic() + timeout
    while not cancel_event.is_set():
        remaining = deadline - monotonic()
        if remaining <= 0:
            transport_stats["output_backpressure_timeouts"] += 1
            raise _OutputBackpressureError("output queue remained full")
        try:
            outgoing.put(item, timeout=min(0.1, remaining))
            return True
        except queue.Full:
            if item is not None and item.get("kind") == "partial":
                transport_stats["dropped_partials"] += 1
                return False
    return False


def _force_sender_stop(outgoing: queue.Queue[dict[str, object] | None]) -> None:
    """Make room for the terminal sentinel after cancellation or transport failure."""
    while True:
        try:
            outgoing.put_nowait(None)
            return
        except queue.Full:
            try:
                outgoing.get_nowait()
            except queue.Empty:
                return


async def _async_queue_get(source: queue.Queue[_QueueItem]) -> _QueueItem:
    """Wait for a thread queue without occupying an executor worker."""
    while True:
        try:
            return source.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)


async def _async_queue_put(
    destination: queue.Queue[_QueueItem],
    item: _QueueItem,
    *,
    timeout: float,
    cancel_event: threading.Event,
) -> None:
    """Apply bounded async backpressure without an uncancellable queue.put thread."""
    deadline = monotonic() + timeout
    while not cancel_event.is_set():
        try:
            destination.put_nowait(item)
            return
        except queue.Full:
            if monotonic() >= deadline:
                raise TimeoutError("queue remained full")
            await asyncio.sleep(0.01)
    raise RuntimeError("queue operation cancelled")


def _json(item: object) -> str:
    return json.dumps(
        item,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON number: {value}")


def _control_message(
    message: object,
    *,
    label: str,
    max_bytes: int,
) -> dict[str, object]:
    if not isinstance(message, str):
        raise TypeError(f"{label} must be a JSON object")
    if len(message) > max_bytes or len(message.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    payload = json.loads(
        message,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _bounded_request_float(
    request: dict[str, object],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = request.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise TypeError(f"{key} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{key} must be a number") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _optional_request_string(
    request: dict[str, object],
    key: str,
    default: str | None = None,
) -> str | None:
    raw = request.get(key)
    if raw is None or raw == "":
        return default
    if not isinstance(raw, str):
        raise TypeError(f"{key} must be a string")
    return raw


def _optional_request_number(
    request: dict[str, object],
    key: str,
) -> float | None:
    raw = request.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(raw)


def _request_options(request: dict[str, object], key: str) -> dict[str, Any]:
    raw = request.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(isinstance(name, str) for name in raw):
        raise TypeError(f"{key} must be a JSON object with string keys")
    return dict(raw)


def _bounded_request_int(
    request: dict[str, object],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = request.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{key} must be an integer")
    if not minimum <= raw <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return raw


def _recovery_config_key(
    request: dict[str, object],
    *,
    backend: str,
    model: str | None,
    device: str,
    internal_chunk_ms: int,
) -> str:
    excluded = {
        "auth",
        "resume_token",
        "resume_session_id",
        "acknowledged_event_sequence",
        "type",
    }
    effective = {
        key: value for key, value in request.items()
        if key not in excluded
    }
    effective.update({
        "backend": backend,
        "model": model,
        "device": device,
        "internal_chunk_ms": internal_chunk_ms,
    })
    encoded = json.dumps(
        effective,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    default_backend: str = "transformers-whisper",
    default_model: str | None = None,
    default_device: str = "auto",
    default_language: str | None = None,
    default_compute_type: str | None = None,
    default_executable: str | None = None,
    default_model_path: str | None = None,
    default_backend_options: dict[str, object] | None = None,
    warmup_file: Path | None = None,
    ffmpeg: str = "ffmpeg",
    policy: ServerPolicy | None = None,
    internal_chunk_ms: int = 100,
    initialization_timeout: float = 120.0,
    finalization_timeout: float = 120.0,
    worker_shutdown_timeout: float = 5.0,
    output_backpressure_timeout: float = 5.0,
    max_recovery_events: int = 2_048,
    max_recovery_event_bytes: int = 512 * 1024,
    max_recovery_event_bytes_per_session: int = 8 * 1024 * 1024,
    max_recovery_sessions: int = 32,
    max_recovery_audio_bytes: int = 512 * 1024 * 1024,
    max_recovery_total_bytes: int = 2 * 1024 * 1024 * 1024,
    recovery_ttl_seconds: float = 300.0,
    max_control_message_bytes: int = 64 * 1024,
    max_concurrent_sessions: int = 32,
    start_timeout: float = 10.0,
    client_idle_timeout: float = 60.0,
    backend_replicas: int = 1,
    preload: bool = False,
    require_immutable_revision: bool = False,
    allowed_origins: tuple[str | None, ...] = (None,),
    shutdown_event: asyncio.Event | None = None,
    shutdown_grace_timeout: float = 30.0,
) -> None:
    policy = policy or ServerPolicy.defaults(default_backend, default_model)
    policy.validate_bind(host)
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    if not 20 <= internal_chunk_ms <= 100:
        raise ValueError("internal_chunk_ms must be between 20 and 100")
    for name, value in (
        ("initialization_timeout", initialization_timeout),
        ("finalization_timeout", finalization_timeout),
        ("worker_shutdown_timeout", worker_shutdown_timeout),
        ("output_backpressure_timeout", output_backpressure_timeout),
        ("start_timeout", start_timeout),
        ("client_idle_timeout", client_idle_timeout),
        ("shutdown_grace_timeout", shutdown_grace_timeout),
        ("recovery_ttl_seconds", recovery_ttl_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_recovery_events <= 0:
        raise ValueError("max_recovery_events must be positive")
    if max_recovery_event_bytes <= 0:
        raise ValueError("max_recovery_event_bytes must be positive")
    if max_recovery_event_bytes_per_session <= 0:
        raise ValueError("max_recovery_event_bytes_per_session must be positive")
    if max_recovery_event_bytes > max_recovery_event_bytes_per_session:
        raise ValueError(
            "max_recovery_event_bytes cannot exceed "
            "max_recovery_event_bytes_per_session"
        )
    if max_recovery_sessions <= 0:
        raise ValueError("max_recovery_sessions must be positive")
    if max_recovery_audio_bytes <= 0:
        raise ValueError("max_recovery_audio_bytes must be positive")
    if max_recovery_total_bytes <= 0:
        raise ValueError("max_recovery_total_bytes must be positive")
    if max_recovery_audio_bytes > max_recovery_total_bytes:
        raise ValueError(
            "max_recovery_audio_bytes cannot exceed max_recovery_total_bytes"
        )
    if (
        isinstance(max_control_message_bytes, bool)
        or not isinstance(max_control_message_bytes, int)
        or not 1 <= max_control_message_bytes <= 1024 * 1024
    ):
        raise ValueError("max_control_message_bytes must be between 1 and 1048576")
    if max_concurrent_sessions <= 0:
        raise ValueError("max_concurrent_sessions must be positive")
    if not 1 <= backend_replicas <= 8:
        raise ValueError("backend_replicas must be between 1 and 8")
    if not isinstance(require_immutable_revision, bool):
        raise TypeError("require_immutable_revision must be a boolean")
    if not allowed_origins or any(
        origin is not None and (not isinstance(origin, str) or not origin.strip())
        for origin in allowed_origins
    ):
        raise ValueError("allowed_origins must contain None or non-empty origins")
    try:
        from websockets.asyncio.server import serve as websocket_serve
        from websockets.exceptions import ConnectionClosed
    except ImportError as error:
        raise RuntimeError("WebSocket support requires: pip install 'turnalign[server]'") from error
    backend_pool = BackendPool(max_entries=8, max_entries_per_key=backend_replicas)
    recovery_store = RecoveryStore(
        max_sessions=max_recovery_sessions,
        max_events_per_session=max_recovery_events,
        max_event_bytes=max_recovery_event_bytes,
        max_event_bytes_per_session=max_recovery_event_bytes_per_session,
        max_audio_bytes_per_session=max_recovery_audio_bytes,
        max_total_audio_bytes=max_recovery_total_bytes,
    )
    active_session_lock = asyncio.Lock()
    active_sessions = 0
    handler_tasks: set[asyncio.Task[object]] = set()
    active_connections: set[Any] = set()
    recovery_sweeper: asyncio.Task[None] | None = None
    loop = asyncio.get_running_loop()
    owns_shutdown_event = shutdown_event is None
    shutdown_event = shutdown_event or asyncio.Event()
    sigterm_handler_installed = False
    previous_sigterm_handler = None

    if owns_shutdown_event:
        try:
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
            loop.add_signal_handler(signal.SIGTERM, shutdown_event.set)
            sigterm_handler_installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            # Signal handlers are unavailable on Windows and non-main threads.
            pass

    if preload:
        preload_config = AsrConfig(
            model=default_model,
            device=default_device,
            language=default_language,
            compute_type=default_compute_type,
            executable=default_executable,
            model_path=default_model_path,
            extra=dict(default_backend_options or {}),
        )
        preload_keys: list[str] = []

        def create_preloaded_backend():
            backend = create_asr(default_backend, preload_config)
            try:
                if require_immutable_revision:
                    require_immutable_model_revision(backend)
                if warmup_file is not None:
                    list(backend.transcribe(file_chunks(warmup_file, ffmpeg=ffmpeg)))
            except BaseException:
                close_resources(
                    (backend,),
                    logger=LOGGER,
                    reason="preload validation or warmup failure",
                )
                raise
            return backend

        try:
            for _ in range(backend_replicas):
                key, _backend = await asyncio.to_thread(
                    backend_pool.acquire,
                    default_backend,
                    preload_config,
                    create_preloaded_backend,
                )
                preload_keys.append(key)
        except BaseException:
            backend_pool.close()
            recovery_store.close()
            raise
        finally:
            for key in preload_keys:
                backend_pool.release(key)

    async def session_handler(websocket) -> None:
        thread: threading.Thread | None = None
        worker_started = False
        sender: asyncio.Task[None] | None = None
        cancel_event = threading.Event()
        incoming: queue.Queue[AudioChunk | None] | None = None
        recovery_session: RecoverySession | None = None
        session_completed = False
        backend_in_use: object | None = None
        backend_lease_lock = threading.Lock()

        def cancel_worker() -> None:
            cancel_event.set()
            with backend_lease_lock:
                backend = backend_in_use
                cancel = getattr(backend, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        LOGGER.warning("backend cancellation failed", exc_info=True)

        def stop_input() -> None:
            if incoming is None:
                return
            try:
                incoming.put_nowait(None)
            except queue.Full:
                try:
                    incoming.get_nowait()
                except queue.Empty:
                    pass
                incoming.put_nowait(None)

        try:
            try:
                first = await asyncio.wait_for(websocket.recv(), timeout=start_timeout)
            except asyncio.TimeoutError as error:
                raise TimeoutError("start message timed out") from error
            request = _control_message(
                first,
                label="start message",
                max_bytes=max_control_message_bytes,
            )
            if request.get("type") != "start":
                raise ValueError("first message type must be start")
            backend_name, model_name, language, compute_type = policy.validate_start(
                request,
                default_backend=default_backend,
                default_model=default_model,
                default_language=default_language,
                default_compute_type=default_compute_type,
            )
            raw_hotwords = request.get("hotwords") or []
            if isinstance(raw_hotwords, str):
                raw_hotwords = [raw_hotwords]
            if not isinstance(raw_hotwords, list) or not all(
                isinstance(item, str) for item in raw_hotwords
            ):
                raise ValueError("hotwords must be a JSON array of strings")
            sample_rate = _bounded_request_int(
                request, "sample_rate", 16_000, minimum=8_000, maximum=96_000
            )
            channels = _bounded_request_int(
                request, "channels", 1, minimum=1, maximum=8
            )
            vad_threshold = _bounded_request_float(
                request, "vad_threshold", 0.012, minimum=0.0, maximum=1.0
            )
            silence_seconds = _bounded_request_float(
                request, "silence_seconds", 0.7, minimum=0.1, maximum=10.0
            )
            max_utterance_seconds = _bounded_request_float(
                request,
                "max_utterance_seconds",
                20.0,
                minimum=1.0,
                maximum=300.0,
            )
            partial_seconds = _bounded_request_float(
                request, "partial_seconds", 2.0, minimum=0.1, maximum=30.0
            )
            if partial_seconds > max_utterance_seconds:
                raise ValueError(
                    "partial_seconds must not exceed max_utterance_seconds"
                )
            config = AsrConfig(
                model=model_name,
                device=default_device,
                language=language,
                compute_type=compute_type,
                executable=_optional_request_string(
                    request, "executable", default_executable
                ),
                model_path=_optional_request_string(
                    request, "model_path", default_model_path
                ),
                extra=dict(default_backend_options or {}),
                hints=AsrHints(
                    hotwords=tuple(raw_hotwords),
                    context=_optional_request_string(request, "context"),
                    boost=_optional_request_number(request, "hotword_boost"),
                ),
            )
            try:
                validate_asr_hints(backend_name, config.hints)
            except LookupError:
                # Test doubles and dynamically supplied factories are validated again
                # against the acquired implementation below.
                pass
            session_hints = supports_session_hints(backend_name)
            pooled_config = (
                replace(config, hints=AsrHints())
                if session_hints else config
            )
            requested_session_id = request.get("resume_session_id")
            if requested_session_id is not None and not isinstance(requested_session_id, str):
                raise TypeError("resume_session_id must be a string")
            raw_resume_token = request.get("resume_token")
            if raw_resume_token is None:
                resume_token: str | None = None
            elif isinstance(raw_resume_token, str):
                resume_token = raw_resume_token
            else:
                raise TypeError("resume_token must be a string")
            if requested_session_id is not None and not resume_token:
                raise PermissionError("recovery authentication failed")
            if requested_session_id is None and resume_token is not None:
                raise ValueError("resume_token requires resume_session_id")
            acknowledged_event_sequence = _bounded_request_int(
                request,
                "acknowledged_event_sequence",
                -1,
                minimum=-1,
                maximum=2**63 - 1,
            )
            recovery_session, resumed = recovery_store.open(
                _recovery_config_key(
                    request,
                    backend=backend_name,
                    model=model_name,
                    device=default_device,
                    internal_chunk_ms=internal_chunk_ms,
                ),
                requested_session_id,
                resume_token,
            )
            replay_audio_start = recovery_session.transcribed_through
            replay_audio_end = recovery_session.timeline.end
            incoming = queue.Queue(maxsize=128)
            outgoing: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=128)
            initialized: queue.Queue[dict[str, object] | BaseException] = queue.Queue(maxsize=1)
            session_id = recovery_session.session_id
            transport_stats = {
                "frames": 0,
                "bytes": 0,
                "internal_chunks": 0,
                "queue_peak": 0,
                "backpressure_pauses": 0,
                "dropped_partials": 0,
                "output_backpressure_timeouts": 0,
            }
            flow_state = {"paused": False}

            def chunks():
                if resumed and replay_audio_end > replay_audio_start:
                    yield from recovery_session.timeline.iter_chunks(
                        internal_chunk_ms,
                        start=replay_audio_start,
                        end=replay_audio_end,
                    )
                while True:
                    item = incoming.get()
                    if item is None or cancel_event.is_set():
                        return
                    if flow_state["paused"] and incoming.qsize() <= 32:
                        flow_state["paused"] = False
                        emit({
                            "type": "flow_control",
                            "action": "resume",
                            "queue_depth": incoming.qsize(),
                        })
                    yield item

            def emit(item: dict[str, object] | None) -> None:
                if item is None and cancel_event.is_set():
                    _force_sender_stop(outgoing)
                    return
                _queue_output(
                    outgoing,
                    item,
                    cancel_event,
                    transport_stats,
                    output_backpressure_timeout,
                )

            def worker() -> None:
                nonlocal backend_in_use, session_completed
                announced = False
                pool_key: str | None = None
                try:
                    def create_and_warmup():
                        nonlocal backend_in_use
                        backend = create_asr(backend_name, pooled_config)
                        with backend_lease_lock:
                            backend_in_use = backend
                        try:
                            if require_immutable_revision:
                                require_immutable_model_revision(backend)
                            if warmup_file is not None:
                                list(backend.transcribe(file_chunks(warmup_file, ffmpeg=ffmpeg)))
                        except BaseException:
                            with backend_lease_lock:
                                if backend_in_use is backend:
                                    backend_in_use = None
                            close_resources(
                                (backend,),
                                logger=LOGGER,
                                reason="backend validation or warmup failure",
                            )
                            raise
                        return backend

                    pool_key, backend = backend_pool.acquire(
                        backend_name,
                        pooled_config,
                        create_and_warmup,
                        cancel_event,
                    )
                    with backend_lease_lock:
                        backend_in_use = backend
                    validate_asr_hints(backend_name, config.hints, backend)
                    if session_hints:
                        set_hints = getattr(backend, "set_hints", None)
                        if not callable(set_hints):
                            raise RuntimeError(
                                "backend declares session hints without set_hints()"
                            )
                        set_hints(config.hints)
                    components = []
                    try:
                        aligner_name = request.get("aligner")
                        diarizer_name = request.get("diarizer")
                        online_diarizer_name = request.get("online_diarizer")
                        aligner = create_component(
                            "alignment",
                            str(aligner_name),
                            _request_options(request, "aligner_options"),
                        ) if aligner_name else None
                        if aligner is not None:
                            components.append(aligner)
                        diarizer = create_component(
                            "diarization",
                            str(diarizer_name),
                            _request_options(request, "diarizer_options"),
                        ) if diarizer_name else None
                        if diarizer is not None:
                            components.append(diarizer)
                        online_diarizer = create_component(
                            "online_diarization",
                            str(online_diarizer_name),
                            _request_options(request, "online_diarizer_options"),
                        ) if online_diarizer_name else None
                    except BaseException:
                        close_resources(
                            components,
                            logger=LOGGER,
                            reason="component initialization failure",
                        )
                        raise
                    initialized.put({
                        "capabilities": asdict(backend.capabilities),
                        "backend_name": backend.name,
                    })
                    announced = True
                    for event in transcribe_events(
                        chunks(),
                        backend,
                        live=True,
                        vad_threshold=vad_threshold,
                        silence_seconds=silence_seconds,
                        max_utterance_seconds=max_utterance_seconds,
                        partial_seconds=partial_seconds,
                        aligner=aligner,
                        diarizer=diarizer,
                        online_diarizer=online_diarizer,
                        cancel_event=cancel_event,
                        close_backend=False,
                        segment_index_start=recovery_session.next_segment_index,
                    ):
                        if event.kind == "end":
                            event.metadata.update(
                                {
                                    "websocket_frames": transport_stats["frames"],
                                    "websocket_bytes": transport_stats["bytes"],
                                    "input_queue_peak": transport_stats["queue_peak"],
                                    "internal_chunks": transport_stats["internal_chunks"],
                                    "backpressure_pauses": transport_stats[
                                        "backpressure_pauses"
                                    ],
                                    "dropped_partials": transport_stats[
                                        "dropped_partials"
                                    ],
                                    "output_backpressure_timeouts": transport_stats[
                                        "output_backpressure_timeouts"
                                    ],
                                }
                            )
                            session_completed = True
                        event.source_timestamp = event.end
                        payload = recovery_store.append_event(recovery_session, event)
                        emit(payload)
                        if event.kind == "end":
                            LOGGER.info(
                                "session_complete session_id=%s frames=%d bytes=%d "
                                "dropped_partials=%d backpressure_pauses=%d",
                                session_id,
                                transport_stats["frames"],
                                transport_stats["bytes"],
                                transport_stats["dropped_partials"],
                                transport_stats["backpressure_pauses"],
                            )
                except Exception as error:  # noqa: BLE001 - plugin boundary
                    if announced:
                        try:
                            emit(policy.public_error(error))
                        except _OutputBackpressureError:
                            LOGGER.warning(
                                "unable to report session failure because output is stalled"
                            )
                    else:
                        initialized.put(error)
                finally:
                    with backend_lease_lock:
                        leased_backend = backend_in_use
                        if session_hints and leased_backend is not None:
                            set_hints = getattr(leased_backend, "set_hints", None)
                            if callable(set_hints):
                                try:
                                    set_hints(AsrHints())
                                except Exception:
                                    LOGGER.warning(
                                        "unable to clear backend session hints",
                                        exc_info=True,
                                    )
                        backend_in_use = None
                        if pool_key is not None:
                            if config.hints.active and not session_hints:
                                backend_pool.discard(pool_key)
                            else:
                                backend_pool.release(pool_key)
                    try:
                        emit(None)
                    except _OutputBackpressureError:
                        cancel_event.set()
                        _force_sender_stop(outgoing)
                    recovery_store.release(
                        recovery_session,
                        completed=session_completed,
                    )

            thread = threading.Thread(
                target=worker,
                name=f"turnalign-session-{session_id[:8]}",
                daemon=True,
            )
            thread.start()
            worker_started = True
            try:
                initialization = await asyncio.wait_for(
                    _async_queue_get(initialized),
                    timeout=initialization_timeout,
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError("ASR initialization timed out") from error
            if isinstance(initialization, BaseException):
                raise initialization

            public_config: dict[str, object] = {
                "model": config.model,
                "device": config.device,
                "language": config.language,
                "compute_type": config.compute_type,
            }
            public_config = {key: value for key, value in public_config.items() if value is not None}
            if config.hints.active:
                public_config["hints"] = config.hints.private_metadata("backend-selected")
            await websocket.send(_json({
                "type": "ready",
                "protocol_version": 1,
                "session_id": session_id,
                "resume_token": recovery_session.resume_token,
                "resumed": resumed,
                "model_loaded": True,
                "backend": backend_name,
                "sample_rate": sample_rate,
                "channels": channels,
                "config": public_config,
                "capabilities": initialization["capabilities"],
                "next_audio_sequence": recovery_session.next_audio_sequence,
                "next_event_sequence": recovery_session.next_event_sequence,
            }))
            LOGGER.info(
                "session_ready session_id=%s backend=%s resumed=%s",
                session_id,
                backend_name,
                resumed,
            )

            async def send_events() -> None:
                while True:
                    item = await _async_queue_get(outgoing)
                    if item is None:
                        return
                    await websocket.send(_json(item))

            sender = asyncio.create_task(send_events())
            for replay_event in recovery_store.replay_after(
                recovery_session,
                acknowledged_event_sequence,
            ):
                await _async_queue_put(
                    outgoing,
                    replay_event,
                    timeout=output_backpressure_timeout,
                    cancel_event=cancel_event,
                )

            position = recovery_session.timeline.end
            frame_buffer = bytearray()
            internal_size = max(
                2 * channels,
                round(sample_rate * internal_chunk_ms / 1000) * channels * 2,
            )

            async def enqueue(data: bytes, *, final: bool = False) -> int:
                nonlocal position
                item = AudioChunk(data, position, sample_rate, channels, is_final=final)
                if incoming.qsize() >= 96 and not flow_state["paused"]:
                    flow_state["paused"] = True
                    transport_stats["backpressure_pauses"] += 1
                    await websocket.send(_json({
                        "type": "flow_control",
                        "action": "pause",
                        "queue_depth": incoming.qsize(),
                    }))
                # Recovery storage is the acceptance boundary. Never expose a
                # chunk to the inference thread before it can be replayed.
                audio_sequence = recovery_store.append_audio(recovery_session, item)
                try:
                    incoming.put_nowait(item)
                except queue.Full:
                    try:
                        await _async_queue_put(
                            incoming,
                            item,
                            timeout=1.0,
                            cancel_event=cancel_event,
                        )
                    except TimeoutError as error:
                        raise RuntimeError(
                            "audio input queue remained full; client must honor flow control"
                        ) from error
                transport_stats["internal_chunks"] += 1
                transport_stats["queue_peak"] = max(
                    transport_stats["queue_peak"], incoming.qsize()
                )
                position += item.duration
                return audio_sequence

            end_requested = False
            last_audio_sequence = (
                recovery_session.next_audio_sequence - 1
                if recovery_session.next_audio_sequence > 0 else None
            )

            async def send_audio_ack() -> None:
                payload: dict[str, object] = {
                    "type": "audio_ack",
                    "session_id": session_id,
                    "buffered_bytes": len(frame_buffer),
                }
                if last_audio_sequence is not None:
                    payload["acknowledged_sequence"] = last_audio_sequence
                await websocket.send(_json(payload))

            try:
                while True:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=client_idle_timeout,
                        )
                    except asyncio.TimeoutError as error:
                        raise TimeoutError("client became idle") from error
                    if isinstance(message, bytes):
                        if len(message) % (2 * channels):
                            raise ValueError("binary frame must contain complete signed 16-bit PCM frames")
                        if len(message) > sample_rate * channels * 2 * 10:
                            raise ValueError("binary frame exceeds 10 seconds")
                        transport_stats["frames"] += 1
                        transport_stats["bytes"] += len(message)
                        frame_buffer.extend(message)
                        while len(frame_buffer) >= internal_size:
                            data = bytes(frame_buffer[:internal_size])
                            del frame_buffer[:internal_size]
                            last_audio_sequence = await enqueue(data)
                        await send_audio_ack()
                        buffered_end = position + len(frame_buffer) / (
                            sample_rate * channels * 2
                        )
                        if buffered_end > policy.max_session_seconds:
                            raise ValueError("maximum session duration exceeded")
                        continue
                    control = _control_message(
                        message,
                        label="control message",
                        max_bytes=max_control_message_bytes,
                    )
                    if control.get("type") == "end":
                        end_requested = True
                        if frame_buffer:
                            last_audio_sequence = await enqueue(
                                bytes(frame_buffer), final=True
                            )
                            frame_buffer.clear()
                            await send_audio_ack()
                        break
                    if control.get("type") == "cancel":
                        session_completed = True
                        cancel_worker()
                        break
                    if control.get("type") == "ping":
                        await websocket.send(_json({"type": "pong"}))
                        continue
                    raise ValueError(f"unsupported control message: {control.get('type')}")
            finally:
                if not end_requested:
                    cancel_worker()
                stop_input()
            if end_requested:
                try:
                    await asyncio.wait_for(sender, timeout=finalization_timeout)
                except asyncio.TimeoutError as error:
                    cancel_worker()
                    raise TimeoutError("ASR finalization timed out") from error
            else:
                _force_sender_stop(outgoing)
                try:
                    await asyncio.wait_for(sender, timeout=worker_shutdown_timeout)
                except asyncio.TimeoutError:
                    sender.cancel()
        except Exception as error:
            cancel_worker()
            stop_input()
            if isinstance(error, ConnectionClosed):
                LOGGER.info("WebSocket client disconnected")
            elif isinstance(error, (
                PermissionError,
                BackendPoolCapacityError,
                RecoveryCapacityError,
                RecoveryConflictError,
                TimeoutError,
                TypeError,
                ValueError,
                KeyError,
                LookupError,
            )):
                LOGGER.info("WebSocket request rejected: %s", type(error).__name__)
            else:
                LOGGER.exception("WebSocket session failed")
            try:
                await websocket.send(_json(policy.public_error(error)))
            except Exception:
                LOGGER.debug("unable to send WebSocket error to disconnected peer", exc_info=True)
        finally:
            cancel_worker()
            stop_input()
            if sender is not None and not sender.done():
                sender.cancel()
            if thread is not None and thread.is_alive():
                await asyncio.to_thread(thread.join, worker_shutdown_timeout)
            if recovery_session is not None and (
                not worker_started or thread is None or not thread.is_alive()
            ):
                recovery_store.release(recovery_session, completed=session_completed)

    async def handler(websocket) -> None:
        nonlocal active_sessions
        task = asyncio.current_task()
        admitted = False
        if task is not None:
            handler_tasks.add(task)
        active_connections.add(websocket)
        try:
            async with active_session_lock:
                at_capacity = active_sessions >= max_concurrent_sessions
                if not at_capacity:
                    active_sessions += 1
                    admitted = True
            if at_capacity:
                await websocket.send(_json(policy.public_error(ServerBusyError())))
                return
            await session_handler(websocket)
        finally:
            if admitted:
                async with active_session_lock:
                    active_sessions -= 1
            if task is not None:
                handler_tasks.discard(task)
            active_connections.discard(websocket)

    def process_request(connection, request):
        path = request.path.partition("?")[0]
        if path in {"/healthz", "/readyz"}:
            payload: dict[str, object] = {"status": "ok"}
            if path == "/readyz":
                payload.update({"ready": True, "preloaded": preload})
            response = connection.respond(
                HTTPStatus.OK,
                json.dumps(payload, separators=(",", ":")) + "\n",
            )
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            response.headers["Cache-Control"] = "no-store"
            return response
        origin_headers = request.headers.get_all("Origin")
        origin = origin_headers[0] if len(origin_headers) == 1 else None
        if len(origin_headers) > 1 or origin not in allowed_origins:
            response = connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
            response.headers["Cache-Control"] = "no-store"
            return response
        return None

    async def sweep_recovery_sessions() -> None:
        interval = min(60.0, max(0.1, recovery_ttl_seconds / 2))
        while True:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                removed = recovery_store.prune_expired(recovery_ttl_seconds)
                if removed:
                    LOGGER.info("expired %d inactive recovery sessions", removed)

    websocket_server = None
    try:
        if shutdown_event.is_set():
            return
        websocket_server = await websocket_serve(
            handler,
            host,
            port,
            max_size=20 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            compression=None,
            # Origin is checked in process_request to reject normally rather
            # than emitting an exception traceback for expected hostile input.
            origins=None,
            process_request=process_request,
            server_header=None,
        )
        LOGGER.info(
            "server_listening host=%s port=%d backend=%s replicas=%d preload=%s",
            host,
            port,
            default_backend,
            backend_replicas,
            preload,
        )
        recovery_sweeper = asyncio.create_task(sweep_recovery_sessions())
        await shutdown_event.wait()
        LOGGER.info("server_shutdown_requested active_sessions=%d", active_sessions)
    finally:
        if recovery_sweeper is not None:
            recovery_sweeper.cancel()
            try:
                await recovery_sweeper
            except asyncio.CancelledError:
                pass
        if websocket_server is not None:
            async def close_server() -> None:
                # websockets 14 doesn't accept close code/reason on Server.close().
                # Stop admission first, then close peers explicitly with 1012.
                websocket_server.close(close_connections=False)
                peers = tuple(active_connections)
                if peers:
                    await asyncio.gather(*(
                        peer.close(code=1012, reason="service restart")
                        for peer in peers
                    ), return_exceptions=True)
                await websocket_server.wait_closed()

            try:
                await asyncio.wait_for(
                    close_server(),
                    timeout=shutdown_grace_timeout,
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "graceful shutdown exceeded %.3f seconds; cancelling %d handlers",
                    shutdown_grace_timeout,
                    len(handler_tasks),
                )
                pending_handlers = tuple(handler_tasks)
                for task in pending_handlers:
                    task.cancel()
                if pending_handlers:
                    await asyncio.gather(*pending_handlers, return_exceptions=True)
                await websocket_server.wait_closed()
        if sigterm_handler_installed:
            loop.remove_signal_handler(signal.SIGTERM)
            if previous_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
        backend_pool.close()
        recovery_store.close()
        LOGGER.info("server_stopped")
