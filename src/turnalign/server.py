from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import queue
import threading
from dataclasses import asdict
from pathlib import Path

from .audio import file_chunks
from .hints import AsrHints
from .model_pool import BackendPool
from .models import AudioChunk
from .plugins import AsrConfig
from .policy import ServerPolicy
from .recovery import RecoverySession, RecoveryStore
from .registry import create_asr, create_component
from .session import transcribe_events

LOGGER = logging.getLogger(__name__)


def _json(item: object) -> str:
    return json.dumps(item, ensure_ascii=False)


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
    warmup_file: Path | None = None,
    ffmpeg: str = "ffmpeg",
    policy: ServerPolicy | None = None,
    internal_chunk_ms: int = 100,
    initialization_timeout: float = 120.0,
    worker_shutdown_timeout: float = 5.0,
) -> None:
    policy = policy or ServerPolicy.defaults(default_backend, default_model)
    policy.validate_bind(host)
    if not 20 <= internal_chunk_ms <= 100:
        raise ValueError("internal_chunk_ms must be between 20 and 100")
    try:
        from websockets.asyncio.server import serve as websocket_serve
    except ImportError as error:
        raise RuntimeError("WebSocket support requires: pip install 'turnalign[server]'") from error
    backend_pool = BackendPool()
    recovery_store = RecoveryStore()

    async def handler(websocket) -> None:
        thread: threading.Thread | None = None
        sender: asyncio.Task[None] | None = None
        cancel_event = threading.Event()
        incoming: queue.Queue[AudioChunk | None] | None = None
        recovery_session: RecoverySession | None = None
        session_completed = False

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
            first = await websocket.recv()
            if not isinstance(first, str):
                raise TypeError("first frame must be start JSON")
            request = json.loads(first)
            if not isinstance(request, dict):
                raise TypeError("start message must be a JSON object")
            if request.get("type") != "start":
                raise ValueError("first message type must be start")
            backend_name, model_name = policy.validate_start(
                request,
                default_backend=default_backend,
                default_model=default_model,
            )
            raw_hotwords = request.get("hotwords") or []
            if isinstance(raw_hotwords, str):
                raw_hotwords = [raw_hotwords]
            if not isinstance(raw_hotwords, list) or not all(
                isinstance(item, str) for item in raw_hotwords
            ):
                raise ValueError("hotwords must be a JSON array of strings")
            sample_rate = int(request.get("sample_rate", 16_000))
            channels = int(request.get("channels", 1))
            if not 8_000 <= sample_rate <= 96_000:
                raise ValueError("sample_rate must be between 8000 and 96000")
            if not 1 <= channels <= 8:
                raise ValueError("channels must be between 1 and 8")
            config = AsrConfig(
                model=model_name,
                device=default_device,
                language=request.get("language"),
                compute_type=request.get("compute_type"),
                executable=request.get("executable"),
                model_path=request.get("model_path"),
                hints=AsrHints(
                    hotwords=tuple(raw_hotwords),
                    context=request.get("context"),
                    boost=request.get("hotword_boost"),
                ),
            )
            requested_session_id = request.get("resume_session_id")
            if requested_session_id is not None and not isinstance(requested_session_id, str):
                raise TypeError("resume_session_id must be a string")
            acknowledged_event_sequence = int(request.get("acknowledged_event_sequence", -1))
            if acknowledged_event_sequence < -1:
                raise ValueError("acknowledged_event_sequence must be at least -1")
            recovery_session, resumed = recovery_store.open(
                _recovery_config_key(
                    request,
                    backend=backend_name,
                    model=model_name,
                    device=default_device,
                    internal_chunk_ms=internal_chunk_ms,
                ),
                requested_session_id,
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
                if item is not None and item.get("kind") == "partial" and outgoing.full():
                    transport_stats["dropped_partials"] += 1
                    return
                if item is None and cancel_event.is_set():
                    try:
                        outgoing.put_nowait(None)
                    except queue.Full:
                        try:
                            outgoing.get_nowait()
                        except queue.Empty:
                            pass
                        outgoing.put_nowait(None)
                    return
                while not cancel_event.is_set():
                    try:
                        outgoing.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        if item is not None and item.get("kind") == "partial":
                            transport_stats["dropped_partials"] += 1
                            return
                        if cancel_event.is_set():
                            return

            def worker() -> None:
                nonlocal session_completed
                announced = False
                pool_key: str | None = None
                try:
                    def create_and_warmup():
                        backend = create_asr(backend_name, config)
                        if warmup_file is not None:
                            try:
                                list(backend.transcribe(file_chunks(warmup_file, ffmpeg=ffmpeg)))
                            except Exception:
                                backend.close()
                                raise
                        return backend

                    pool_key, backend = backend_pool.acquire(
                        backend_name,
                        config,
                        create_and_warmup,
                        cancel_event,
                    )
                    aligner_name = request.get("aligner")
                    diarizer_name = request.get("diarizer")
                    online_diarizer_name = request.get("online_diarizer")
                    aligner = create_component(
                        "alignment", str(aligner_name), request.get("aligner_options") or {}
                    ) if aligner_name else None
                    diarizer = create_component(
                        "diarization", str(diarizer_name), request.get("diarizer_options") or {}
                    ) if diarizer_name else None
                    online_diarizer = create_component(
                        "online_diarization",
                        str(online_diarizer_name),
                        request.get("online_diarizer_options") or {},
                    ) if online_diarizer_name else None
                    initialized.put({
                        "capabilities": asdict(backend.capabilities),
                        "backend_name": backend.name,
                    })
                    announced = True
                    for event in transcribe_events(
                        chunks(),
                        backend,
                        live=True,
                        vad_threshold=float(request.get("vad_threshold", 0.012)),
                        silence_seconds=float(request.get("silence_seconds", 0.7)),
                        max_utterance_seconds=float(
                            request.get("max_utterance_seconds", 20.0)
                        ),
                        partial_seconds=float(request.get("partial_seconds", 2.0)),
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
                                }
                            )
                            session_completed = True
                        event.source_timestamp = event.end
                        payload = recovery_store.append_event(recovery_session, event)
                        emit(payload)
                except Exception as error:  # noqa: BLE001 - plugin boundary
                    if announced:
                        emit(policy.public_error(error))
                    else:
                        initialized.put(error)
                finally:
                    if pool_key is not None:
                        backend_pool.release(pool_key)
                    emit(None)

            thread = threading.Thread(
                target=worker,
                name=f"turnalign-session-{session_id[:8]}",
                daemon=True,
            )
            thread.start()
            try:
                initialization = await asyncio.wait_for(
                    asyncio.to_thread(initialized.get),
                    timeout=initialization_timeout,
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError("ASR initialization timed out") from error
            if isinstance(initialization, BaseException):
                raise initialization

            public_config = {
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

            async def send_events() -> None:
                while True:
                    item = await asyncio.to_thread(outgoing.get)
                    if item is None:
                        return
                    await websocket.send(_json(item))

            sender = asyncio.create_task(send_events())
            for replay_event in recovery_store.replay_after(
                recovery_session,
                acknowledged_event_sequence,
            ):
                await asyncio.to_thread(outgoing.put, replay_event)

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
                try:
                    incoming.put_nowait(item)
                except queue.Full:
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(incoming.put, item),
                            timeout=1.0,
                        )
                    except asyncio.TimeoutError as error:
                        raise RuntimeError(
                            "audio input queue remained full; client must honor flow control"
                        ) from error
                audio_sequence = recovery_store.append_audio(recovery_session, item)
                transport_stats["internal_chunks"] += 1
                transport_stats["queue_peak"] = max(
                    transport_stats["queue_peak"], incoming.qsize()
                )
                position += item.duration
                return audio_sequence

            end_requested = False
            try:
                async for message in websocket:
                    if isinstance(message, bytes):
                        if len(message) % (2 * channels):
                            raise ValueError("binary frame must contain complete signed 16-bit PCM frames")
                        if len(message) > sample_rate * channels * 2 * 10:
                            raise ValueError("binary frame exceeds 10 seconds")
                        transport_stats["frames"] += 1
                        transport_stats["bytes"] += len(message)
                        frame_buffer.extend(message)
                        last_audio_sequence = None
                        while len(frame_buffer) >= internal_size:
                            data = bytes(frame_buffer[:internal_size])
                            del frame_buffer[:internal_size]
                            last_audio_sequence = await enqueue(data)
                        await websocket.send(_json({
                            "type": "audio_ack",
                            "session_id": session_id,
                            "acknowledged_sequence": last_audio_sequence,
                            "buffered_bytes": len(frame_buffer),
                        }))
                        if position > policy.max_session_seconds:
                            raise ValueError("maximum session duration exceeded")
                        continue
                    control = json.loads(message)
                    if control.get("type") == "end":
                        end_requested = True
                        if frame_buffer:
                            await enqueue(bytes(frame_buffer), final=True)
                            frame_buffer.clear()
                        break
                    if control.get("type") == "cancel":
                        session_completed = True
                        cancel_event.set()
                        break
                    if control.get("type") == "ping":
                        await websocket.send(_json({"type": "pong"}))
                        continue
                    raise ValueError(f"unsupported control message: {control.get('type')}")
            finally:
                if not end_requested:
                    cancel_event.set()
                stop_input()
            await sender
        except Exception as error:  # noqa: BLE001 - connection boundary
            cancel_event.set()
            stop_input()
            try:
                await websocket.send(_json(policy.public_error(error)))
            except Exception:
                LOGGER.debug("unable to send WebSocket error to disconnected peer", exc_info=True)
        finally:
            cancel_event.set()
            stop_input()
            if sender is not None and not sender.done():
                sender.cancel()
            if thread is not None and thread.is_alive():
                await asyncio.to_thread(thread.join, worker_shutdown_timeout)
            if recovery_session is not None:
                recovery_store.release(recovery_session, completed=session_completed)

    try:
        async with websocket_serve(
            handler,
            host,
            port,
            max_size=20 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ):
            await asyncio.Future()
    finally:
        backend_pool.close()
        recovery_store.close()
