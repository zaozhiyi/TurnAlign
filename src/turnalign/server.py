from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path

from .audio import file_chunks
from .models import AudioChunk
from .plugins import AsrConfig
from .registry import create_asr, create_component
from .session import transcribe_events


def _json(item: object) -> str:
    return json.dumps(item, ensure_ascii=False)


async def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    default_backend: str = "transformers-whisper",
    default_model: str | None = None,
    default_device: str = "auto",
    warmup_file: Path | None = None,
    ffmpeg: str = "ffmpeg",
) -> None:
    try:
        from websockets.asyncio.server import serve as websocket_serve
    except ImportError as error:
        raise RuntimeError("WebSocket support requires: pip install 'turnalign[server]'") from error

    async def handler(websocket) -> None:
        try:
            first = await websocket.recv()
            if not isinstance(first, str):
                await websocket.send(_json({"type": "error", "message": "first frame must be start JSON"}))
                return
            request = json.loads(first)
            if request.get("type") != "start":
                await websocket.send(_json({"type": "error", "message": "first message type must be start"}))
                return
            backend_name = str(request.get("backend") or default_backend)
            sample_rate = int(request.get("sample_rate", 16_000))
            channels = int(request.get("channels", 1))
            if sample_rate <= 0 or channels <= 0:
                raise ValueError("sample_rate and channels must be positive")
            config = AsrConfig(
                model=request.get("model") or default_model,
                device=str(request.get("device") or default_device),
                language=request.get("language"),
                compute_type=request.get("compute_type"),
                executable=request.get("executable"),
                model_path=request.get("model_path"),
            )
            incoming: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=128)
            outgoing: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=128)
            transport_stats = {"frames": 0, "bytes": 0, "queue_peak": 0}
            loop = asyncio.get_running_loop()

            def chunks():
                while True:
                    item = incoming.get()
                    if item is None:
                        return
                    yield item

            def stop_input() -> None:
                try:
                    incoming.put_nowait(None)
                except queue.Full:
                    incoming.get_nowait()
                    incoming.put_nowait(None)

            def emit(item: dict[str, object] | None) -> None:
                asyncio.run_coroutine_threadsafe(outgoing.put(item), loop).result()

            def worker() -> None:
                try:
                    backend = create_asr(backend_name, config)
                    if warmup_file is not None:
                        try:
                            list(backend.transcribe(file_chunks(warmup_file, ffmpeg=ffmpeg)))
                        except Exception:
                            backend.close()
                            raise
                    aligner_name = request.get("aligner")
                    diarizer_name = request.get("diarizer")
                    aligner = create_component(
                        "alignment", str(aligner_name), request.get("aligner_options") or {}
                    ) if aligner_name else None
                    diarizer = create_component(
                        "diarization", str(diarizer_name), request.get("diarizer_options") or {}
                    ) if diarizer_name else None
                    for event in transcribe_events(
                        chunks(), backend, live=True,
                        vad_threshold=float(request.get("vad_threshold", 0.012)),
                        silence_seconds=float(request.get("silence_seconds", 0.7)),
                        max_utterance_seconds=float(request.get("max_utterance_seconds", 20.0)),
                        partial_seconds=float(request.get("partial_seconds", 2.0)),
                        aligner=aligner,
                        diarizer=diarizer,
                    ):
                        payload = event.to_dict()
                        if event.kind == "end":
                            payload["metadata"].update({
                                "websocket_frames": transport_stats["frames"],
                                "websocket_bytes": transport_stats["bytes"],
                                "input_queue_peak": transport_stats["queue_peak"],
                            })
                        emit(payload)
                except Exception as error:
                    emit({"type": "error", "message": str(error)})
                finally:
                    emit(None)

            thread = threading.Thread(target=worker, name="turnalign-session", daemon=True)
            thread.start()
            await websocket.send(_json({
                "type": "ready",
                "backend": backend_name,
                "sample_rate": sample_rate,
                "channels": channels,
                "config": {key: value for key, value in asdict(config).items() if value is not None},
            }))

            async def send_events() -> None:
                while True:
                    item = await outgoing.get()
                    if item is None:
                        return
                    await websocket.send(_json(item))

            sender = asyncio.create_task(send_events())
            position = 0.0
            try:
                async for message in websocket:
                    if isinstance(message, bytes):
                        if len(message) % (2 * channels):
                            raise ValueError("binary frame must contain complete signed 16-bit PCM frames")
                        if len(message) > sample_rate * channels * 2 * 10:
                            raise ValueError("binary frame exceeds 10 seconds")
                        chunk = AudioChunk(message, position, sample_rate, channels)
                        try:
                            incoming.put_nowait(chunk)
                        except queue.Full as error:
                            raise RuntimeError("audio input queue is full; client is sending faster than inference") from error
                        transport_stats["frames"] += 1
                        transport_stats["bytes"] += len(message)
                        transport_stats["queue_peak"] = max(
                            transport_stats["queue_peak"], incoming.qsize()
                        )
                        position += chunk.duration
                        continue
                    control = json.loads(message)
                    if control.get("type") == "end":
                        break
                    if control.get("type") == "ping":
                        await websocket.send(_json({"type": "pong"}))
                        continue
                    raise ValueError(f"unsupported control message: {control.get('type')}")
            finally:
                stop_input()
            await sender
        except Exception as error:
            try:
                await websocket.send(_json({"type": "error", "message": str(error)}))
            except Exception:
                pass

    async with websocket_serve(handler, host, port, max_size=20 * 1024 * 1024):
        await asyncio.Future()
