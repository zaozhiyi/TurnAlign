import asyncio
import hashlib
import json
import os
import queue
import signal
import socket
import sys
import tempfile
import unittest
import wave
from array import array
from contextlib import suppress
from pathlib import Path
from threading import Event
from unittest.mock import patch

from turnalign.models import Hypothesis
from turnalign.plugins import BackendCapabilities
from turnalign.policy import ServerPolicy
from turnalign.recovery import RecoveryStore
from turnalign.server import (
    _control_message,
    _is_loopback_peer,
    _json,
    _OutputBackpressureError,
    _queue_output,
    _reject_unknown_fields,
    serve,
)
from turnalign.websocket_gate import (
    WebSocketSessionResult,
    _loaded_models,
    _probe_audio_material,
    run_websocket_gate,
)

try:
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidStatus
except ImportError:
    connect = None
    ConnectionClosed = InvalidStatus = Exception


class FakeServerBackend:
    name = "fake-server"
    capabilities = BackendCapabilities(hotwords=True, context_prompt=True)

    def transcribe(self, chunks):
        items = list(chunks)
        if items:
            yield Hypothesis("local test", items[0].start, items[-1].start + items[-1].duration)

    def close(self):
        return None


class MetricsAccessTests(unittest.TestCase):
    def test_metrics_peer_must_be_an_ip_loopback_address(self):
        self.assertTrue(_is_loopback_peer(("127.0.0.1", 1234)))
        self.assertTrue(_is_loopback_peer(("::1", 1234, 0, 0)))
        self.assertTrue(_is_loopback_peer(("::1%lo0", 1234, 0, 0)))
        self.assertFalse(_is_loopback_peer(("192.168.1.2", 1234)))
        self.assertFalse(_is_loopback_peer(("203.0.113.10", 1234)))
        self.assertFalse(_is_loopback_peer(("localhost", 1234)))
        self.assertFalse(_is_loopback_peer(None))

    def test_control_and_start_messages_reject_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            _reject_unknown_fields(
                {"type": "start", "unexpected": 1},
                allowed=frozenset({"type"}),
                label="start message",
            )

    def test_websocket_gate_accepts_only_bound_loaded_model_evidence(self):
        valid = _loaded_models([{
            "path": "/var/lib/turnalign/models/model.bin",
            "sha256": "a" * 64,
            "bytes": 12,
        }])
        self.assertEqual(len(valid), 1)
        with self.assertRaises(ValueError):
            _loaded_models([{
                "path": "/tmp/model.bin",
                "sha256": "a" * 64,
                "bytes": 12,
            }])

    def test_probe_audio_is_non_silent_and_length_bound(self):
        audio = _probe_audio_material(
            audio_seconds=0.1,
            sample_rate=16_000,
            channels=1,
            probe_audio_path=None,
        )
        self.assertEqual(len(audio.pcm), 1_600 * 2)
        self.assertGreater(sum(audio.pcm), 0)
        self.assertIsNone(audio.artifact_sha256)

    def test_probe_audio_binds_the_retained_wav_not_only_decoded_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(16_000)
                destination.writeframes(b"\x01\x00" * 1_600)
            artifact = path.read_bytes()
            audio = _probe_audio_material(
                audio_seconds=0.1,
                sample_rate=16_000,
                channels=1,
                probe_audio_path=path,
            )
            self.assertEqual(audio.artifact_sha256, hashlib.sha256(artifact).hexdigest())
            self.assertEqual(audio.artifact_bytes, len(artifact))
            self.assertNotEqual(audio.artifact_sha256, hashlib.sha256(audio.pcm).hexdigest())


class RecordingServerBackend(FakeServerBackend):
    def __init__(self):
        self.received = []

    def transcribe(self, chunks):
        items = list(chunks)
        self.received.append([
            (round(item.start, 6), round(item.duration, 6))
            for item in items
        ])
        if items:
            yield Hypothesis("local test", items[0].start, items[-1].start + items[-1].duration)


class HintAwareServerBackend(FakeServerBackend):
    session_hints = True

    def __init__(self):
        self.hints = None
        self.hint_history = []

    def set_hints(self, hints):
        self.hints = hints
        self.hint_history.append(hints.hotwords)


class ClosableComponent:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class SlowServerBackend(FakeServerBackend):
    def __init__(self, release):
        self.release = release

    def transcribe(self, chunks):
        items = list(chunks)
        self.release.wait(timeout=2)
        if items:
            yield Hypothesis("slow test", items[0].start, items[-1].start + items[-1].duration)


class CancellableServerBackend(FakeServerBackend):
    def __init__(self):
        self.started = Event()
        self.released = Event()
        self.cancelled = Event()

    def transcribe(self, chunks):
        items = list(chunks)
        self.started.set()
        self.released.wait(timeout=5)
        if items and not self.cancelled.is_set():
            yield Hypothesis(
                "late result",
                items[0].start,
                items[-1].start + items[-1].duration,
            )

    def cancel(self):
        self.cancelled.set()
        self.released.set()


class BlockingCancelServerBackend(CancellableServerBackend):
    def __init__(self):
        super().__init__()
        self.cancel_started = Event()
        self.cancel_release = Event()
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1
        self.cancel_started.set()
        self.cancel_release.wait(timeout=2)
        super().cancel()


class PermanentlyBlockingCancelServerBackend(CancellableServerBackend):
    def __init__(self):
        super().__init__()
        self.cancel_started = Event()
        self.cancel_release = Event()
        self.closed = Event()

    def transcribe(self, chunks):
        self.started.set()
        list(chunks)
        return ()

    def cancel(self):
        self.cancel_started.set()
        self.cancel_release.wait()
        super().cancel()

    def close(self):
        self.closed.set()


class UncancellableServerBackend(FakeServerBackend):
    def __init__(self):
        self.started = Event()
        self.released = Event()

    def transcribe(self, chunks):
        items = list(chunks)
        self.started.set()
        self.released.wait(timeout=5)
        if items:
            yield Hypothesis(
                "late result",
                items[0].start,
                items[-1].start + items[-1].duration,
            )


class TrackingServerBackend(FakeServerBackend):
    def __init__(self):
        self.closed = Event()

    def close(self):
        self.closed.set()


class PinnedTrackingServerBackend(TrackingServerBackend):
    model_revision = "a" * 40


class FailingCloseServerBackend(FakeServerBackend):
    def __init__(self):
        self.closed = Event()

    def close(self):
        self.closed.set()
        raise RuntimeError("backend close failed")


class OversizedEventBackend(FakeServerBackend):
    def transcribe(self, chunks):
        items = list(chunks)
        if items:
            yield Hypothesis(
                "private oversized output " + "x" * 2_000,
                items[0].start,
                items[-1].start + items[-1].duration,
            )


class PersistenceOrderSession:
    def __init__(self, backend):
        self.backend = backend

    def accept_audio(self, _chunk):
        self.backend.saw_persisted.append(self.backend.persisted.is_set())
        self.backend.consumed.set()
        return ()

    def finish(self):
        return ()

    def cancel(self):
        return None

    def close(self):
        return None


class PersistenceOrderBackend(FakeServerBackend):
    capabilities = BackendCapabilities(streaming=True, external_vad=False)

    def __init__(self, persisted, consumed):
        self.persisted = persisted
        self.consumed = consumed
        self.saw_persisted = []

    def start_session(self):
        return PersistenceOrderSession(self)


class OutputQueueTests(unittest.TestCase):
    def test_control_json_is_strict_and_server_json_is_standard(self):
        for message, detail in (
            ('{"type":"start","type":"cancel"}', "duplicate JSON key"),
            ('{"type":"start","value":NaN}', "non-standard JSON number"),
            ('{"type":"start","value":Infinity}', "non-standard JSON number"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, detail
            ):
                _control_message(message, label="test", max_bytes=1_024)
        with self.assertRaisesRegex(ValueError, "Out of range float"):
            _json({"value": float("nan")})

    def test_partial_is_dropped_immediately_when_output_is_full(self):
        outgoing = queue.Queue(maxsize=1)
        outgoing.put_nowait({"kind": "commit"})
        stats = {"dropped_partials": 0, "output_backpressure_timeouts": 0}
        queued = _queue_output(
            outgoing,
            {"kind": "partial"},
            Event(),
            stats,
            timeout=0.01,
        )
        self.assertFalse(queued)
        self.assertEqual(stats["dropped_partials"], 1)

    def test_durable_event_times_out_instead_of_blocking_forever(self):
        outgoing = queue.Queue(maxsize=1)
        outgoing.put_nowait({"kind": "commit"})
        stats = {"dropped_partials": 0, "output_backpressure_timeouts": 0}
        with self.assertRaisesRegex(_OutputBackpressureError, "remained full"):
            _queue_output(
                outgoing,
                {"kind": "end"},
                Event(),
                stats,
                timeout=0.01,
            )
        self.assertEqual(stats["output_backpressure_timeouts"], 1)


class ServerConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_transport_buffers_are_explicitly_bounded(self):
        captured = {}
        started = asyncio.Event()
        shutdown_event = asyncio.Event()

        class FakeWebSocketServer:
            def close(self, *, close_connections=True):
                return None

            async def wait_closed(self):
                return None

        async def fake_websocket_serve(_handler, _host, _port, **options):
            captured.update(options)
            started.set()
            return FakeWebSocketServer()

        with patch("websockets.asyncio.server.serve", new=fake_websocket_serve):
            server_task = asyncio.create_task(serve(shutdown_event=shutdown_event))
            await asyncio.wait_for(started.wait(), timeout=1)
            try:
                self.assertEqual(captured["max_size"], 16 * 1024 * 1024)
                self.assertGreaterEqual(
                    captured["max_size"],
                    96_000 * 8 * 2 * 10,
                )
                self.assertEqual(captured["max_queue"], 4)
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=1)

    async def test_rejects_invalid_lifecycle_limits_before_binding(self):
        for options, message in (
            ({"port": -1}, "port"),
            ({"port": True}, "port"),
            ({"internal_chunk_ms": 20.5}, "internal_chunk_ms"),
            ({"initialization_timeout": 0}, "initialization_timeout"),
            ({"initialization_timeout": True}, "initialization_timeout"),
            ({"finalization_timeout": 0}, "finalization_timeout"),
            ({"worker_shutdown_timeout": 0}, "worker_shutdown_timeout"),
            ({"output_backpressure_timeout": 0}, "output_backpressure_timeout"),
            ({"max_recovery_events": 0}, "max_recovery_events"),
            ({"max_recovery_events": 1.5}, "max_recovery_events"),
            ({"max_recovery_event_bytes": 0}, "max_recovery_event_bytes"),
            ({
                "max_recovery_event_bytes": 9,
                "max_recovery_event_bytes_per_session": 8,
            }, "cannot exceed"),
            ({"max_recovery_sessions": 0}, "max_recovery_sessions"),
            ({"max_recovery_audio_bytes": 0}, "max_recovery_audio_bytes"),
            ({"max_recovery_total_bytes": 0}, "max_recovery_total_bytes"),
            ({
                "max_recovery_audio_bytes": 9,
                "max_recovery_total_bytes": 8,
            }, "cannot exceed"),
            ({"max_concurrent_sessions": 0}, "max_concurrent_sessions"),
            ({"max_concurrent_sessions": True}, "max_concurrent_sessions"),
            ({"start_timeout": 0}, "start_timeout"),
            ({"client_idle_timeout": 0}, "client_idle_timeout"),
            ({"shutdown_grace_timeout": 0}, "shutdown_grace_timeout"),
            ({"recovery_ttl_seconds": 0}, "recovery_ttl_seconds"),
            ({"max_control_message_bytes": 0}, "max_control_message_bytes"),
            ({"max_control_message_bytes": True}, "max_control_message_bytes"),
            ({"max_control_message_bytes": 1024 * 1024 + 1}, "max_control_message_bytes"),
            ({"backend_replicas": 0}, "backend_replicas"),
            ({"backend_replicas": 1.5}, "backend_replicas"),
            ({"backend_replicas": 9}, "backend_replicas"),
            ({"allowed_origins": ()}, "allowed_origins"),
            ({"allowed_origins": "https://app.example"}, "allowed_origins"),
            ({"allowed_origins": (None, "")}, "allowed_origins"),
            ({"allowed_origins": (None, None)}, "allowed_origins"),
            ({
                "allowed_origins": (
                    None,
                    "https://app.example",
                    "https://app.example",
                )
            }, "allowed_origins"),
            ({"allowed_origins": (None, " https://app.example")}, "allowed_origins"),
            ({"allowed_origins": (None, "https://app.\texample")}, "allowed_origins"),
            ({"allowed_origins": (None, "ws://app.example")}, "exact"),
            ({"allowed_origins": (None, "https://user@app.example")}, "exact"),
            ({"allowed_origins": (None, "https://app.example/")}, "exact"),
            ({"allowed_origins": (None, "https://app.example?x=1")}, "exact"),
            ({"allowed_origins": (None, "https://app.example#x")}, "exact"),
            ({"allowed_origins": (None, "https://app.example:99999")}, "port"),
            ({"allowed_origins": (None, "https://app.example:")}, "exact"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                await serve(default_backend="fake", **options)
        for name in (
            "initialization_timeout",
            "finalization_timeout",
            "worker_shutdown_timeout",
            "output_backpressure_timeout",
            "start_timeout",
            "client_idle_timeout",
            "shutdown_grace_timeout",
            "recovery_ttl_seconds",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                await serve(default_backend="fake", **{name: float("nan")})

    async def test_rejects_inconsistent_defaults_before_preload_or_binding(self):
        policy = ServerPolicy(allowed_backends=frozenset({"approved"}))
        with patch("turnalign.server.create_asr") as create, self.assertRaisesRegex(
            ValueError, "default backend"
        ):
            await serve(
                default_backend="forbidden",
                policy=policy,
                preload=True,
            )
        create.assert_not_called()

        with self.assertRaisesRegex(TypeError, "preload"):
            await serve(default_backend="fake", preload=1)

    async def test_websocket_gate_rejects_credentials_in_uri(self):
        for uri, message in (
            ("wss://user:secret@example.test/ws", "auth-token-env"),
            ("wss://@example.test/ws", "auth-token-env"),
            ("wss://example.test/ws?token=secret", "auth-token-env"),
            ("wss://example.test/ws#secret", "auth-token-env"),
            ("wss://example.test:99999/ws", "invalid port"),
        ):
            with self.subTest(uri=uri), self.assertRaisesRegex(ValueError, message):
                await run_websocket_gate(uri)
        with self.assertRaisesRegex(ValueError, "at least two audio frames"):
            await run_websocket_gate(
                "ws://example.test/ws",
                audio_seconds=0.1,
                frame_ms=100,
                verify_recovery=True,
            )
        with self.assertRaisesRegex(ValueError, "recovery_resume_timeout"):
            await run_websocket_gate(
                "ws://example.test/ws",
                recovery_resume_timeout=float("nan"),
            )

    @unittest.skipIf(os.name == "nt", "POSIX SIGTERM is unavailable on Windows")
    async def test_standalone_server_exits_cleanly_on_sigterm(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        program = """
import asyncio
import sys
import turnalign.server as server

asyncio.run(server.serve(
    "127.0.0.1",
    int(sys.argv[1]),
    default_backend="fake",
))
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            program,
            str(port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            healthy = False
            for _ in range(100):
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                except OSError:
                    if process.returncode is not None:
                        break
                    await asyncio.sleep(0.01)
                    continue
                writer.write(
                    b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), timeout=1)
                writer.close()
                await writer.wait_closed()
                self.assertIn(b" 200 ", response.split(b"\r\n", 1)[0])
                healthy = True
                break
            if not healthy:
                stdout, stderr = await process.communicate()
                self.fail(
                    "server did not become healthy: "
                    + (stdout + stderr).decode(errors="replace")
                )
            process.send_signal(signal.SIGTERM)
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2)
            self.assertEqual(
                process.returncode,
                0,
                (stdout + stderr).decode(errors="replace"),
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


@unittest.skipIf(connect is None, "websockets optional dependency is not installed")
class WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_backend_event_is_redacted_and_not_sent(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()
        with patch(
            "turnalign.server.create_asr", return_value=OversizedEventBackend()
        ):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                max_recovery_event_bytes=256,
                max_recovery_event_bytes_per_session=1_024,
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(array("h", [1_500] * 1_600).tobytes())
                    self.assertEqual(json.loads(await websocket.recv())["type"], "audio_ack")
                    await websocket.send(json.dumps({"type": "end"}))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "session_error")
                    self.assertNotIn("oversized output", json.dumps(response))
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=2)

    async def test_backend_close_failure_does_not_block_terminal_event_or_shutdown(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()
        backend = FailingCloseServerBackend()
        with patch("turnalign.server.create_asr", return_value=backend), self.assertLogs(
            "turnalign.model_pool", level="WARNING"
        ):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "hotwords": ["private"],
                    }))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(json.dumps({"type": "end"}))
                    while True:
                        response = json.loads(await asyncio.wait_for(
                            websocket.recv(), timeout=1
                        ))
                        if response.get("kind") == "end":
                            break
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=2)
        self.assertTrue(backend.closed.is_set())

    async def test_oversized_utf8_start_is_rejected_before_model_creation(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()
        with patch("turnalign.server.create_asr") as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                max_control_message_bytes=80,
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "context": "汉" * 20,
                    }, ensure_ascii=False))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "invalid_request")
                create.assert_not_called()
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=2)

    async def test_health_readiness_origin_policy_and_graceful_shutdown(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()
        backend = TrackingServerBackend()
        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                allowed_origins=(None, "https://app.example"),
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                for path, expected in (
                    ("/healthz", {"status": "ok"}),
                    ("/readyz", {"status": "ok", "ready": True, "preloaded": False}),
                ):
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.write(
                        f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
                    )
                    await writer.drain()
                    response = await asyncio.wait_for(reader.read(), timeout=1)
                    writer.close()
                    await writer.wait_closed()
                    headers, body = response.split(b"\r\n\r\n", 1)
                    self.assertIn(b" 200 ", headers)
                    self.assertIn(b"content-type: application/json", headers.lower())
                    self.assertEqual(json.loads(body), expected)

                with self.assertRaises(InvalidStatus):
                    async with connect(
                        f"ws://127.0.0.1:{port}",
                        origin="https://untrusted.example",
                    ):
                        pass

                async with connect(
                    f"ws://127.0.0.1:{port}",
                    origin="https://app.example",
                ) as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    shutdown_event.set()
                    with self.assertRaises(ConnectionClosed):
                        await websocket.recv()

                await asyncio.wait_for(server_task, timeout=2)
                self.assertTrue(backend.closed.is_set())
            finally:
                shutdown_event.set()
                if not server_task.done():
                    await asyncio.wait_for(server_task, timeout=2)

    async def test_metrics_are_label_free_and_track_completed_sessions(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()

        async def scrape() -> tuple[bytes, str]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), timeout=1)
            writer.close()
            await writer.wait_closed()
            headers, body = response.split(b"\r\n\r\n", 1)
            return headers, body.decode("utf-8")

        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    audio = array("h", [1500] * 1600).tobytes()
                    await websocket.send(audio)
                    self.assertEqual(json.loads(await websocket.recv())["type"], "audio_ack")
                    await websocket.send(json.dumps({"type": "end"}))
                    while True:
                        event = json.loads(await asyncio.wait_for(websocket.recv(), 1))
                        if event.get("kind") == "end":
                            break

                for _ in range(20):
                    headers, body = await scrape()
                    if "turnalign_sessions_terminal_total 1" in body:
                        break
                    await asyncio.sleep(0.01)
                self.assertIn(b" 200 ", headers)
                self.assertIn(
                    b"content-type: text/plain; version=0.0.4; charset=utf-8",
                    headers.lower(),
                )
                self.assertIn("turnalign_active_sessions 0", body)
                self.assertIn("turnalign_connections_total 1", body)
                self.assertIn("turnalign_sessions_admitted_total 1", body)
                self.assertIn("turnalign_sessions_terminal_total 1", body)
                self.assertIn("turnalign_audio_frames_total 1", body)
                self.assertIn(f"turnalign_audio_bytes_total {len(audio)}", body)
                self.assertNotIn("{", body)
                self.assertNotIn("local test", body)
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=2)

    async def test_segmentation_controls_reject_non_finite_and_unsafe_values(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr") as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1", port, default_backend="fake"
            ))
            await asyncio.sleep(0.05)
            try:
                invalid = (
                    ("sample_rate", 16_000.5),
                    ("channels", True),
                    ("acknowledged_event_sequence", True),
                    ("vad_threshold", float("nan")),
                    ("silence_seconds", -1),
                    ("max_utterance_seconds", 301),
                    ("partial_seconds", 0),
                    ("partial_seconds", True),
                    ("context", []),
                )
                for key, value in invalid:
                    with self.subTest(key=key, value=value):
                        async with connect(f"ws://127.0.0.1:{port}") as websocket:
                            await websocket.send(json.dumps({
                                "type": "start",
                                "backend": "fake",
                                key: value,
                            }))
                            response = json.loads(await websocket.recv())
                            self.assertEqual(response["code"], "invalid_request")
                create.assert_not_called()
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_unsupported_hints_fail_before_model_creation(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr") as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="transformers-whisper",
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "transformers-whisper",
                        "hotword_boost": 2,
                    }))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "invalid_request")
                create.assert_not_called()
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_control_message_must_be_a_json_object(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve(
                "127.0.0.1", port, default_backend="fake"
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(json.dumps(["end"]))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "invalid_request")
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_short_frame_is_acknowledged_when_end_flushes_it(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve(
                "127.0.0.1", port, default_backend="fake"
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(array("h", [1500] * 320).tobytes())
                    buffered = json.loads(await websocket.recv())
                    self.assertEqual(buffered["type"], "audio_ack")
                    self.assertNotIn("acknowledged_sequence", buffered)
                    self.assertEqual(buffered["buffered_bytes"], 640)
                    await websocket.send(json.dumps({"type": "end"}))
                    flushed = json.loads(await websocket.recv())
                    self.assertEqual(flushed["acknowledged_sequence"], 0)
                    self.assertEqual(flushed["buffered_bytes"], 0)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_finalization_timeout_is_structured_and_cancels_backend(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = CancellableServerBackend()
        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                finalization_timeout=0.05,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(array("h", [1500] * 1600).tobytes())
                    self.assertEqual(json.loads(await websocket.recv())["type"], "audio_ack")
                    await websocket.send(json.dumps({"type": "end"}))
                    while True:
                        response = json.loads(await asyncio.wait_for(websocket.recv(), 1))
                        if response.get("type") == "error":
                            break
                    self.assertEqual(response["code"], "timeout")
                    self.assertTrue(backend.cancelled.is_set())
            finally:
                backend.released.set()
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_connection_capacity_is_bounded_before_thread_creation(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        server_task = asyncio.create_task(serve(
            "127.0.0.1",
            port,
            default_backend="fake",
            max_concurrent_sessions=1,
            start_timeout=2,
        ))
        await asyncio.sleep(0.05)
        try:
            async with (
                connect(f"ws://127.0.0.1:{port}"),
                connect(f"ws://127.0.0.1:{port}") as rejected,
            ):
                response = json.loads(await rejected.recv())
                self.assertEqual(response["code"], "server_busy")
        finally:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task

    async def test_start_and_client_idle_timeouts_are_structured(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                start_timeout=0.05,
                client_idle_timeout=0.05,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "timeout")
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "timeout")
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_websocket_gate_runs_concurrent_protocol_sessions(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = FakeServerBackend()
        backend.model_revision = "a" * 40
        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                default_model="model-a",
                default_device="cpu",
            ))
            await asyncio.sleep(0.05)
            try:
                report = await run_websocket_gate(
                    f"ws://127.0.0.1:{port}",
                    sessions=3,
                    audio_seconds=0.2,
                    frame_ms=20,
                    backend="fake",
                )
                self.assertTrue(report.passed)
                self.assertEqual(report.passed_sessions, 3)
                self.assertTrue(report.identity_consistent)
                self.assertEqual(report.backend, "fake")
                self.assertEqual(report.backend_implementation, "fake-server")
                self.assertEqual(report.model, "model-a")
                self.assertEqual(report.model_revision, "a" * 40)
                self.assertEqual(report.device, "cpu")
                self.assertTrue(all(
                    result.model_revision == "a" * 40
                    for result in report.results
                ))
                self.assertEqual(report.commits, 3)
                self.assertEqual(report.events, 6)
                self.assertGreaterEqual(
                    sum(result.audio_acks for result in report.results),
                    3,
                )
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_websocket_gate_rejects_inconsistent_deployment_identities(self):
        async def session_result(_uri, session, **_options):
            return WebSocketSessionResult(
                session=session,
                passed=True,
                backend="fake",
                backend_implementation="fake-server",
                model="model-a",
                model_revision=("a" if session == 1 else "b") * 40,
                device="cpu",
                ready_seconds=0.1,
                total_seconds=0.2,
            )

        with patch(
            "turnalign.websocket_gate._run_session",
            side_effect=session_result,
        ):
            report = await run_websocket_gate(
                "ws://example.test/ws",
                sessions=2,
            )
        self.assertFalse(report.passed)
        self.assertEqual(report.status, "failed")
        self.assertFalse(report.identity_consistent)
        self.assertIsNone(report.model_revision)

    async def test_websocket_gate_injects_disconnect_and_verifies_resume(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                report = await run_websocket_gate(
                    f"ws://127.0.0.1:{port}",
                    sessions=1,
                    audio_seconds=0.6,
                    frame_ms=100,
                    backend="fake",
                    verify_recovery=True,
                    recovery_resume_timeout=1,
                )
                self.assertTrue(report.passed, report.to_dict())
                self.assertTrue(report.recovery_probe_required)
                self.assertIsNotNone(report.recovery_probe)
                recovery = report.recovery_probe
                self.assertTrue(recovery.passed)
                self.assertEqual(
                    recovery.resumed_next_audio_sequence,
                    recovery.first_last_acknowledged_sequence + 1,
                )
                self.assertGreater(
                    recovery.final_acknowledged_sequence,
                    recovery.first_last_acknowledged_sequence,
                )
                self.assertEqual(recovery.final_buffered_bytes, 0)
                serialized_report = json.dumps(report.to_dict())
                self.assertNotIn("local test", serialized_report)
                self.assertNotIn("resume_token", serialized_report)
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=2)

    async def test_backend_replicas_enable_same_config_parallel_sessions(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch(
            "turnalign.server.create_asr",
            side_effect=lambda *_args, **_options: FakeServerBackend(),
        ) as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                backend_replicas=2,
            ))
            await asyncio.sleep(0.05)
            try:
                report = await run_websocket_gate(
                    f"ws://127.0.0.1:{port}",
                    sessions=2,
                    audio_seconds=0.2,
                    frame_ms=20,
                    realtime=True,
                    backend="fake",
                )
                self.assertTrue(report.passed)
                self.assertEqual(create.call_count, 2)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_preload_builds_all_replicas_before_accepting_sessions(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch(
            "turnalign.server.create_asr",
            side_effect=lambda *_args, **_options: FakeServerBackend(),
        ) as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                default_language="zh",
                backend_replicas=2,
                preload=True,
            ))
            await asyncio.sleep(0.05)
            try:
                self.assertEqual(create.call_count, 2)
                report = await run_websocket_gate(
                    f"ws://127.0.0.1:{port}",
                    sessions=2,
                    audio_seconds=0.1,
                    backend="fake",
                    language="zh",
                )
                self.assertTrue(report.passed)
                self.assertEqual(create.call_count, 2)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_preload_failure_prevents_server_start(self):
        with patch(
            "turnalign.server.create_asr",
            side_effect=RuntimeError("model unavailable"),
        ), self.assertRaisesRegex(RuntimeError, "model unavailable"):
            await serve(default_backend="fake", preload=True)

    async def test_preload_can_require_an_immutable_model_revision(self):
        mutable = TrackingServerBackend()
        with patch(
            "turnalign.server.create_asr",
            return_value=mutable,
        ), self.assertRaisesRegex(RuntimeError, "not pinned"):
            await serve(
                default_backend="fake",
                preload=True,
                require_immutable_revision=True,
            )
        self.assertTrue(mutable.closed.is_set())

        pinned = PinnedTrackingServerBackend()
        shutdown_event = asyncio.Event()
        shutdown_event.set()
        with patch("turnalign.server.create_asr", return_value=pinned):
            await serve(
                default_backend="fake",
                preload=True,
                require_immutable_revision=True,
                shutdown_event=shutdown_event,
            )
        self.assertTrue(pinned.closed.is_set())

    async def test_lazy_model_revision_failure_is_a_server_error_before_ready(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = TrackingServerBackend()
        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                require_immutable_revision=True,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                    }))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "session_error")
                    self.assertNotEqual(response.get("type"), "ready")
                await asyncio.to_thread(backend.closed.wait, 1)
                self.assertTrue(backend.closed.is_set())
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_websocket_gate_reports_threshold_failure_without_transcript(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                report = await run_websocket_gate(
                    f"ws://127.0.0.1:{port}",
                    sessions=1,
                    audio_seconds=0.1,
                    min_commits=2,
                    backend="fake",
                )
                self.assertFalse(report.passed)
                self.assertEqual(report.failed_sessions, 1)
                self.assertIsNotNone(report.results[0].ready_seconds)
                self.assertEqual(report.results[0].events, 2)
                self.assertEqual(report.results[0].commits, 1)
                self.assertNotIn("local test", json.dumps(report.to_dict()))
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_websocket_gate_rejects_malformed_audio_acknowledgement(self):
        from websockets.asyncio.server import serve as websocket_serve

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        async def malformed_server(websocket):
            await websocket.recv()
            await websocket.send(json.dumps({
                "type": "ready",
                "protocol_version": 1,
                "session_id": "malformed-session",
                "backend": "fake",
                "backend_implementation": "fake-server",
                "model_revision": None,
                "config": {},
            }))
            await websocket.recv()
            await websocket.send(json.dumps({
                "type": "audio_ack",
                "session_id": "malformed-session",
                "buffered_bytes": -1,
            }))
            with suppress(Exception):
                await websocket.wait_closed()

        async with websocket_serve(malformed_server, "127.0.0.1", port):
            report = await run_websocket_gate(
                f"ws://127.0.0.1:{port}",
                sessions=1,
                audio_seconds=0.1,
                frame_ms=20,
                timeout=2,
            )
        self.assertFalse(report.passed)
        self.assertIn("invalid buffered_bytes", report.results[0].failure)

    async def test_websocket_gate_rejects_ambiguous_or_binary_server_json(self):
        from websockets.asyncio.server import serve as websocket_serve

        for response, expected in (
            (
                (
                    '{"type":"ready","type":"error","protocol_version":1,'
                    '"session_id":"ambiguous"}'
                ),
                "duplicate JSON key",
            ),
            (
                (
                    '{"type":"ready","protocol_version":1,'
                    '"session_id":"nonstandard","value":NaN}'
                ),
                "non-standard JSON number",
            ),
            (
                (
                    b'{"type":"ready","protocol_version":1,'
                    b'"session_id":"binary"}'
                ),
                "unexpected binary frame",
            ),
        ):
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()

            async def malformed_server(websocket, payload=response):
                await websocket.recv()
                await websocket.send(payload)
                with suppress(Exception):
                    await websocket.wait_closed()

            async with websocket_serve(malformed_server, "127.0.0.1", port):
                report = await run_websocket_gate(
                    f"ws://127.0.0.1:{port}",
                    sessions=1,
                    audio_seconds=0.1,
                    frame_ms=20,
                    timeout=2,
                )
            with self.subTest(expected=expected):
                self.assertFalse(report.passed)
                self.assertIn(expected, report.results[0].failure)

    async def test_audio_is_persisted_before_inference_can_observe_it(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        persisted = Event()
        consumed = Event()
        backend = PersistenceOrderBackend(persisted, consumed)
        original_append = RecoveryStore.append_audio

        def delayed_append(store, session, chunk):
            # With the unsafe queue-first ordering, the worker consumes this
            # chunk while persistence is deliberately paused here.
            consumed.wait(timeout=0.2)
            sequence = original_append(store, session, chunk)
            persisted.set()
            return sequence

        with (
            patch("turnalign.server.create_asr", return_value=backend),
            patch.object(RecoveryStore, "append_audio", delayed_append),
        ):
            server_task = asyncio.create_task(serve(
                "127.0.0.1", port, default_backend="fake"
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(array("h", [1_000] * 1_600).tobytes())
                    self.assertEqual(json.loads(await websocket.recv())["type"], "audio_ack")
                    await websocket.send(json.dumps({"type": "end"}))
                    async for message in websocket:
                        if json.loads(message).get("kind") == "end":
                            break
                self.assertEqual(backend.saw_persisted, [True])
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_raw_pcm_session_returns_commit_and_end(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start", "backend": "fake",
                        "sample_rate": 16_000, "channels": 1,
                        "hotwords": ["PRIVATE_TERM"],
                    }))
                    ready = json.loads(await websocket.recv())
                    self.assertEqual(ready["type"], "ready")
                    self.assertTrue(ready["model_loaded"])
                    self.assertEqual(ready["protocol_version"], 1)
                    self.assertTrue(ready["session_id"])
                    self.assertEqual(ready["config"]["hints"]["hotword_count"], 1)
                    self.assertNotIn("PRIVATE_TERM", json.dumps(ready))
                    pcm = array("h", [1500] * 1600 + [0] * 12_800).tobytes()
                    await websocket.send(pcm)
                    await websocket.send(json.dumps({"type": "end"}))
                    received = []
                    async for message in websocket:
                        event = json.loads(message)
                        if event.get("kind"):
                            received.append(event)
                        if event.get("kind") == "end":
                            break
                    self.assertEqual([item["kind"] for item in received], ["commit", "end"])
                    self.assertEqual(received[0]["source_timestamp"], received[0]["end"])
                    self.assertEqual(received[-1]["metadata"]["websocket_frames"], 1)
                    self.assertEqual(received[-1]["metadata"]["internal_chunks"], 9)
                    self.assertGreater(received[-1]["metadata"]["websocket_bytes"], 0)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_ready_waits_until_backend_finishes_loading(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        loading = Event()
        release = Event()

        def create_backend(*_args, **_kwargs):
            loading.set()
            release.wait(timeout=2)
            return FakeServerBackend()

        with patch("turnalign.server.create_asr", side_effect=create_backend):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    await asyncio.to_thread(loading.wait, 1)
                    ready_task = asyncio.create_task(websocket.recv())
                    await asyncio.sleep(0.02)
                    self.assertFalse(ready_task.done())
                    release.set()
                    ready = json.loads(await ready_task)
                    self.assertEqual(ready["type"], "ready")
                    await websocket.send(json.dumps({"type": "end"}))
                    async for message in websocket:
                        if json.loads(message).get("kind") == "end":
                            break
            finally:
                release.set()
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_backend_load_failure_never_sends_ready_and_redacts_path(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch(
            "turnalign.server.create_asr",
            side_effect=RuntimeError("failed loading /Users/private/model.bin"),
        ):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                with self.assertLogs("turnalign.server", level="ERROR") as logs:
                    async with connect(f"ws://127.0.0.1:{port}") as websocket:
                        await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                        response = json.loads(await websocket.recv())
                        self.assertEqual(response["type"], "error")
                        self.assertNotEqual(response.get("type"), "ready")
                        self.assertNotIn("/Users/private", response["message"])
                self.assertIn("WebSocket session failed", "\n".join(logs.output))
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_partial_component_initialization_is_cleaned_up(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        aligner = ClosableComponent()
        policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            allowed_components=frozenset({"align", "diarize"}),
        )
        with (
            patch("turnalign.server.create_asr", return_value=FakeServerBackend()),
            patch(
                "turnalign.server.create_component",
                side_effect=[aligner, RuntimeError("component failed")],
            ),
        ):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                policy=policy,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "aligner": "align",
                        "diarizer": "diarize",
                    }))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "session_error")
                self.assertTrue(aligner.closed)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_default_policy_rejects_client_controlled_executable(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
        await asyncio.sleep(0.05)
        try:
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({
                    "type": "start",
                    "backend": "fake",
                    "executable": "/tmp/untrusted",
                }))
                response = json.loads(await websocket.recv())
                self.assertEqual(response["code"], "invalid_request")
                self.assertNotIn("/tmp/untrusted", response["message"])
        finally:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task

    async def test_sequential_sessions_reuse_one_loaded_backend(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = FakeServerBackend()
        with patch("turnalign.server.create_asr", return_value=backend) as create:
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                for _ in range(3):
                    async with connect(f"ws://127.0.0.1:{port}") as websocket:
                        await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                        self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                        await websocket.send(json.dumps({"type": "end"}))
                        async for message in websocket:
                            if json.loads(message).get("kind") == "end":
                                break
                self.assertEqual(create.call_count, 1)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_configured_auth_token_is_required_before_model_loading(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            auth_token="private-token",
        )
        with patch("turnalign.server.create_asr") as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                policy=policy,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "unauthorized")
                    self.assertNotIn("private-token", json.dumps(response))
                create.assert_not_called()
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_backpressure_emits_pause_then_resume(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        release = Event()
        ten_seconds = array("h", [1500] * 160_000).tobytes()
        with patch(
            "turnalign.server.create_asr",
            return_value=SlowServerBackend(release),
        ):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(ten_seconds)
                    await websocket.send(ten_seconds)
                    actions = []
                    while "resume" not in actions:
                        payload = json.loads(await asyncio.wait_for(websocket.recv(), 2))
                        if payload.get("type") == "flow_control":
                            actions.append(payload["action"])
                            if payload["action"] == "pause":
                                release.set()
                    self.assertEqual(actions[:2], ["pause", "resume"])
                    await websocket.send(json.dumps({"type": "cancel"}))
            finally:
                release.set()
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_cancel_releases_model_for_the_next_session(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with patch(
            "turnalign.server.create_asr",
            return_value=FakeServerBackend(),
        ) as create:
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(json.dumps({"type": "cancel"}))
                await asyncio.sleep(0.05)
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(json.dumps({"type": "end"}))
                    async for message in websocket:
                        if json.loads(message).get("kind") == "end":
                            break
                self.assertEqual(create.call_count, 1)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_cancel_interrupts_a_backend_with_a_cancel_hook(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = CancellableServerBackend()
        ten_seconds = array("h", [1500] * 160_000).tobytes()
        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                worker_shutdown_timeout=0.5,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(ten_seconds)
                    self.assertTrue(await asyncio.to_thread(backend.started.wait, 2))
                    await websocket.send(json.dumps({"type": "cancel"}))
                self.assertTrue(await asyncio.to_thread(backend.cancelled.wait, 1))
            finally:
                backend.released.set()
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_blocking_cancel_hook_does_not_freeze_server_or_reuse_backend(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        blocking = BlockingCancelServerBackend()
        replacement = FakeServerBackend()
        ten_seconds = array("h", [1500] * 160_000).tobytes()
        with patch(
            "turnalign.server.create_asr",
            side_effect=(blocking, replacement),
        ) as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                backend_replicas=2,
                max_concurrent_sessions=2,
                worker_shutdown_timeout=0.05,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as first:
                    await first.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await first.recv())["type"], "ready")
                    await first.send(ten_seconds)
                    self.assertTrue(await asyncio.to_thread(blocking.started.wait, 2))
                    await first.send(json.dumps({"type": "cancel"}))
                    self.assertTrue(
                        await asyncio.wait_for(
                            asyncio.to_thread(blocking.cancel_started.wait, 1),
                            timeout=0.5,
                        )
                    )

                    async with connect(f"ws://127.0.0.1:{port}") as second:
                        await second.send(json.dumps({
                            "type": "start",
                            "backend": "fake",
                        }))
                        ready = json.loads(
                            await asyncio.wait_for(second.recv(), timeout=0.5)
                        )
                        self.assertEqual(ready["type"], "ready")
                        await second.send(json.dumps({"type": "end"}))
                        async for message in second:
                            if json.loads(message).get("kind") == "end":
                                break
                self.assertEqual(create.call_count, 2)
                self.assertEqual(blocking.cancel_calls, 1)
            finally:
                blocking.cancel_release.set()
                blocking.released.set()
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_permanent_cancel_hook_is_detached_after_its_deadline(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        blocking = PermanentlyBlockingCancelServerBackend()
        replacement = FakeServerBackend()
        ten_seconds = array("h", [1500] * 160_000).tobytes()
        with patch(
            "turnalign.server.create_asr",
            side_effect=(blocking, replacement),
        ) as create:
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                backend_replicas=1,
                max_concurrent_sessions=1,
                worker_shutdown_timeout=0.2,
                backend_cancel_timeout=0.05,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as first:
                    await first.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await first.recv())["type"], "ready")
                    await first.send(ten_seconds)
                    self.assertTrue(await asyncio.to_thread(blocking.started.wait, 2))
                    await first.send(json.dumps({"type": "cancel"}))
                    self.assertTrue(
                        await asyncio.to_thread(blocking.cancel_started.wait, 1)
                    )
                await asyncio.sleep(0.15)

                async with connect(f"ws://127.0.0.1:{port}") as second:
                    await second.send(json.dumps({"type": "start", "backend": "fake"}))
                    ready = json.loads(
                        await asyncio.wait_for(second.recv(), timeout=0.5)
                    )
                    self.assertEqual(ready["type"], "ready")
                    await second.send(json.dumps({"type": "end"}))
                    async for message in second:
                        if json.loads(message).get("kind") == "end":
                            break
                self.assertEqual(create.call_count, 2)
                self.assertFalse(blocking.closed.is_set())
            finally:
                blocking.cancel_release.set()
                self.assertTrue(await asyncio.to_thread(blocking.closed.wait, 1))
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_uncancellable_backend_does_not_hold_connection_or_release_session_early(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = UncancellableServerBackend()
        ten_seconds = array("h", [1500] * 160_000).tobytes()
        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                worker_shutdown_timeout=0.05,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    ready = json.loads(await websocket.recv())
                    session_id = ready["session_id"]
                    resume_token = ready["resume_token"]
                    await websocket.send(ten_seconds)
                    self.assertTrue(await asyncio.to_thread(backend.started.wait, 2))
                    await websocket.send(json.dumps({"type": "cancel"}))
                    await asyncio.wait_for(websocket.wait_closed(), 1)
                self.assertFalse(backend.released.is_set())

                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "resume_session_id": session_id,
                        "resume_token": resume_token,
                    }))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["type"], "error")
                    self.assertEqual(response["code"], "session_conflict")
            finally:
                backend.released.set()
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_client_frame_sizes_produce_the_same_internal_chunks(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = RecordingServerBackend()
        pcm = array("h", [1500] * 1_600 + [0] * 12_800).tobytes()

        async def run_session(frames):
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                for frame in frames:
                    await websocket.send(frame)
                await websocket.send(json.dumps({"type": "end"}))
                async for message in websocket:
                    if json.loads(message).get("kind") == "end":
                        break

        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                await run_session([pcm])
                await run_session([pcm[index:index + 640] for index in range(0, len(pcm), 640)])
                self.assertEqual(backend.received[0], backend.received[1])
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_session_hints_reuse_model_and_are_cleared_between_leases(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = HintAwareServerBackend()
        captured_configs = []

        def create_backend(_name, config):
            captured_configs.append(config)
            return backend

        with (
            patch("turnalign.server.create_asr", side_effect=create_backend) as create,
            patch("turnalign.server.supports_session_hints", return_value=True),
        ):
            server_task = asyncio.create_task(serve(
                "127.0.0.1", port, default_backend="fake"
            ))
            await asyncio.sleep(0.05)
            try:
                for hotword in ("PRIVATE_ONE", "PRIVATE_TWO"):
                    async with connect(f"ws://127.0.0.1:{port}") as websocket:
                        await websocket.send(json.dumps({
                            "type": "start",
                            "backend": "fake",
                            "hotwords": [hotword],
                        }))
                        self.assertEqual(
                            json.loads(await websocket.recv())["type"], "ready"
                        )
                        await websocket.send(json.dumps({"type": "end"}))
                        async for message in websocket:
                            if json.loads(message).get("kind") == "end":
                                break
                self.assertEqual(create.call_count, 1)
                self.assertFalse(captured_configs[0].hints.active)
                self.assertIn(("PRIVATE_ONE",), backend.hint_history)
                self.assertIn(("PRIVATE_TWO",), backend.hint_history)
                self.assertEqual(backend.hint_history[-1], ())
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_operator_backend_paths_and_options_are_not_client_controlled(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        captured = []

        def create_backend(_name, config):
            captured.append(config)
            return FakeServerBackend()

        with patch("turnalign.server.create_asr", side_effect=create_backend):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                default_executable="/opt/trusted-cli",
                default_model_path="/models/trusted.bin",
                default_backend_options={"threads": 4},
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    self.assertEqual(json.loads(await websocket.recv())["type"], "ready")
                    await websocket.send(json.dumps({"type": "end"}))
                    async for message in websocket:
                        if json.loads(message).get("kind") == "end":
                            break
                self.assertEqual(captured[0].executable, "/opt/trusted-cli")
                self.assertEqual(captured[0].model_path, "/models/trusted.bin")
                self.assertEqual(captured[0].extra, {"threads": 4})
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_disconnected_session_resumes_from_disk_backed_audio(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = RecordingServerBackend()
        voice = array("h", [1500] * 3_200).tobytes()
        silence = array("h", [0] * 11_200).tobytes()

        with patch("turnalign.server.create_asr", return_value=backend):
            server_task = asyncio.create_task(serve("127.0.0.1", port, default_backend="fake"))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    ready = json.loads(await websocket.recv())
                    session_id = ready["session_id"]
                    resume_token = ready["resume_token"]
                    await websocket.send(voice)
                    audio_ack = json.loads(await websocket.recv())
                    self.assertEqual(audio_ack["type"], "audio_ack")
                    self.assertEqual(audio_ack["acknowledged_sequence"], 1)

                await asyncio.sleep(0.05)
                for invalid_token in ("wrong-token", "错误令牌🔒", "\ud800"):
                    async with connect(f"ws://127.0.0.1:{port}") as websocket:
                        await websocket.send(json.dumps({
                            "type": "start",
                            "backend": "fake",
                            "resume_session_id": session_id,
                            "resume_token": invalid_token,
                        }))
                        rejected = json.loads(await websocket.recv())
                        self.assertEqual(rejected["code"], "unauthorized")

                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "sample_rate": 8_000,
                        "resume_session_id": session_id,
                        "resume_token": resume_token,
                    }))
                    rejected = json.loads(await websocket.recv())
                    self.assertEqual(rejected["code"], "invalid_request")

                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "resume_session_id": session_id,
                        "resume_token": resume_token,
                        "acknowledged_event_sequence": -1,
                    }))
                    ready = json.loads(await websocket.recv())
                    self.assertTrue(ready["resumed"])
                    self.assertEqual(ready["next_audio_sequence"], 2)
                    await websocket.send(silence)
                    await websocket.send(json.dumps({"type": "end"}))
                    transcript = []
                    async for message in websocket:
                        payload = json.loads(message)
                        if payload.get("kind"):
                            transcript.append(payload)
                        if payload.get("kind") == "end":
                            break
                self.assertEqual([item["kind"] for item in transcript], ["commit", "end"])
                self.assertEqual(transcript[0]["segment_id"], "seg-000000")
                self.assertEqual(transcript[0]["start"], 0)
                self.assertAlmostEqual(transcript[0]["end"], 0.9)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task

    async def test_inactive_recovery_session_expires_after_ttl(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        shutdown_event = asyncio.Event()
        with patch("turnalign.server.create_asr", return_value=FakeServerBackend()):
            server_task = asyncio.create_task(serve(
                "127.0.0.1",
                port,
                default_backend="fake",
                recovery_ttl_seconds=0.05,
                shutdown_event=shutdown_event,
            ))
            await asyncio.sleep(0.05)
            try:
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    ready = json.loads(await websocket.recv())
                    session_id = ready["session_id"]
                    resume_token = ready["resume_token"]

                await asyncio.sleep(0.25)
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "resume_session_id": session_id,
                        "resume_token": resume_token,
                    }))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["code"], "unauthorized")
            finally:
                shutdown_event.set()
                await asyncio.wait_for(server_task, timeout=2)


if __name__ == "__main__":
    unittest.main()
