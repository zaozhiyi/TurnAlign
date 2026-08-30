import asyncio
import json
import socket
import unittest
from array import array
from contextlib import suppress
from threading import Event
from unittest.mock import patch

from turnalign.models import Hypothesis
from turnalign.plugins import BackendCapabilities
from turnalign.policy import ServerPolicy
from turnalign.server import serve

try:
    from websockets.asyncio.client import connect
except ImportError:
    connect = None


class FakeServerBackend:
    name = "fake-server"
    capabilities = BackendCapabilities()

    def transcribe(self, chunks):
        items = list(chunks)
        if items:
            yield Hypothesis("local test", items[0].start, items[-1].start + items[-1].duration)

    def close(self):
        return None


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


class SlowServerBackend(FakeServerBackend):
    def __init__(self, release):
        self.release = release

    def transcribe(self, chunks):
        items = list(chunks)
        self.release.wait(timeout=2)
        if items:
            yield Hypothesis("slow test", items[0].start, items[-1].start + items[-1].duration)


@unittest.skipIf(connect is None, "websockets optional dependency is not installed")
class WebSocketTests(unittest.IsolatedAsyncioTestCase):
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
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({"type": "start", "backend": "fake"}))
                    response = json.loads(await websocket.recv())
                    self.assertEqual(response["type"], "error")
                    self.assertNotEqual(response.get("type"), "ready")
                    self.assertNotIn("/Users/private", response["message"])
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
                    await websocket.send(voice)
                    audio_ack = json.loads(await websocket.recv())
                    self.assertEqual(audio_ack["type"], "audio_ack")
                    self.assertEqual(audio_ack["acknowledged_sequence"], 1)

                await asyncio.sleep(0.05)
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "sample_rate": 8_000,
                        "resume_session_id": session_id,
                    }))
                    rejected = json.loads(await websocket.recv())
                    self.assertEqual(rejected["code"], "invalid_request")

                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.send(json.dumps({
                        "type": "start",
                        "backend": "fake",
                        "resume_session_id": session_id,
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


if __name__ == "__main__":
    unittest.main()
