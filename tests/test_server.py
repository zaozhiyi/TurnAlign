import asyncio
import json
import socket
import unittest
from array import array
from contextlib import suppress
from unittest.mock import patch

from turnalign.models import Hypothesis
from turnalign.plugins import BackendCapabilities
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
                    }))
                    ready = json.loads(await websocket.recv())
                    self.assertEqual(ready["type"], "ready")
                    pcm = array("h", [1500] * 1600 + [0] * 12_800).tobytes()
                    await websocket.send(pcm)
                    await websocket.send(json.dumps({"type": "end"}))
                    received = []
                    async for message in websocket:
                        event = json.loads(message)
                        received.append(event)
                        if event.get("kind") == "end":
                            break
                    self.assertEqual([item["kind"] for item in received], ["commit", "end"])
                    self.assertEqual(received[-1]["metadata"]["websocket_frames"], 1)
                    self.assertGreater(received[-1]["metadata"]["websocket_bytes"], 0)
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task


if __name__ == "__main__":
    unittest.main()
