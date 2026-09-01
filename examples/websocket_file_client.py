"""Stream a local PCM16 WAV file to a TurnAlign WebSocket server."""

import argparse
import asyncio
import json
import wave


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--backend", default="transformers-whisper")
    parser.add_argument("--model")
    parser.add_argument("--language")
    args = parser.parse_args()

    from websockets.asyncio.client import connect

    async with connect(args.uri) as socket:
        with wave.open(args.file, "rb") as source:
            if source.getsampwidth() != 2:
                raise ValueError("example client requires PCM16 WAV")
            await socket.send(json.dumps({
                "type": "start",
                "backend": args.backend,
                "model": args.model,
                "language": args.language,
                "sample_rate": source.getframerate(),
                "channels": source.getnchannels(),
            }))
            ready = json.loads(await socket.recv())
            public_ready = dict(ready)
            public_ready.pop("resume_token", None)
            print(json.dumps(public_ready, ensure_ascii=False))
            can_send = asyncio.Event()
            can_send.set()

            async def receive() -> None:
                async for message in socket:
                    print(message)
                    payload = json.loads(message)
                    if payload.get("type") == "flow_control":
                        if payload.get("action") == "pause":
                            can_send.clear()
                        elif payload.get("action") == "resume":
                            can_send.set()
                    if payload.get("kind") == "end":
                        return

            receiver = asyncio.create_task(receive())
            frames = max(1, source.getframerate() // 10)
            while data := source.readframes(frames):
                await can_send.wait()
                await socket.send(data)
                await asyncio.sleep(len(data) / (2 * source.getnchannels() * source.getframerate()))
            await socket.send(json.dumps({"type": "end"}))
            await receiver


if __name__ == "__main__":
    asyncio.run(main())
