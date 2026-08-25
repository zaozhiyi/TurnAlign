# WebSocket protocol

The server is local-only by default (`127.0.0.1`). One connection represents one
audio session. Audio is never uploaded by the TurnAlign core.

## Start

The first frame must be JSON text:

```json
{
  "type": "start",
  "backend": "glm-asr",
  "model": "zai-org/GLM-ASR-Nano-2512",
  "hotwords": ["PHRASE_A", "PHRASE_B"],
  "context": "optional private topic context",
  "language": "zh",
  "device": "rocm:0",
  "sample_rate": 16000,
  "channels": 1,
  "aligner": "my-aligner-plugin",
  "diarizer": "my-diarizer-plugin",
  "silence_seconds": 0.7,
  "partial_seconds": 2.0,
  "max_utterance_seconds": 20
}
```

The server replies with `{"type":"ready", ...}` after the worker starts. Model
loading may continue until the first audio segment is decoded.

`hotwords` must be a JSON array of strings. `context` is optional and is rejected
when the selected backend does not advertise prompt support. The ready message
and transcript events report hint counts and application method only; they never
echo the private values.

## Audio and output

Send signed 16-bit little-endian PCM as binary frames. Each frame must contain
complete samples and may contain no more than ten seconds of audio. The server
returns TurnAlign JSON events (`partial`, `commit`, `replace`, `speaker_merge`,
and `end`) as text frames.

When alignment or diarization plugins are selected, low-latency commits are sent
first. After input finishes, enriched segments use `replace` with the same
`segment_id` and a higher revision. The terminal `end` event includes audio
duration, processing duration, real-time factor, and processing speed.

Finish the session with:

```json
{"type":"end"}
```

Errors are returned as `{"type":"error","message":"..."}`. Do not expose the
development server to an untrusted network: client-selected model identifiers
may trigger model downloads, and command backends can reference server-local
executables or model paths.

## Agent use

Codex, Claude Code and other coding agents can call the CLI directly or implement
this small protocol. JSONL stdout is stable enough for shell pipelines; WebSocket
is intended for long-running applications and remote audio producers.
