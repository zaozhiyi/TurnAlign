# WebSocket protocol v1

The server binds to `127.0.0.1` by default. One connection represents one audio
session. TurnAlign does not upload audio to a third-party service.

## Trust boundary

The default server policy accepts only the backend and model selected by the
server operator. Client-controlled executable paths, model paths, aligners and
diarizers are rejected unless the corresponding `turnalign serve --allow-*`
option is set. Binding to a non-loopback address requires `--allow-remote`.

For a non-local deployment, terminate TLS and enforce rate limits at a trusted
reverse proxy. TurnAlign can additionally require a start-message token loaded
from an environment variable:

```bash
export TURNALIGN_AUTH_TOKEN='replace-me'
turnalign serve --allow-remote --auth-token-env TURNALIGN_AUTH_TOKEN
```

Do not put a literal token in shell history.

## Start and ready

The first frame must be a JSON object:

```json
{
  "type": "start",
  "backend": "glm-asr",
  "model": "zai-org/GLM-ASR-Nano-2512",
  "auth": "required only when configured by the server",
  "hotwords": ["PHRASE_A", "PHRASE_B"],
  "context": "optional private topic context",
  "language": "zh",
  "sample_rate": 16000,
  "channels": 1,
  "silence_seconds": 0.7,
  "partial_seconds": 2.0,
  "max_utterance_seconds": 20
}
```

`ready` is sent only after the selected model and allowed components have loaded
successfully:

```json
{
  "type": "ready",
  "protocol_version": 1,
  "session_id": "uuid",
  "model_loaded": true,
  "backend": "glm-asr",
  "sample_rate": 16000,
  "channels": 1,
  "config": {},
  "capabilities": {}
}
```

If initialization fails, the server sends an error and never sends `ready`.
Private hotword/context values are never copied into ready, event or error
messages; only counts and usage flags are exposed.

## Audio, framing and flow control

Send signed 16-bit little-endian PCM as binary frames. A client frame must contain
complete samples and cannot exceed ten seconds. The server re-frames all accepted
input into fixed 20–100 ms internal chunks, so client frame size does not change
VAD boundaries. The final remainder on an explicit `end` may be shorter than
20 ms; an unacknowledged remainder is resent by the client after disconnect.

When the input queue is under pressure the server may send:

```json
{"type":"flow_control","action":"pause","queue_depth":96}
```

Clients must pause audio transmission until a matching `action:"resume"` message
arrives. If the queue remains full, the session fails with a structured error
instead of silently dropping PCM. Revisable `partial` events may be coalesced
under output pressure; `commit`, `replace` and `end` are retained.

Finish or cancel with:

```json
{"type":"end"}
{"type":"cancel"}
```

`ping` receives `pong`.

## Transcript events

The server returns `partial`, `commit`, `replace`, `speaker_merge`, and `end`
events. Every event includes `protocol_version`, `session_id`, an event
`sequence`, and the latest `acknowledged_sequence` for accepted internal audio.

The legal segment lifecycle is:

```text
new -> partial -> partial -> commit -> replace -> replace
new -> commit -> replace
```

`speaker_merge` contains distinct `metadata.from_speaker` and
`metadata.to_speaker`. Post-processing always reuses the first-pass
`segment_id`, increments `revision`, and does not append a duplicate segment.

The `end` metadata includes audio and processing time, real-time factor, source
frame/byte counts, normalized internal chunk count, queue peak, backpressure
count and dropped-partial count.

## Disconnect recovery

Every accepted internal audio chunk is appended to a disk-backed in-process
recovery timeline. After each binary client frame the server returns:

```json
{
  "type": "audio_ack",
  "session_id": "uuid",
  "acknowledged_sequence": 12,
  "buffered_bytes": 0
}
```

After an unexpected disconnect, reconnect with the same configuration:

```json
{
  "type": "start",
  "backend": "glm-asr",
  "resume_session_id": "uuid",
  "acknowledged_event_sequence": 8
}
```

The server replays stored events after sequence 8, reprocesses accepted audio
after the last committed boundary, continues audio/event/segment sequences, and
preserves the same session ID. An open partial is recovery-committed before new
segments begin, avoiding duplicate IDs or revision rollback.

Recovery is scoped to the lifetime of the server process and a bounded
128-session store. Clients must retain source audio for process crashes or
server replacement; durable cross-process recovery remains planned.

Errors use:

```json
{"type":"error","code":"invalid_request","message":"..."}
```

Internal paths and backend exception details are redacted by default.
