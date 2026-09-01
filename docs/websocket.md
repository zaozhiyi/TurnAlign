# WebSocket protocol v1

The server binds to `127.0.0.1` by default. One connection represents one audio
session. TurnAlign does not upload audio to a third-party service.

## Trust boundary

The default server policy accepts only the backend and model selected by the
server operator. Client-controlled executable paths, model paths, aligners and
diarizers are rejected unless the corresponding `turnalign serve --allow-*`
option is set. Binding to a non-loopback address requires `--allow-remote`.
Language and compute type are also fixed to the server defaults; additional
values require `--allow-language` or `--allow-compute-type`. This prevents
clients from bypassing preloaded replicas with unbounded configuration variants.

For a non-local deployment, terminate TLS and enforce rate limits at a trusted
reverse proxy. TurnAlign can additionally require a start-message token loaded
from an environment variable:

```bash
export TURNALIGN_AUTH_TOKEN='replace-me'
turnalign serve --allow-remote --auth-token-env TURNALIGN_AUTH_TOKEN \
  --language zh --preload
```

Do not put a literal token in shell history.

Connections without an `Origin` header are accepted by default. Browser
connections are rejected unless their exact Origin is listed with repeated
`--allow-origin` options. This is separate from token authentication; configure
both when exposing the service to a browser application. WebSocket compression
is disabled because PCM audio is effectively incompressible and decompression is
unnecessary work at the trust boundary.

One process accepts at most 32 active sessions by default. It waits at most ten
seconds for the first `start` message and 60 seconds between later client frames.
Operators can tune these bounded-resource controls with
`--max-concurrent-sessions`, `--start-timeout`, and `--client-idle-timeout`.
Model/component loading and cancelled-worker shutdown are separately bounded by
`--initialization-timeout` and `--worker-shutdown-timeout`. Final ASR and
post-processing after an explicit `end` are bounded independently by
`--finalization-timeout` (120 seconds by default). Capacity exhaustion returns
`server_busy`; client, initialization or finalization expiry returns `timeout`.
Recovery storage retains at most 32 sessions, 2,048 events per session, 512 MiB
of audio per session, and 2 GiB of audio per process by default. Tune these with
`--max-recovery-sessions`, `--max-recovery-events`,
`--max-recovery-audio-mib`, and `--max-recovery-total-mib`. A completed session
closes its temporary audio file immediately; an unfinished disconnected session
keeps bounded audio until it resumes or is evicted.
Each serialized recoverable event is additionally capped at 512 KiB, and one
session retains at most 8 MiB of event JSON. The oldest replay entries are
evicted until both count and byte windows fit. Configure these limits with
`--max-recovery-event-kib` and `--max-recovery-events-mib`. A backend result that
exceeds the single-event ceiling isn't truncated or exposed; the session fails
with a redacted `session_error`.
Inactive sessions expire after 300 seconds by default. A background sweeper
closes their audio files and removes replay state; configure this resume window
with `--recovery-ttl-seconds`. Active sessions are never TTL-evicted.

A backend configuration has one loaded model instance by default, so same-model
sessions queue while that instance is leased. `--backend-replicas N` permits up
to eight independent instances for true same-process parallel inference, at the
cost of approximately N times the model memory. Prefer multiple one-replica
processes when process isolation or accelerator scheduling is more important.
Use `--preload` to load every default-backend replica before the listening socket
opens. An optional `--warmup-file` then performs inference during that startup
phase. Weight download, compatibility, or memory failures consequently fail the
deployment instead of the first user request. Pre-download and checksum model
artifacts rather than relying on network access during startup. For backends
that expose a model repository revision, add
`--require-immutable-model-revision`; the server then rejects a mutable tag or
missing revision during preload (or first model creation when preload is off).
Backends that consume a separately checksummed local model file don't expose
repository revision metadata and must use that artifact-verification process
instead.

## Health and shutdown

The listener answers `GET /healthz` with a minimal liveness response and
`GET /readyz` with readiness and preload state. Responses are JSON and marked
`no-store`; they contain no model paths, credentials, transcript text, or backend
errors. Production readiness probes should use `/readyz`. With `--preload`, the
port doesn't open until every configured replica loads and optional warmup
inference succeeds.

On POSIX, `SIGTERM` initiates graceful shutdown. The listener stops accepting
connections, existing WebSockets receive close code 1012, and handlers may clean
up for 30 seconds by default. `--shutdown-grace-timeout` bounds this interval;
handlers still running afterwards are cancelled before model pools and recovery
audio are closed. Set the deployment platform's termination grace period longer
than this value. Windows service managers should use their normal process-control
integration; POSIX signal behavior isn't assumed there.

`turnalign serve` emits timestamped lifecycle logs to stderr at `INFO` by
default. `--log-level` accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, or
`CRITICAL`. Startup, readiness, completion, recovery expiry, and shutdown logs
include session identifiers and transport counters but never transcript text,
hotwords, context prompts, authentication tokens, or model paths.

Command backends can be configured entirely by the trusted operator with
`--executable`, `--model-path`, and repeated `--backend-option KEY=VALUE`
arguments. Clients cannot override paths unless `--allow-client-paths` is
explicitly enabled, and there is no client-controlled backend-options field.

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

Start and later control JSON are independently limited to 65,536 UTF-8 bytes by
default. This prevents large metadata, hint, or unknown-field payloads from
reaching JSON and model initialization while preserving the larger binary-frame
limit required by high-rate multichannel PCM. Configure the control limit with
`--max-control-message-bytes` (maximum 1 MiB).

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

Segmentation controls are finite and bounded: `vad_threshold` is 0–1,
`silence_seconds` is 0.1–10, `max_utterance_seconds` is 1–300, and
`partial_seconds` is 0.1–30 and cannot exceed the maximum utterance duration.
Invalid values are rejected before a model is leased.

## Audio, framing and flow control

Send signed 16-bit little-endian PCM as binary frames. A client frame must contain
complete samples and cannot exceed ten seconds. The server re-frames all accepted
input into fixed 20–100 ms internal chunks, so client frame size does not change
VAD boundaries. The final remainder on an explicit `end` may be shorter than
20 ms; the server accepts it as a final internal chunk and acknowledges it
before emitting terminal events.

When the input queue is under pressure the server may send:

```json
{"type":"flow_control","action":"pause","queue_depth":96}
```

Clients must pause audio transmission until a matching `action:"resume"` message
arrives. If the queue remains full, the session fails with a structured error
instead of silently dropping PCM. Revisable `partial` events may be coalesced
under output pressure. `commit`, `replace` and `end` are never silently dropped:
if a peer cannot drain them before the server timeout, the session is terminated
instead of blocking the inference worker indefinitely.

Finish or cancel with:

```json
{"type":"end"}
{"type":"cancel"}
```

Cancellation invokes an optional backend cancel hook. The bundled whisper.cpp
adapter terminates its active CLI process and force-kills it after a bounded
grace period if necessary. Backends without cooperative
cancellation cannot be force-stopped inside a Python thread; the server still
closes the transport after its shutdown timeout and keeps the recovery session
locked until the old worker actually exits, preventing concurrent resume writes.

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
frame/byte counts, normalized internal chunk count, queue peak, input
backpressure count, dropped-partial count and output-backpressure timeout count.

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

If a frame is smaller than one internal chunk and no earlier chunk has been
accepted, `acknowledged_sequence` is omitted and `buffered_bytes` reports the
unaccepted remainder. The acknowledgement after `end` includes the sequence of
the flushed final remainder. On resumed sessions, acknowledgements continue
from the previous accepted sequence.

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
32-session store. Each session retains the newest 2,048 events by default;
operators can change the window with `--max-recovery-events`. A resume request
whose acknowledgement predates the retained window is rejected explicitly, so
the client can restart from its retained source audio instead of receiving an
incomplete replay. Clients must retain source audio for process crashes, stale
replay windows or server replacement; durable cross-process recovery remains
planned.

Durable output events wait up to five seconds for a stalled client by default;
operators can tune this with `--output-backpressure-timeout`. Partial events may
be dropped under output pressure and the terminal metadata reports the count.

Errors use:

```json
{"type":"error","code":"invalid_request","message":"..."}
```

`unauthorized`, `invalid_request`, `server_busy`, `session_conflict`, `timeout`,
and `session_error` distinguish authentication, input, capacity, active-resume,
deadline and internal failures. Internal paths and backend exception details are
redacted by default; trusted server logs retain unexpected exception details.

## Deployment gate

`websocket-gate` opens concurrent sessions against an already running endpoint,
sends generated silence as PCM16, honors pause/resume flow control, validates
every event stream, and reports ready/total p95 latency, failures, audio
acknowledgements, backpressure, and dropped partials. It never retains transcript
text. Use `--realtime` for soak tests; omit it for burst tests. Authentication is
accepted only through an environment variable, and credentials or query strings
in the URI are rejected.

```bash
turnalign websocket-gate wss://asr.example/ws --sessions 8 \
  --audio-seconds 60 --realtime --max-ready-seconds 10 \
  --max-total-seconds 75 --min-audio-acks 600 \
  --max-dropped-partials 0 --verify-recovery \
  --auth-token-env TURNALIGN_AUTH_TOKEN
```

The gate requires at least one audio acknowledgement per session and permits no
dropped partials by default. Tighten `--min-audio-acks` to the expected frame or
internal-chunk count for a specific deployment. Burst tests may legitimately
exercise flow control; use `--max-backpressure-pauses` when pauses themselves
must be a release blocker. Totals and per-session values are retained in the
JSON report.

`--verify-recovery` adds a sequential fault-injection probe after the normal
load sessions. It sends silence until an accepted audio sequence is acknowledged
with an empty server frame buffer, deliberately closes the connection without
`end`, and retries resume for up to `--recovery-resume-timeout` seconds while the
old worker exits. The gate then requires the same session ID,
`resumed: true`, an exact `next_audio_sequence` continuation, advancing ACKs,
and a terminal `end` after the remaining audio. Recovery counters and failures
are stored under `recovery_probe`; transcript text is never retained.
Because v1 recovery is process-local, a multi-process or multi-pod deployment
must keep reconnects on the original instance for the recovery TTL. Run this
probe through the public load balancer: failure there exposes missing session
affinity or the need for a future durable shared recovery store.

Terminate TLS at a production reverse proxy or service mesh and pass WebSocket
upgrade traffic to the TurnAlign listener. The public gate should target the
external `wss://` URL so certificate, proxy timeout, authentication and upgrade
configuration are tested together.

Set the server's model replica/process count to the intended concurrency before
running this gate. Otherwise ready latency correctly includes time queued for a
single model instance.

Generated silence tests transport and lifecycle behavior, not recognition
quality. Run `release-gate` with representative speech and `quality-gate` with
human-labelled references as separate release requirements; use `evaluate` for
non-blocking analysis only.
