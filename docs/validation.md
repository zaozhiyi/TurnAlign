# Local validation report

Initial full-workload date: 2026-08-24
Windows low-load backend follow-up: 2026-08-27
Host: Windows, AMD RX 7650 GRE; core tests do not require a GPU.

Apple Silicon real-streaming follow-up: 2026-08-31, macOS 26.6.2 arm64,
FunASR 1.4.11 and PyTorch 2.13.0 on CPU.

Service-lifecycle hardening follow-up: 2026-09-01, macOS arm64.

## Automated checks

- The full unit/integration suite covers rolling batch partials, stateful
  incremental FunASR calls, two-pass segment replacement, disk-backed audio
  slicing, loopback WebSocket initialization/policy/model reuse,
  alignment/diarization replacement, private-hint redaction, secure-by-default
  whisper.cpp prompt handling, execution profiles, and whisper.cpp Vulkan
  argument constraints.
- Full-input FunASR components reject recordings beyond their configured
  materialization limit before constructing the float model input.
- All source and test modules pass `compileall`.
- Ruff is enforced in CI and passes over `src` and `tests`.
- The built wheel passes all 237 tests from site-packages on Python 3.10 with
  `websockets` 14.0 and on Python 3.12 with `websockets` 17.1. The Python 3.12
  run also passes under `python -O`, so production invariants do not depend on
  removable `assert` statements.
- Mypy checks the complete `src/turnalign` package, including untyped function
  bodies, with Python 3.10 as the minimum-language target. Adding this gate
  exposed and fixed a real offline-refinement fallback bug: an aligner that
  implements only `align()` (without the optional batch method) previously
  treated an already-extracted text string as a hypothesis object. A dedicated
  regression now exercises that plugin contract.
- Bandit reports no medium- or high-severity source findings. Its five remaining
  low-severity findings are the deliberate `subprocess` imports/calls used by
  the local audio and whisper.cpp adapters; calls use argument arrays with
  `shell=False`. Minimal server installations with both tested WebSocket
  versions report no known dependency vulnerabilities under `pip-audit`; CI
  repeats this audit weekly and includes the pinned build and publishing tools.
- `turnalign release-gate` provides an executable real-model gate for protocol
  validity, native streaming, initialization latency, first-partial latency,
  optional first-commit latency, commit count, sample duration, and RTF. It
  writes event JSONL evidence, includes every applied threshold and requirement
  in its report, and exits non-zero on threshold failure.
- `turnalign serve --require-immutable-model-revision` applies the same model
  supply-chain invariant to the live service. Preload fails before binding when
  a backend reports a mutable or missing revision; lazy creation rejects it
  before sending `ready`.
- `turnalign quality-gate` turns human-labelled reference events into an
  executable release decision. It validates both complete event streams,
  enforces configured CER/WER, speaker-error and revision-stability ceilings,
  checks minimum labelled segment/character/speech coverage, emits a JSON
  report, and exits non-zero on failure. At least one quality ceiling is
  required, so a sample-size check alone cannot be mistaken for model quality.
- `turnalign websocket-gate` provides a transcript-free deployment gate for
  concurrent sessions, protocol validity, flow control, acknowledgements,
  ready/total latency and dropped partials. Missing acknowledgements and any
  dropped partials fail by default; pause and latency limits are configurable.
  Generated silence intentionally
  keeps transport/load evidence separate from speech-quality evidence.
- All three executable gates can atomically persist their JSON verdict with
  `--report`. `turnalign production-gate` independently rechecks the critical
  thresholds, requires a public `wss://` concurrent recovery/soak result, and
  binds the source commit plus six required artifact classes and all reports by
  SHA-256 into one final pass/fail record.
- WebSocket processes now bound active sessions and time out clients that do
  not send the initial start message or become idle. Initialization, worker
  shutdown, output pressure and recovery-event retention are independently
  bounded and configurable.
- The same listener now serves minimal `/healthz` and `/readyz` probes, disables
  WebSocket compression, rejects browser origins unless explicitly allowed, and
  performs bounded graceful shutdown. An isolated subprocess test sends a real
  POSIX `SIGTERM` and requires a clean zero exit code; active-connection coverage
  also verifies model-pool closure.
- A label-free `/metrics` endpoint exposes bounded operational counters without
  transcript text, session/model labels or credentials. Regression coverage
  verifies session/audio accounting and the reference Nginx configuration
  explicitly blocks public metrics proxying.
- Disk-backed recovery now enforces independent per-session and process-wide
  audio byte ceilings, bounds retained session count, rolls reservations back on
  write failure, and closes completed-session audio immediately. Unit coverage
  verifies both capacity paths and reclaimed capacity.
- Inactive recovery sessions now have a configurable 300-second default TTL.
  The server sweeper removes expired replay state and closes its disk timeline;
  integration coverage disconnects a live client, waits past a short TTL, and
  verifies that resume is rejected while active sessions remain protected.
- Recovery requires a per-session high-entropy secret in addition to the logged
  session ID. Missing, incorrect, and expired credentials share the same public
  authorization failure, and deployment-gate reports retain neither the secret
  nor transcript text. Accepted audio reaches the recovery timeline before the
  inference thread, closing an event-acknowledgement race.
- Server lifecycle logging is enabled at configurable severity. Regression
  checks cover CLI configuration; emitted readiness/completion messages contain
  session and transport metadata without transcript or private-hint values.
- Text control frames have an independent UTF-8 byte ceiling, preventing the
  larger PCM frame allowance from becoming an oversized JSON/model-configuration
  path. Duplicate keys and non-standard numeric constants are rejected, and
  server output forbids non-finite JSON numbers. A multibyte Unicode regression
  proves size rejection before backend creation.
- Plugin cleanup isolates third-party `close()` failures across model-pool
  discard/eviction/shutdown, streaming sessions, post-processing components,
  two-pass refinement, and preload failures. Tests verify that all remaining
  resources are still closed and that a hinted WebSocket session still delivers
  its terminal event and releases cleanly when backend close raises.
- Recovery events are bounded by serialized UTF-8 bytes as well as count. Tests
  verify atomic rejection without sequence corruption, byte-window eviction,
  stale replay detection, and a redacted WebSocket `session_error` for an
  oversized backend result.
- The deployment gate can now inject an acknowledged mid-stream disconnect and
  require successful resume of the same session. Integration coverage verifies
  exact audio-sequence continuation, post-resume ACK advancement, zero final
  buffering, terminal completion, and transcript-free reporting.
- A clean real FunASR gate exposed that relying on `torchaudio` to install
  PyTorch was not portable. Both FunASR extras now declare `torch` explicitly;
  platform-specific accelerator builds may still require the PyTorch vendor
  index documented for that host.
- The real FunASR run also exposed two exact-boundary finalization bugs. The
  streaming adapter now always sends an explicit `is_final=True` flush, even
  when no PCM tail remains, and promotes the last partial to a commit if that
  flush returns no new text. Input decoding is preflighted before expensive
  model initialization, and protocol violations become structured gate
  failures instead of an unhandled traceback.
- GLM-ASR full transcript: 259 commits plus one `end` event; validation passes.
- Whisper Medium GPU full transcript: 259 commits plus one `end` event; validation passes.
- FunASR/CAM++ full diarization: 2,777 non-empty commits plus one `end` event; validation passes.
- Each 259-segment JSONL replay takes about 120 ms on this host.
- A dependency-free `py3-none-any` wheel and source distribution build
  successfully, and both pass `twine check`. The pinned Hatchling backend and a
  commit-derived `SOURCE_DATE_EPOCH` produce byte-identical wheel and sdist
  hashes across two independent builds. CI enforces that comparison, emits a
  SHA-256 manifest, and retains the checked artifacts for 14 days. Build tools
  are version-pinned, while first-party GitHub Actions are pinned to immutable
  commit hashes with checkout credential persistence disabled.
- The rebuilt wheel installs into a clean target directory and its packaged
  `doctor --device cpu` command returns a valid runtime plan.
- The source distribution includes a scoped Linux CPU systemd/Nginx deployment
  reference. Static regression checks require loopback-only application binding,
  start-message authentication, preload and warm-up, immutable model revisions,
  bounded recovery capacity, an unprivileged systemd security profile, TLS
  WebSocket upgrade headers, proxy rate/connection limits, disabled retry, and
  proxy timeouts longer than the application's idle timeout.
  Ubuntu CI also parses the real unit with `systemd-analyze verify` and the
  proxy configuration with `nginx -t` using an ephemeral self-signed test
  certificate.
- CodeQL runs the extended Python security query suite on pull requests and
  uploads trusted main/scheduled results to code scanning. Fork pull requests
  analyze without upload because their token is intentionally read-only. All
  workflow actions are pinned to immutable commits; Dependabot checks Python
  and GitHub Actions weekly, and `SECURITY.md` documents the current private
  reporting limitation and non-public fallback process.
- `turnalign evaluate reference.jsonl hypothesis.jsonl` reports CER, WER,
  permutation-invariant speaker error, segment counts, and revision updates per
  segment. The speaker metric is a single-active-speaker interval score, not a
  collar/overlap-aware pyannote DER. Synthetic coverage includes unequal speaker
  counts, missed speech, and false alarms during reference silence.
  `quality-gate` uses the same metrics but makes the result enforceable. For
  Mandarin without reference word segmentation, CER is the meaningful text
  metric; whitespace-based WER is intended for word-tokenized references.
  Text comparison is strict by default. Any Unicode normalization (`NFC` or
  `NFKC`), case folding, or punctuation removal must be enabled explicitly and
  is serialized in the report, preventing an undocumented preprocessing change
  from silently moving the release metric.
  Thresholds and minimum corpus size must be derived from the target scenarios
  and product tolerance rather than copied from the repository.

Example release invocation after setting project-owned acceptance values:

```bash
turnalign quality-gate reference.jsonl release-events.jsonl \
  --max-cer "$MAX_CER" \
  --unicode-normalization "$UNICODE_NORMALIZATION" \
  --max-diarization-error "$MAX_SPEAKER_ERROR" \
  --max-revision-updates-per-segment "$MAX_REVISIONS_PER_SEGMENT" \
  --min-reference-segments "$MIN_LABELLED_SEGMENTS" \
  --min-reference-characters "$MIN_LABELLED_CHARACTERS" \
  --min-reference-speech-seconds "$MIN_LABELLED_SPEECH_SECONDS"
```

Both inputs must be valid common-event JSONL streams ending in an `end` event.
Keep the JSON report and exact labelled-corpus revision with the release
evidence. Do not enable the speaker ceiling unless the reference includes
speaker-labelled intervals.

## Apple Silicon real-streaming gate

The CPU control run used the model repository's public 5.547-second Mandarin
sample with `paraformer-zh-streaming`. It passed the compatibility thresholds:
7 partials, 1 commit, first partial in 1.093 seconds, processing time 3.981
seconds, and RTF 0.7178. Cold initialization was 108.530 seconds; rebuilding
the model in the already-imported process took 2.622 seconds. Production must
therefore preload and keep workers warm rather than start one process per
request. The ModelScope `v2.0.4` tags were resolved during this review and the
built-in Paraformer aliases now use the immutable commits directly: streaming
`562b758fecc801f13079d846d06b0b024fd670c4` and batch
`71684869ca6d8bfa59057d8a367b3fb7345a0c02`. FunASR's package update check is
disabled. The built-in GLM-ASR and Transformers Whisper defaults are likewise
pinned to exact Hugging Face commits. Production release runs should enable
`--require-immutable-model-revision`; custom Hugging Face models pass
`revision=COMMIT_SHA`, while custom FunASR models pass
`model_revision=COMMIT_SHA` through `--backend-option`.

The pinned streaming commit was then loaded from a clean optional-dependency
environment and its public sample passed the strengthened release gate. The
final packaged-code rerun recorded the exact model revision, 7 partials, 1
commit, 7.476-second warm initialization, 0.992-second first partial,
3.656-second first commit and RTF 0.6611. All checkpoint keys matched. The same
real backend also passed service preload with immutable-revision enforcement
before socket binding. This verifies the immutable default, release report and
live-service startup contract together; it remains a single public execution
sample, not recognition-quality evidence.

A separate private 30-second sample produced one valid final commit at RTF
0.5304 but no partials, so it correctly failed the real-time-display gate. No
transcript text is retained in this report. This is evidence that one public
sample is insufficient: the release corpus must contain representative speech,
silence, accents, domain terms, and long uninterrupted utterances. Neither run
had human labels, so these figures prove execution and protocol behavior, not
CER/WER quality.

## Apple Silicon WebSocket deployment gate

The deployment gate was also run against the real `funasr-streaming` server on
the same CPU host. A clean environment first failed immediately because the
FunASR optional extra was absent, with no private details returned to the client.
After installing the extra, a first-boot attempt downloaded the 881 MB pinned
weight and exceeded the 180-second client gate. This is expected negative
evidence: production artifacts must prefetch and checksum weights rather than
download on a user request.

With two cached model replicas and `--preload`, both models loaded before the
socket began listening. Two simultaneous five-second, real-time-paced sessions
then passed: ready latency was 0.0025/0.0053 seconds and total latency was
5.799/5.802 seconds. The server returned 100 audio acknowledgements, two valid
terminal events, no dropped partials and no backpressure pauses. Silence
intentionally produced no transcript commits. Stopping the service returned
exit code 130 without a traceback. This proves local two-replica transport and
lifecycle behavior; target-machine memory, concurrency, proxy/TLS, long soak and
fault injection remain deployment-specific gates.

The official 5.547-second Mandarin sample was then streamed through the same
preloaded WebSocket path. It produced seven partial revisions, one commit and one
end event with 56 acknowledged internal chunks, no dropped partials and no
backpressure pause. End-to-end processing was 6.200 seconds (RTF 1.1178) on this
CPU run. This test exposed a finalization gap specific to a short final PCM
remainder: when FunASR returned no new text for that remainder, the last partial
was not promoted. The adapter now synthesizes the final commit for both empty
exact-boundary flushes and empty short-tail flushes, with regression coverage.

### Synthetic transport stress follow-up

After strengthening the deployment gate, a dependency-free backend was used to
exercise transport independently of model speed. A 32-session burst with eight
backend replicas sent 3,200 client frames and received 3,200 valid audio
acknowledgements, 32 terminal events, no flow-control pauses and no dropped
partials. Ready p95 was 0.268 seconds and total p95 was 0.312 seconds. A separate
eight-session, ten-second real-time-paced run received 800 valid acknowledgements
and eight terminal events with no pauses or drops; total p95 was 10.151 seconds.
Every acknowledgement was checked for session identity, non-negative buffer
depth and monotonic accepted-audio sequence, and each session ended with a zero
buffer depth. These runs validate local protocol concurrency and the gate itself,
not model throughput, TLS/proxy behavior or recognition quality.

## Cross-platform device matrix

- Real AMD probe: RX 7650 GRE is selected as `rocm`, with PyTorch device
  `cuda:0`, FP16, ROCm 7.2 and 7.98 GiB detected VRAM.
- Simulated NVIDIA probe: CUDA, device enumeration, VRAM and CUDA ONNX Provider
  mapping pass.
- Simulated Apple probe: MPS, FP16, CoreML ONNX Provider and CPU-operator
  fallback plan pass.
- CPU is always available; explicit selection, multi-GPU index selection,
  environment override, backend capability filtering and failure on unavailable
  accelerators all pass.

## Windows low-load backend follow-up

The follow-up host also has an AMD Ryzen 5 5600GT integrated Radeon. It reports
about 4,079 MiB of dedicated GPU memory and 14,282 MiB of shared system memory.
The shared amount is addressable system RAM, not equivalent-bandwidth VRAM.

### whisper.cpp Vulkan

The unsigned `lemonade-sdk/whisper.cpp-rocm` v1.8.4 Windows Vulkan asset was
pinned to SHA-256
`e0d20a0f92e31b98adc0faf71172efc810b701e6391a9d858ca045bff26f77cd`.
The `ggml-small-q5_1.bin` model was pinned to revision
`5359861c739e955e79d9a303bcbc70fb988958b1` and SHA-256
`ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb`.

- Direct `whisper-cli -dev 1` on the integrated GPU completed in 4.301 seconds.
- TurnAlign `--device vulkan:1` completed end to end at RTF 0.9579. Whole-system
  CPU averaged 12.63%, peaked at 18.83%, and event validation passed.
- The short `small-q5_1` transcript was poor and there is no human reference,
  so this is execution-path evidence rather than a WER/CER or quality result.
- RX 7650 GRE `-dev 0` failed twice with `0xC0000409`, with Flash Attention both
  enabled and disabled. This failure is scoped to the pinned downstream build
  and current driver rather than generalized to every Vulkan build.

### DirectML experiment outside TurnAlign

TurnAlign does not ship a DirectML adapter. A separate A/B used PyTorch
2.4.1+cpu, torch-directml 0.2.5.dev240914, and Transformers 4.57.6. On this
stack, accepting the raw `model.generate()` Tensor materialized `[0, 0]` and
decoded as punctuation. Setting `return_dict_in_generate=True` and reading
`.sequences` restored a structurally valid sequence.

- Integrated Radeon FP16: 3.916 seconds, RTF 0.9138.
- RX 7650 GRE FP16: 1.450 seconds, RTF 0.3383.
- Integrated Radeon FP32: 5.090 seconds, RTF 1.1876, but semantically unreliable
  text and therefore not recommended.

The FP16 runs prove that the integrated GPU is not limited to its roughly 4 GiB
dedicated allocation, but they are short-sample, version-specific runtime
evidence without an accuracy reference. They are not a TurnAlign supported
backend claim.

### CPU fallback

The existing 120-second faster-whisper Medium INT8 run used four threads, VAD,
and Windows `BelowNormal`. It reached about 4.27x real time with 42.03% average
whole-system CPU. A 12-thread configuration had saturated the CPU, so the
follow-up did not rerun that high-load setting.

## Full diarization run

Input: 7,764.551 seconds (129.4 minutes), 16 kHz mono WAV.  Pipeline:
FSMN-VAD -> Paraformer timestamps/text -> CAM++ embeddings -> global
UMAP/HDBSCAN clustering -> GLM-ASR text alignment.

- AMD RX 7650 GRE through PyTorch 2.9.1 + ROCm 7.2.1.
- Model load: 6.1 seconds; full inference: 369.2 seconds (RTF 0.048,
  about 21x real time); peak allocated VRAM: 1,605.7 MB.
- The model returned 2,781 raw VAD/ASR turns; four empty-ASR turns are dropped,
  leaving 2,777 exported turns, 3 speaker clusters and 7,159.5 seconds of speech
  (92.2% input coverage).
- Raw turns: no invalid/out-of-order timestamps and no adjacent overlaps.
- Readable GLM fusion: 466 turns, no empty text, invalid timestamps or
  overlaps. Median local GLM/Paraformer character alignment is 88.6%; P10 is
  75.7%.
- Speaker 0 dominates the pre-discussion portion. Speakers 1 and 2 dominate
  the main discussion after approximately 25 minutes. Rare later speaker-0
  assignments are retained as possible background/third-person speech rather
  than silently relabelled.

## Bugs fixed during validation

1. LocalAgreement with three confirmations could starve forever when every new
   hypothesis extended the previous one. It now computes the common prefix over
   a fixed history window.
2. A final hypothesis that changes committed text now requests `replace` rather
   than appending duplicate/conflicting text.
3. Replay now creates missing output directories and emits a terminal `end` event.
4. JSONL input is streamed line by line instead of loaded fully into memory.
5. Invalid `NaN`/infinite timestamps, incomplete PCM frames and reversed ranges
   are rejected.
6. Event validation accepts the legal `partial -> commit -> replace` lifecycle
   and rejects stale revisions, unknown replacements, illegal transitions,
   out-of-order commits, early end events and events after session end.
7. Equal speaker-overlap ties now use diarization confidence deterministically.
8. JSONL parse failures include source path and line number.
9. Long-audio CAM++ clustering activates optional UMAP/HDBSCAN code that short
   smoke tests do not reach. The isolated runtime now installs `umap-learn`,
   `hdbscan`, and `pynndescent` explicitly.
10. The diarization event exporter originally emitted revision zero and an
    unsupported `is_final` field. Running the actual CLI validator exposed the
    mismatch; it now emits protocol-compatible revision-one events.
11. GLM/Paraformer character alignment could create backward/overlapping
    micro-turns around English tokens. Local timestamp clamping and sub-300 ms
    deglitch merging now guarantee a chronological, non-overlapping export.
12. A native FunASR stream whose last WebSocket frame was shorter than the model
    chunk could end after a partial when the final model call returned empty.
    The last partial is now promoted to a commit before the terminal event.

## Known limitations

- Microphone capture requires a working PortAudio input device; automated tests cover PCM sources and endpointing without recording private audio.
- LocalAgreement is connected to the rolling batch live path. It ignores
  punctuation, spacing, and case while locating the stable prefix; token-aware
  agreement still needs evaluation on actual partial hypotheses.
- GLM/Whisper model adapters and optional FSMN-VAD, Paraformer alignment, and
  CAM++ diarization components are now in-process and lazily loaded. The full
  FunASR path remains offline post-processing rather than live diarization.
- An online diarization session contract and provisional event labelling are
  implemented, but no built-in online speaker model has a defensible DER yet.
- The common timeline and alignment slices are disk-backed and bounded in
  batches. Current FSMN-VAD and CAM++ runtimes still require a full float array
  because their upstream offline API is not incremental.
- WebSocket v1 supports in-process resume, event replay, and disk-backed audio
  continuation. Event replay uses a bounded window and rejects stale
  acknowledgements; durable recovery after server process loss is not yet
  implemented.
- This recording has no human speaker-turn ground truth, so the run validates
  completeness and consistency but cannot report a defensible DER. Noise,
  overlapped speech and short interjections may still be misclustered.
- The GLM fusion keeps the better GLM text, but its speaker boundary is an
  approximate local alignment inside each 30-second GLM source window.
- Model weight licenses must be checked independently from the MIT core package.

## Required next validation milestone

Run `websocket-gate` on the actual deployment topology at the intended session
count in both burst and real-time soak modes, including proxy/TLS/auth and fault
injection. Build a small, licensed and manually labelled release corpus, then
enforce per-scenario CER/WER, first-partial latency, commit latency and RTF
thresholds on the actual deployment hardware. For speaker quality, add a sliding-window
speaker-change front end (1.5–2.0 second CAM++ windows with overlap, median
smoothing and hysteresis), then measure DER on labelled speaker turns.
