# Local validation report

Initial full-workload date: 2026-08-24
Windows low-load backend follow-up: 2026-08-27
Host: Windows, AMD RX 7650 GRE; core tests do not require a GPU.

## Automated checks

- 75 unit/integration tests pass, one optional test is skipped by environment,
  and 10 subtests cover rolling batch-model partials, a loopback WebSocket
  session, alignment/diarization replacement flow, private-hint redaction,
  execution profiles, and whisper.cpp Vulkan argument constraints.
- All source and test modules pass `compileall`.
- GLM-ASR full transcript: 259 commits plus one `end` event; validation passes.
- Whisper Medium GPU full transcript: 259 commits plus one `end` event; validation passes.
- FunASR/CAM++ full diarization: 2,777 non-empty commits plus one `end` event; validation passes.
- Each 259-segment JSONL replay takes about 120 ms on this host.
- A dependency-free `py3-none-any` wheel builds successfully.
- The rebuilt wheel installs into a clean target directory and its packaged
  `doctor --device cpu` command returns a valid runtime plan.

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
6. Event validation rejects stale revisions, unknown replacements, reused IDs,
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

## Known limitations

- Microphone capture requires a working PortAudio input device; automated tests cover PCM sources and endpointing without recording private audio.
- LocalAgreement currently compares exact characters; punctuation normalization
  and token-aware agreement still need evaluation on actual partial hypotheses.
- GLM/Whisper model adapters and optional FSMN-VAD, Paraformer alignment, and
  CAM++ diarization components are now in-process and lazily loaded. The full
  FunASR path remains offline post-processing rather than live diarization.
- Live/online diarization and WebSocket reconnection are not yet implemented.
- This recording has no human speaker-turn ground truth, so the run validates
  completeness and consistency but cannot report a defensible DER. Noise,
  overlapped speech and short interjections may still be misclustered.
- The GLM fusion keeps the better GLM text, but its speaker boundary is an
  approximate local alignment inside each 30-second GLM source window.
- Model weight licenses must be checked independently from the MIT core package.

## Required next validation milestone

Add a sliding-window speaker-change front end (1.5–2.0 second CAM++ windows with
overlap, median smoothing and hysteresis), then compare against a small manually
labelled subset. This is needed to measure DER and improve very short turns
without changing the ASR model.
