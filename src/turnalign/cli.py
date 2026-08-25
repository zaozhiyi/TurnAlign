from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .audio import file_chunks, input_devices, microphone_chunks, write_wave
from .backends.jsonl import JsonlBackend
from .devices import runtime_report
from .hints import AsrHints
from .models import TranscriptEvent
from .plugins import AsrConfig
from .profiles import PROFILE_NAMES, profile_catalog, select_execution_profile
from .registry import available, create_asr, create_component
from .server import serve
from .session import transcribe_events
from .validation import EventStreamValidator


def replay(source: Path, output: Path | None) -> int:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    destination = output.open("w", encoding="utf-8") if output else None
    last_end = 0.0
    count = 0
    try:
        for index, hypothesis in enumerate(JsonlBackend(source).hypotheses()):
            count = index + 1
            last_end = max(last_end, hypothesis.end)
            event = TranscriptEvent(
                kind="commit",
                segment_id=f"seg-{index:06d}",
                revision=1,
                start=hypothesis.start,
                end=hypothesis.end,
                text=hypothesis.text,
                metadata=hypothesis.metadata,
            )
            line = json.dumps(event.to_dict(), ensure_ascii=False)
            if destination:
                destination.write(line + "\n")
            else:
                print(line)
        end_event = TranscriptEvent(
            kind="end",
            segment_id="session",
            revision=1,
            start=last_end,
            end=last_end,
            metadata={"segments": count},
        )
        line = json.dumps(end_event.to_dict(), ensure_ascii=False)
        if destination:
            destination.write(line + "\n")
        else:
            print(line)
        return 0
    finally:
        if destination:
            destination.close()


def validate_events(source: Path) -> int:
    validator = EventStreamValidator()
    count = 0
    commits = 0
    with source.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                event = TranscriptEvent.from_dict(json.loads(line))
                validator.accept(event)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            count += 1
            commits += event.kind == "commit"
    if not validator.ended:
        raise ValueError(f"{source}: missing end event")
    print(json.dumps({"status": "ok", "events": count, "commits": commits}, ensure_ascii=False))
    return 0


def _resolved_device(requested: str) -> str:
    if requested != "auto":
        return requested
    selected = runtime_report("auto")["selected"]
    accelerator = selected["accelerator"]
    if accelerator in {"cuda", "rocm"}:
        index = str(selected["device"]).split(":")[-1]
        return f"{accelerator}:{index}"
    return accelerator


def _extra_options(items: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"backend option must be key=value: {item}")
        key, raw = item.split("=", 1)
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            result[key] = raw
    return result


def _config(args, *, device: str | None = None) -> AsrConfig:
    return AsrConfig(
        model=args.model,
        device=device or _resolved_device(args.device),
        language=args.language,
        compute_type=args.compute_type,
        executable=getattr(args, "executable", None),
        model_path=getattr(args, "model_path", None),
        extra=_extra_options(args.backend_option),
        hints=_hints(args),
    )


def _hints(args) -> AsrHints:
    hotwords = list(getattr(args, "hotword", []) or [])
    for path in getattr(args, "hotwords_file", []) or []:
        with path.open("r", encoding="utf-8-sig") as source:
            hotwords.extend(line.strip() for line in source)
    context = getattr(args, "context", None)
    context_file = getattr(args, "context_file", None)
    if context_file is not None:
        context = context_file.read_text(encoding="utf-8-sig")
    return AsrHints(
        hotwords=tuple(hotwords),
        context=context,
        boost=getattr(args, "hotword_boost", None),
    )


def _write_events(events, output: Path | None) -> int:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    destination = output.open("w", encoding="utf-8") if output else None
    try:
        for event in events:
            line = json.dumps(event.to_dict(), ensure_ascii=False)
            if destination:
                destination.write(line + "\n")
                destination.flush()
            else:
                print(line, flush=True)
        return 0
    finally:
        if destination:
            destination.close()


def transcribe_file(args) -> int:
    profile = select_execution_profile(
        args.execution_profile, requested_device=args.device
    )
    explicit_device = args.device != "auto" or "TURNALIGN_DEVICE" in os.environ
    asr_device = _resolved_device(args.device) if explicit_device else profile.asr_device
    asr_config = _config(args, device=asr_device)
    backend = create_asr(args.backend, asr_config)
    vad_backend = None
    if args.vad_backend != "none":
        vad_options = _extra_options(args.vad_option)
        if args.vad_backend == "energy":
            vad_options.setdefault("min_silence_seconds", args.silence_seconds)
            vad_options.setdefault("max_segment_seconds", args.max_utterance_seconds)
        elif args.vad_backend == "fsmn-vad":
            vad_options.setdefault("device", profile.vad_device)
        vad_backend = create_component("vad", args.vad_backend, vad_options)
    aligner_options = _extra_options(args.aligner_option)
    diarizer_options = _extra_options(args.diarizer_option)
    if args.aligner == "paraformer":
        aligner_options.setdefault("device", profile.alignment_device)
        aligner_options.setdefault("batch_size", profile.alignment_batch_size)
    if args.diarizer == "campp":
        diarizer_options.setdefault("device", profile.diarization_device)
    aligner = create_component("alignment", args.aligner, aligner_options) if args.aligner else None
    diarizer = create_component("diarization", args.diarizer, diarizer_options) if args.diarizer else None
    parallel_diarization = bool(args.parallel_postprocess and diarizer is not None)
    if args.parallel_postprocess is None and diarizer is not None:
        parallel_diarization = profile.parallel_diarization
    decoded = file_chunks(args.source, args.chunk_ms, args.ffmpeg)
    recorded_audio = list(decoded) if parallel_diarization else None
    chunks = iter(recorded_audio) if recorded_audio is not None else decoded
    vad_output = args.vad_output
    if vad_backend is not None and vad_output is None and args.output is not None:
        vad_output = args.output.with_name(f"{args.output.stem}.vad.jsonl")
    if vad_output is not None:
        vad_output.parent.mkdir(parents=True, exist_ok=True)
    audit_file = vad_output.open("w", encoding="utf-8") if vad_output is not None else None

    def write_vad_audit(item: dict[str, object]) -> None:
        if audit_file is None:
            return
        audit_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        audit_file.flush()

    try:
        return _write_events(
            transcribe_events(
                chunks, backend, vad=vad_backend is not None,
                vad_threshold=args.vad_threshold,
                silence_seconds=args.silence_seconds,
                max_utterance_seconds=args.max_utterance_seconds,
                partial_seconds=args.partial_seconds,
                aligner=aligner,
                diarizer=diarizer,
                vad_backend=vad_backend,
                vad_audit=write_vad_audit,
                recorded_audio=recorded_audio,
                parallel_diarization=parallel_diarization,
                execution_profile=profile.name,
            ),
            args.output,
        )
    finally:
        if audit_file is not None:
            audit_file.close()


def listen(args) -> int:
    backend = create_asr(args.backend, _config(args))
    aligner = create_component("alignment", args.aligner, _extra_options(args.aligner_option)) if args.aligner else None
    diarizer = create_component("diarization", args.diarizer, _extra_options(args.diarizer_option)) if args.diarizer else None
    if args.warmup_file:
        try:
            list(backend.transcribe(file_chunks(args.warmup_file, args.chunk_ms, args.ffmpeg)))
        except Exception:
            backend.close()
            raise
    chunks = microphone_chunks(
        device=_input_device(args.input_device),
        sample_rate=args.sample_rate,
        channels=1,
        chunk_ms=args.chunk_ms,
        duration=args.duration,
    )
    return _write_events(
        transcribe_events(
            chunks, backend, live=True,
            vad_threshold=args.vad_threshold,
            silence_seconds=args.silence_seconds,
            max_utterance_seconds=args.max_utterance_seconds,
            partial_seconds=args.partial_seconds,
            aligner=aligner,
            diarizer=diarizer,
        ),
        args.output,
    )


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-whisper", choices=available("asr"))
    parser.add_argument("--model")
    parser.add_argument("--language")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type")
    parser.add_argument("--executable", help="Executable used by command-based backends")
    parser.add_argument("--model-path", help="Local model path used by command-based backends")
    parser.add_argument("--backend-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--hotword", action="append", default=[], help="Private phrase hint; repeat as needed")
    parser.add_argument(
        "--hotwords-file", action="append", default=[], type=Path,
        help="UTF-8 file with one private phrase per line",
    )
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument("--context", help="Private topic context for prompt-capable ASR")
    context_group.add_argument("--context-file", type=Path, help="UTF-8 private topic context file")
    parser.add_argument(
        "--hotword-boost", type=float,
        help="Numeric boost for backends that explicitly support weighted hotwords",
    )


def _add_segmentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vad-threshold", type=float, default=0.012)
    parser.add_argument("--silence-seconds", type=float, default=0.7)
    parser.add_argument("--max-utterance-seconds", type=float, default=20.0)
    parser.add_argument("--partial-seconds", type=float, default=2.0)


def _add_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aligner", help="Alignment plugin entry-point name")
    parser.add_argument("--aligner-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--diarizer", help="Diarization plugin entry-point name")
    parser.add_argument("--diarizer-option", action="append", default=[], metavar="KEY=VALUE")


def _input_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(prog="turnalign")
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser("replay", help="Replay transcript JSONL as common events")
    replay_parser.add_argument("source", type=Path)
    replay_parser.add_argument("--output", type=Path)
    validate_parser = commands.add_parser("validate-events", help="Validate a common event JSONL file")
    validate_parser.add_argument("source", type=Path)
    doctor_parser = commands.add_parser("doctor", help="Detect and select the local inference device")
    doctor_parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda[:index], rocm[:index], or mps (TURNALIGN_DEVICE overrides it)",
    )
    backends_parser = commands.add_parser("backends", help="List available built-in and plugin components")
    backends_parser.set_defaults(command="backends")
    profiles_parser = commands.add_parser("profiles", help="List cross-platform execution profiles")
    profiles_parser.set_defaults(command="profiles")
    devices_parser = commands.add_parser("audio-devices", help="List microphone input devices")
    devices_parser.set_defaults(command="audio-devices")
    record_parser = commands.add_parser("record", help="Record microphone audio to a local WAV file")
    record_parser.add_argument("output", type=Path)
    record_parser.add_argument("--duration", type=float, required=True)
    record_parser.add_argument("--input-device")
    record_parser.add_argument("--sample-rate", type=int, default=16_000)
    record_parser.add_argument("--chunk-ms", type=int, default=100)
    transcribe_parser = commands.add_parser("transcribe", help="Transcribe a local audio file")
    transcribe_parser.add_argument("source", type=Path)
    transcribe_parser.add_argument("--output", type=Path)
    transcribe_parser.add_argument("--chunk-ms", type=int, default=500)
    transcribe_parser.add_argument("--ffmpeg", default="ffmpeg")
    vad_group = transcribe_parser.add_mutually_exclusive_group()
    vad_group.add_argument(
        "--vad", dest="vad_backend", action="store_const", const="energy",
        help="Deprecated alias for --vad-backend energy",
    )
    vad_group.add_argument("--vad-backend", help="VAD component name (default: energy)")
    vad_group.add_argument("--no-vad", dest="vad_backend", action="store_const", const="none")
    transcribe_parser.set_defaults(vad_backend="energy")
    transcribe_parser.add_argument("--vad-option", action="append", default=[], metavar="KEY=VALUE")
    transcribe_parser.add_argument("--vad-output", type=Path, help="VAD speech/silence audit JSONL")
    transcribe_parser.add_argument(
        "--execution-profile", choices=PROFILE_NAMES, default="auto",
        help="Cross-platform device and scheduling policy (default: auto)",
    )
    transcribe_parser.add_argument(
        "--parallel-postprocess",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run CPU diarization alongside GPU ASR (default: auto by device)",
    )
    _add_backend_arguments(transcribe_parser)
    _add_segmentation_arguments(transcribe_parser)
    _add_postprocess_arguments(transcribe_parser)
    listen_parser = commands.add_parser("listen", help="Transcribe the local microphone until Ctrl+C")
    listen_parser.add_argument("--output", type=Path)
    listen_parser.add_argument("--input-device")
    listen_parser.add_argument("--sample-rate", type=int, default=16_000)
    listen_parser.add_argument("--chunk-ms", type=int, default=100)
    listen_parser.add_argument("--duration", type=float)
    listen_parser.add_argument("--warmup-file", type=Path)
    listen_parser.add_argument("--ffmpeg", default="ffmpeg")
    _add_backend_arguments(listen_parser)
    _add_segmentation_arguments(listen_parser)
    _add_postprocess_arguments(listen_parser)
    serve_parser = commands.add_parser("serve", help="Serve raw PCM transcription over WebSocket")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--backend", default="transformers-whisper", choices=available("asr"))
    serve_parser.add_argument("--model")
    serve_parser.add_argument("--device", default="auto")
    serve_parser.add_argument("--warmup-file", type=Path)
    serve_parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.command == "replay":
        return replay(args.source, args.output)
    if args.command == "validate-events":
        return validate_events(args.source)
    if args.command == "doctor":
        print(json.dumps(runtime_report(args.device), ensure_ascii=False, indent=2))
        return 0
    if args.command == "backends":
        print(json.dumps({
            kind: available(kind) for kind in ("asr", "vad", "alignment", "diarization")
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "profiles":
        automatic = select_execution_profile("auto")
        print(json.dumps({
            "auto_selected": automatic.to_dict(),
            "profiles": profile_catalog(),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audio-devices":
        print(json.dumps(input_devices(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "record":
        write_wave(args.output, microphone_chunks(
            device=_input_device(args.input_device), sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms, duration=args.duration,
        ))
        return 0
    if args.command == "transcribe":
        return transcribe_file(args)
    if args.command == "listen":
        try:
            return listen(args)
        except KeyboardInterrupt:
            return 130
    if args.command == "serve":
        asyncio.run(serve(
            args.host, args.port, default_backend=args.backend,
            default_model=args.model, default_device=_resolved_device(args.device),
            warmup_file=args.warmup_file, ffmpeg=args.ffmpeg,
        ))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
