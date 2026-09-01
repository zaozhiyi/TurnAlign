from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import stat
from contextlib import suppress
from dataclasses import replace
from itertools import chain
from pathlib import Path
from time import perf_counter

from .audio import (
    AudioTimeline,
    file_chunks,
    input_devices,
    microphone_chunks,
    write_wave,
)
from .backends.jsonl import JsonlBackend
from .deployment_rehearsal import (
    RehearsalProbeConfig,
    run_deployment_activation,
    run_deployment_rehearsal,
)
from .devices import runtime_report
from .evaluation import TextNormalization, evaluate_events, evaluate_quality_gate
from .exporters import render_srt, render_text
from .hints import AsrHints
from .jsonutil import strict_json_loads, strict_json_object
from .models import TranscriptEvent
from .offline import OfflineRefinementPipeline
from .pipelines import TwoPassPipeline
from .plugins import AsrConfig
from .policy import AUTH_TOKEN_MAX_BYTES, ServerPolicy, validate_auth_token
from .production_gate import (
    REQUIRED_ARTIFACT_KINDS,
    create_host_profile,
    create_model_manifest,
    run_production_gate,
    write_json_report,
)
from .profiles import PROFILE_NAMES, profile_catalog, select_execution_profile
from .realtime import RealtimePipeline
from .registry import available, create_asr, create_component
from .release_gate import run_release_gate
from .server import serve
from .session import transcribe_events
from .validation import EventStreamValidator
from .websocket_gate import run_websocket_gate

_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _source_commit_argument(value: str) -> str:
    if _SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "source commit must be a lowercase 40-character Git commit"
        )
    return value


def _sha256_file(path: Path) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and path.is_symlink():
        raise ValueError(f"evidence must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open evidence file: {path}") from error
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"evidence must be a regular non-symlink file: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _read_auth_token_file(path: Path) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and path.is_symlink():
        raise ValueError(f"cannot securely open authentication token file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open authentication token file: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"authentication token path is not a regular file: {path}")
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(
                "authentication token file must not be accessible by group or others"
            )
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            raw_token = source.read(AUTH_TOKEN_MAX_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw_token) > AUTH_TOKEN_MAX_BYTES:
        raise ValueError(
            f"authentication token file exceeds {AUTH_TOKEN_MAX_BYTES} bytes"
        )
    if raw_token.endswith(b"\r\n"):
        raw_token = raw_token[:-2]
    elif raw_token.endswith((b"\r", b"\n")):
        raw_token = raw_token[:-1]
    if not raw_token or b"\r" in raw_token or b"\n" in raw_token or b"\x00" in raw_token:
        raise ValueError("authentication token file must contain one non-empty token")
    try:
        return validate_auth_token(raw_token.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("authentication token file must contain UTF-8 text") from error


def _authentication_token(args) -> str | None:
    environment_name = getattr(args, "auth_token_env", None)
    token_path = getattr(args, "auth_token_file", None)
    if environment_name:
        token = os.environ.get(environment_name)
        if not token:
            raise ValueError(
                f"authentication token environment variable is empty: {environment_name}"
            )
        return validate_auth_token(token)
    if token_path is not None:
        return _read_auth_token_file(token_path)
    return None


def _emit_gate_report(report, path: Path | None) -> None:
    payload = report.to_dict()
    if path is not None:
        write_json_report(path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def _artifact_argument(value: str) -> tuple[str, Path]:
    kind, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("artifact must use KIND=PATH")
    if kind not in REQUIRED_ARTIFACT_KINDS:
        choices = ", ".join(sorted(REQUIRED_ARTIFACT_KINDS))
        raise argparse.ArgumentTypeError(f"artifact KIND must be one of: {choices}")
    return kind, Path(raw_path)


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
            line = json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False)
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
        line = json.dumps(end_event.to_dict(), ensure_ascii=False, allow_nan=False)
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
                event = TranscriptEvent.from_dict(
                    strict_json_object(line, label=f"{source}:{line_number}")
                )
                validator.accept(event)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
            count += 1
            commits += event.kind == "commit"
    if not validator.ended:
        raise ValueError(f"{source}: missing end event")
    print(json.dumps({"status": "ok", "events": count, "commits": commits}, ensure_ascii=False))
    return 0


def _read_events(
    source: Path,
    *,
    require_complete: bool = False,
) -> list[TranscriptEvent]:
    events = []
    validator = EventStreamValidator() if require_complete else None
    with source.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                event = TranscriptEvent.from_dict(
                    strict_json_object(line, label=f"{source}:{line_number}")
                )
                if validator is not None:
                    validator.accept(event)
                events.append(event)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    if validator is not None and not validator.ended:
        raise ValueError(f"{source}: missing end event")
    return events


def _text_normalization(args) -> TextNormalization:
    return TextNormalization(
        unicode_form=args.unicode_normalization,
        case_sensitive=not args.ignore_case,
        punctuation_sensitive=not args.ignore_punctuation,
    )


def evaluate_files(reference: Path, hypothesis: Path, args) -> int:
    report = evaluate_events(
        _read_events(reference),
        _read_events(hypothesis),
        text_normalization=_text_normalization(args),
    )
    print(json.dumps(
        report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ))
    return 0


def quality_gate_files(args) -> int:
    source_commit = getattr(args, "source_commit", None)
    report = evaluate_quality_gate(
        _read_events(args.reference, require_complete=True),
        _read_events(args.hypothesis, require_complete=True),
        max_character_error_rate=args.max_cer,
        max_word_error_rate=args.max_wer,
        max_diarization_error_rate=args.max_diarization_error,
        max_revision_updates_per_segment=args.max_revision_updates_per_segment,
        min_reference_segments=args.min_reference_segments,
        min_reference_characters=args.min_reference_characters,
        min_reference_speech_seconds=args.min_reference_speech_seconds,
        text_normalization=_text_normalization(args),
        source_commit=source_commit,
        reference_sha256=(
            _sha256_file(args.reference) if source_commit is not None else None
        ),
        hypothesis_sha256=(
            _sha256_file(args.hypothesis) if source_commit is not None else None
        ),
    )
    _emit_gate_report(report, getattr(args, "report", None))
    return 0 if report.passed else 1


def release_gate(args) -> int:
    source_commit = getattr(args, "source_commit", None)
    input_audio_sha256 = (
        _sha256_file(args.source) if source_commit is not None else None
    )
    chunks = iter(file_chunks(args.source, args.chunk_ms, args.ffmpeg))
    try:
        first_chunk = next(chunks)
    except StopIteration as error:
        raise ValueError(f"{args.source}: audio input is empty") from error
    chunks = chain((first_chunk,), chunks)

    destination = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        destination = args.output.open("w", encoding="utf-8")

    def write_event(event: TranscriptEvent) -> None:
        if destination is not None:
            destination.write(json.dumps(
                event.to_dict(), ensure_ascii=False, allow_nan=False
            ) + "\n")
            destination.flush()

    try:
        initialization_started = perf_counter()
        backend = create_asr(args.backend, _config(args))
        initialization_seconds = perf_counter() - initialization_started
        report = run_release_gate(
            chunks,
            backend,
            max_realtime_factor=args.max_realtime_factor,
            max_first_partial_seconds=args.max_first_partial_seconds,
            max_first_commit_seconds=args.max_first_commit_seconds,
            max_initialization_seconds=args.max_initialization_seconds,
            initialization_seconds=initialization_seconds,
            min_audio_seconds=args.min_audio_seconds,
            min_commits=args.min_commits,
            require_partial=args.require_partial,
            require_native_streaming=args.require_native_streaming,
            require_immutable_model_revision=args.require_immutable_model_revision,
            source_commit=source_commit,
            input_audio_sha256=input_audio_sha256,
            event_sink=write_event,
        )
    finally:
        if destination is not None:
            destination.close()
    _emit_gate_report(report, getattr(args, "report", None))
    return 0 if report.passed else 1


def websocket_gate(args) -> int:
    auth_token = _authentication_token(args)
    report = asyncio.run(run_websocket_gate(
        args.uri,
        sessions=args.sessions,
        audio_seconds=args.audio_seconds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
        timeout=args.timeout,
        min_commits=args.min_commits,
        min_audio_acks=args.min_audio_acks,
        max_dropped_partials=args.max_dropped_partials,
        max_backpressure_pauses=args.max_backpressure_pauses,
        max_ready_seconds=args.max_ready_seconds,
        max_total_seconds=args.max_total_seconds,
        realtime=args.realtime,
        backend=args.backend,
        model=args.model,
        language=args.language,
        compute_type=args.compute_type,
        auth_token=auth_token,
        verify_recovery=args.verify_recovery,
        recovery_resume_timeout=args.recovery_resume_timeout,
        source_commit=getattr(args, "source_commit", None),
    ))
    _emit_gate_report(report, getattr(args, "report", None))
    return 0 if report.passed else 1


def production_gate(args) -> int:
    report = run_production_gate(
        args.release_report,
        args.quality_report,
        args.websocket_report,
        source_commit=args.source_commit,
        artifacts=args.artifact,
    )
    _emit_gate_report(report, args.report)
    return 0 if report.passed else 1


def _deployment_probe(args) -> RehearsalProbeConfig:
    return RehearsalProbeConfig(
        sessions=args.sessions,
        audio_seconds=args.audio_seconds,
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
        timeout=args.timeout,
        min_commits=args.min_commits,
        min_audio_acks=args.min_audio_acks,
        max_dropped_partials=args.max_dropped_partials,
        max_backpressure_pauses=args.max_backpressure_pauses,
        max_ready_seconds=args.max_ready_seconds,
        max_total_seconds=args.max_total_seconds,
        recovery_resume_timeout=args.recovery_resume_timeout,
        backend=args.backend,
        model=args.model,
        language=args.language,
        compute_type=args.compute_type,
    )


def deployment_rehearsal(args) -> int:
    auth_token = _authentication_token(args)
    report = asyncio.run(run_deployment_rehearsal(
        args.previous_commit,
        args.candidate_commit,
        args.uri,
        restart_timeout=args.restart_timeout,
        readiness_timeout=args.readiness_timeout,
        readiness_interval=args.readiness_interval,
        probe=_deployment_probe(args),
        auth_token=auth_token,
    ))
    _emit_gate_report(report, args.report)
    return 0 if report.passed else 1


def deployment_activation(args) -> int:
    auth_token = _authentication_token(args)
    report = asyncio.run(run_deployment_activation(
        args.previous_commit,
        args.candidate_commit,
        args.uri,
        restart_timeout=args.restart_timeout,
        readiness_timeout=args.readiness_timeout,
        readiness_interval=args.readiness_interval,
        probe=_deployment_probe(args),
        auth_token=auth_token,
    ))
    _emit_gate_report(report, args.report)
    return 0 if report.passed else 1


def model_manifest(args) -> int:
    payload = create_model_manifest(
        args.model_id,
        args.model_revision,
        args.file,
    )
    write_json_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def host_profile(args) -> int:
    payload = create_host_profile(args.source_commit, args.artifact)
    write_json_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _effective_device(requested: str) -> str:
    return os.environ.get("TURNALIGN_DEVICE", requested).strip().lower()


def _resolved_device(requested: str) -> str:
    effective = _effective_device(requested)
    if effective != "auto":
        return effective
    selected = runtime_report("auto")["selected"]
    accelerator = selected["accelerator"]
    if accelerator in {"cuda", "rocm"}:
        index = str(selected["device"]).split(":")[-1]
        return f"{accelerator}:{index}"
    return accelerator


def _profile_requested_device(backend: str, requested: str) -> str:
    """Use conservative post-processing defaults for explicit whisper.cpp Vulkan."""

    normalized = _effective_device(requested)
    if backend == "whisper-cpp" and re.fullmatch(r"vulkan(?::\d+)?", normalized):
        return "cpu"
    return requested


def _profile_name(backend: str, requested_profile: str, requested_device: str) -> str:
    """Avoid generic device probing for whisper.cpp-owned Vulkan devices."""

    if (
        requested_profile == "auto"
        and backend == "whisper-cpp"
        and re.fullmatch(r"vulkan(?::\d+)?", _effective_device(requested_device))
    ):
        return "cpu-low-memory"
    return requested_profile


def _extra_options(items: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"backend option must be key=value: {item}")
        key, raw = item.split("=", 1)
        if not key or key != key.strip() or any(character.isspace() for character in key):
            raise ValueError(f"backend option has an invalid key: {key!r}")
        if key in result:
            raise ValueError(f"duplicate backend option: {key}")
        try:
            result[key] = strict_json_loads(raw)
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


def _write_events(events, output: Path | None, output_format: str = "jsonl") -> int:
    destination = None
    try:
        if output_format != "jsonl":
            rendered = (
                render_srt(events)
                if output_format == "srt"
                else render_text(events)
            )
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
            return 0
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            destination = output.open("w", encoding="utf-8")
        for event in events:
            line = json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False)
            if destination:
                destination.write(line + "\n")
                destination.flush()
            else:
                print(line, flush=True)
        return 0
    finally:
        if destination:
            destination.close()
        close = getattr(events, "close", None)
        if callable(close):
            close()


def _close_resources(resources: list[object]) -> None:
    """Best-effort cleanup for failures before a pipeline owns its resources."""
    for resource in reversed(resources):
        close = getattr(resource, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


def transcribe_file(args) -> int:
    profile = select_execution_profile(
        _profile_name(args.backend, args.execution_profile, args.device),
        requested_device=_profile_requested_device(args.backend, args.device),
    )
    explicit_device = args.device != "auto" or "TURNALIGN_DEVICE" in os.environ
    asr_device = _resolved_device(args.device) if explicit_device else profile.asr_device
    asr_config = _config(args, device=asr_device)
    resources: list[object] = []
    recorded_timeline = None
    audit_file = None
    resources_transferred = False
    try:
        backend = create_asr(args.backend, asr_config)
        resources.append(backend)
        vad_backend = None
        if args.vad_backend != "none":
            vad_options = _extra_options(args.vad_option)
            if args.vad_backend == "energy":
                vad_options.setdefault("min_silence_seconds", args.silence_seconds)
                vad_options.setdefault("max_segment_seconds", args.max_utterance_seconds)
            elif args.vad_backend == "fsmn-vad":
                vad_options.setdefault("device", profile.vad_device)
            vad_backend = create_component("vad", args.vad_backend, vad_options)
            resources.append(vad_backend)
        aligner_options = _extra_options(args.aligner_option)
        diarizer_options = _extra_options(args.diarizer_option)
        if args.aligner == "paraformer":
            aligner_options.setdefault("device", profile.alignment_device)
            aligner_options.setdefault("batch_size", profile.alignment_batch_size)
        if args.diarizer == "campp":
            diarizer_options.setdefault("device", profile.diarization_device)
        aligner = (
            create_component("alignment", args.aligner, aligner_options)
            if args.aligner else None
        )
        if aligner is not None:
            resources.append(aligner)
        diarizer = (
            create_component("diarization", args.diarizer, diarizer_options)
            if args.diarizer else None
        )
        if diarizer is not None:
            resources.append(diarizer)
        parallel_diarization = bool(args.parallel_postprocess and diarizer is not None)
        if args.parallel_postprocess is None and diarizer is not None:
            parallel_diarization = profile.parallel_diarization
        decoded = file_chunks(args.source, args.chunk_ms, args.ffmpeg)
        recorded_timeline = (
            AudioTimeline.from_chunks(decoded) if parallel_diarization else None
        )
        chunks = (
            recorded_timeline.iter_chunks(args.chunk_ms)
            if recorded_timeline is not None else decoded
        )
        vad_output = args.vad_output
        if vad_backend is not None and vad_output is None and args.output is not None:
            vad_output = args.output.with_name(f"{args.output.stem}.vad.jsonl")
        if vad_output is not None:
            vad_output.parent.mkdir(parents=True, exist_ok=True)
            audit_file = vad_output.open("w", encoding="utf-8")

        def write_vad_audit(item: dict[str, object]) -> None:
            if audit_file is None:
                return
            audit_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            audit_file.flush()

        def events():
            nonlocal resources_transferred
            resources_transferred = True
            yield from transcribe_events(
                chunks, backend, vad=vad_backend is not None,
                vad_threshold=args.vad_threshold,
                silence_seconds=args.silence_seconds,
                max_utterance_seconds=args.max_utterance_seconds,
                partial_seconds=args.partial_seconds,
                aligner=aligner,
                diarizer=diarizer,
                vad_backend=vad_backend,
                vad_audit=write_vad_audit,
                recorded_timeline=recorded_timeline,
                parallel_diarization=parallel_diarization,
                execution_profile=profile.name,
            )

        return _write_events(
            events(),
            args.output,
            args.output_format,
        )
    finally:
        if audit_file is not None:
            audit_file.close()
        if recorded_timeline is not None:
            recorded_timeline.close()
        if not resources_transferred:
            _close_resources(resources)


def listen(args) -> int:
    config = _config(args)
    resources: list[object] = []
    resources_transferred = False
    try:
        backend = create_asr(args.backend, config)
        resources.append(backend)
        aligner = (
            create_component(
                "alignment", args.aligner, _extra_options(args.aligner_option)
            ) if args.aligner else None
        )
        if aligner is not None:
            resources.append(aligner)
        diarizer = (
            create_component(
                "diarization", args.diarizer, _extra_options(args.diarizer_option)
            ) if args.diarizer else None
        )
        if diarizer is not None:
            resources.append(diarizer)
        online_diarizer = (
            create_component(
                "online_diarization",
                args.online_diarizer,
                _extra_options(args.online_diarizer_option),
            ) if args.online_diarizer else None
        )
        if online_diarizer is not None:
            resources.append(online_diarizer)
        if args.warmup_file:
            list(backend.transcribe(file_chunks(args.warmup_file, args.chunk_ms, args.ffmpeg)))
        chunks = microphone_chunks(
            device=_input_device(args.input_device),
            sample_rate=args.sample_rate,
            channels=1,
            chunk_ms=args.chunk_ms,
            duration=args.duration,
        )
        if args.refinement_backend:
            refinement_backend = create_asr(
                args.refinement_backend,
                replace(config, model=args.refinement_model),
            )
            resources.append(refinement_backend)
            pipeline_events = TwoPassPipeline(
                RealtimePipeline(
                    backend,
                    vad_threshold=args.vad_threshold,
                    silence_seconds=args.silence_seconds,
                    max_utterance_seconds=args.max_utterance_seconds,
                    partial_seconds=args.partial_seconds,
                    online_diarizer=online_diarizer,
                ),
                OfflineRefinementPipeline(
                    refinement_backend,
                    aligner=aligner,
                    diarizer=diarizer,
                ),
            ).events(chunks)
        else:
            pipeline_events = transcribe_events(
                chunks,
                backend,
                live=True,
                vad_threshold=args.vad_threshold,
                silence_seconds=args.silence_seconds,
                max_utterance_seconds=args.max_utterance_seconds,
                partial_seconds=args.partial_seconds,
                aligner=aligner,
                diarizer=diarizer,
                online_diarizer=online_diarizer,
            )

        def events():
            nonlocal resources_transferred
            resources_transferred = True
            yield from pipeline_events

        return _write_events(events(), args.output, args.output_format)
    finally:
        if not resources_transferred:
            _close_resources(resources)


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-whisper", choices=available("asr"))
    parser.add_argument("--model")
    parser.add_argument("--language")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda[:index], rocm[:index], mps, or whisper-cpp vulkan[:index]",
    )
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
    parser.add_argument("--online-diarizer", help="Online diarization plugin entry-point name")
    parser.add_argument(
        "--online-diarizer-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )


def _add_deployment_operation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("uri", help="Public wss:// production endpoint")
    parser.add_argument(
        "--previous-commit",
        type=_source_commit_argument,
        required=True,
    )
    parser.add_argument(
        "--candidate-commit",
        type=_source_commit_argument,
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--audio-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-commits", type=int, default=0)
    parser.add_argument("--min-audio-acks", type=int, default=1)
    parser.add_argument("--max-dropped-partials", type=int, default=0)
    parser.add_argument("--max-backpressure-pauses", type=int, default=0)
    parser.add_argument("--max-ready-seconds", type=float, default=30.0)
    parser.add_argument("--max-total-seconds", type=float, default=180.0)
    parser.add_argument("--recovery-resume-timeout", type=float, default=10.0)
    parser.add_argument("--restart-timeout", type=float, default=120.0)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--readiness-interval", type=float, default=1.0)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language")
    parser.add_argument("--compute-type")
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--auth-token-env",
        help="Environment variable containing the start-message auth token",
    )
    authentication.add_argument(
        "--auth-token-file",
        type=Path,
        help="Restricted file containing the start-message auth token",
    )


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
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Compare reference and hypothesis event JSONL",
    )
    evaluate_parser.add_argument("reference", type=Path)
    evaluate_parser.add_argument("hypothesis", type=Path)
    quality_parser = commands.add_parser(
        "quality-gate",
        help="Enforce labelled transcript and speaker quality thresholds",
    )
    quality_parser.add_argument("reference", type=Path)
    quality_parser.add_argument("hypothesis", type=Path)
    quality_parser.add_argument("--report", type=Path, help="Persist the JSON gate report")
    quality_parser.add_argument("--source-commit", type=_source_commit_argument)
    quality_parser.add_argument("--max-cer", type=float)
    quality_parser.add_argument("--max-wer", type=float)
    quality_parser.add_argument("--max-diarization-error", type=float)
    quality_parser.add_argument("--max-revision-updates-per-segment", type=float)
    quality_parser.add_argument("--min-reference-segments", type=int, default=1)
    quality_parser.add_argument("--min-reference-characters", type=int, default=1)
    quality_parser.add_argument(
        "--min-reference-speech-seconds",
        type=float,
        default=0.0,
    )
    for evaluation_parser in (evaluate_parser, quality_parser):
        evaluation_parser.add_argument(
            "--unicode-normalization",
            choices=("none", "NFC", "NFKC"),
            default="none",
        )
        evaluation_parser.add_argument("--ignore-case", action="store_true")
        evaluation_parser.add_argument("--ignore-punctuation", action="store_true")
    gate_parser = commands.add_parser(
        "release-gate",
        help="Run a real ASR backend against measurable release thresholds",
    )
    gate_parser.add_argument("source", type=Path)
    gate_parser.add_argument("--output", type=Path, help="Optional event JSONL evidence")
    gate_parser.add_argument("--report", type=Path, help="Persist the JSON gate report")
    gate_parser.add_argument("--source-commit", type=_source_commit_argument)
    gate_parser.add_argument("--chunk-ms", type=int, default=100)
    gate_parser.add_argument("--ffmpeg", default="ffmpeg")
    gate_parser.add_argument("--max-realtime-factor", type=float, default=1.0)
    gate_parser.add_argument("--max-first-partial-seconds", type=float, default=3.0)
    gate_parser.add_argument(
        "--max-first-commit-seconds",
        type=float,
        help="Optional wall-clock ceiling from inference start to first commit",
    )
    gate_parser.add_argument("--max-initialization-seconds", type=float, default=120.0)
    gate_parser.add_argument("--min-audio-seconds", type=float, default=10.0)
    gate_parser.add_argument("--min-commits", type=int, default=1)
    gate_parser.add_argument(
        "--require-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    gate_parser.add_argument(
        "--require-native-streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    gate_parser.add_argument(
        "--require-immutable-model-revision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the backend revision to be a 40- or 64-character commit hash",
    )
    _add_backend_arguments(gate_parser)
    websocket_gate_parser = commands.add_parser(
        "websocket-gate",
        help="Run concurrent transport and lifecycle checks against a deployed server",
    )
    websocket_gate_parser.add_argument("uri")
    websocket_gate_parser.add_argument("--report", type=Path, help="Persist the JSON gate report")
    websocket_gate_parser.add_argument("--source-commit", type=_source_commit_argument)
    websocket_gate_parser.add_argument("--sessions", type=int, default=4)
    websocket_gate_parser.add_argument("--audio-seconds", type=float, default=5.0)
    websocket_gate_parser.add_argument("--sample-rate", type=int, default=16_000)
    websocket_gate_parser.add_argument("--channels", type=int, default=1)
    websocket_gate_parser.add_argument("--frame-ms", type=int, default=100)
    websocket_gate_parser.add_argument("--timeout", type=float, default=120.0)
    websocket_gate_parser.add_argument("--min-commits", type=int, default=0)
    websocket_gate_parser.add_argument("--min-audio-acks", type=int, default=1)
    websocket_gate_parser.add_argument(
        "--max-dropped-partials",
        type=int,
        default=0,
        help="Maximum dropped partial events allowed per session",
    )
    websocket_gate_parser.add_argument(
        "--max-backpressure-pauses",
        type=int,
        help="Optional maximum flow-control pauses allowed per session",
    )
    websocket_gate_parser.add_argument("--max-ready-seconds", type=float)
    websocket_gate_parser.add_argument("--max-total-seconds", type=float)
    websocket_gate_parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pace generated silence in real time instead of sending a burst",
    )
    websocket_gate_parser.add_argument(
        "--verify-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject one acknowledged disconnect and require successful resume",
    )
    websocket_gate_parser.add_argument(
        "--recovery-resume-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for a disconnected recovery session to become resumable",
    )
    websocket_gate_parser.add_argument("--backend")
    websocket_gate_parser.add_argument("--model")
    websocket_gate_parser.add_argument("--language")
    websocket_gate_parser.add_argument("--compute-type")
    websocket_gate_auth = websocket_gate_parser.add_mutually_exclusive_group()
    websocket_gate_auth.add_argument(
        "--auth-token-env",
        help="Environment variable containing the start-message auth token",
    )
    websocket_gate_auth.add_argument(
        "--auth-token-file",
        type=Path,
        help="Restricted file containing the start-message auth token",
    )
    rehearsal_parser = commands.add_parser(
        "deployment-rehearsal",
        help=(
            "Atomically roll back and restore a Linux production release while "
            "retaining readiness and public WebSocket evidence"
        ),
    )
    _add_deployment_operation_arguments(rehearsal_parser)
    activation_parser = commands.add_parser(
        "deployment-activate",
        help=(
            "Atomically activate and probe a Linux production release, restoring "
            "the preceding release on failure"
        ),
    )
    _add_deployment_operation_arguments(activation_parser)
    production_gate_parser = commands.add_parser(
        "production-gate",
        help="Bind production gate reports and immutable artifacts into one verdict",
    )
    production_gate_parser.add_argument("release_report", type=Path)
    production_gate_parser.add_argument("quality_report", type=Path)
    production_gate_parser.add_argument("websocket_report", type=Path)
    production_gate_parser.add_argument(
        "--source-commit", type=_source_commit_argument, required=True
    )
    production_gate_parser.add_argument(
        "--artifact",
        action="append",
        type=_artifact_argument,
        required=True,
        metavar="KIND=PATH",
        help="Immutable release artifact; repeat for every required kind",
    )
    production_gate_parser.add_argument(
        "--report",
        type=Path,
        help="Persist the aggregate JSON verdict",
    )
    model_manifest_parser = commands.add_parser(
        "model-manifest",
        help="Hash retained model files into production provenance evidence",
    )
    model_manifest_parser.add_argument("--model-id", required=True)
    model_manifest_parser.add_argument("--model-revision", required=True)
    model_manifest_parser.add_argument(
        "--file",
        action="append",
        type=Path,
        required=True,
        help="Retained model file or archive; repeat for every file",
    )
    model_manifest_parser.add_argument("--output", type=Path, required=True)
    host_profile_parser = commands.add_parser(
        "host-profile",
        help=(
            "Bind the active versioned Wheel runtime and target host to retained "
            "production artifacts"
        ),
    )
    host_profile_parser.add_argument(
        "--source-commit",
        type=_source_commit_argument,
        help="Expected commit; defaults to the identity embedded in the installed Wheel",
    )
    host_profile_parser.add_argument(
        "--artifact",
        action="append",
        type=_artifact_argument,
        required=True,
        metavar="KIND=PATH",
        help="Retained artifact other than host-profile; repeat for every kind",
    )
    host_profile_parser.add_argument("--output", type=Path, required=True)
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
    transcribe_parser.add_argument(
        "--output-format",
        choices=("jsonl", "srt", "txt"),
        default="jsonl",
    )
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
    listen_parser.add_argument(
        "--output-format",
        choices=("jsonl", "srt", "txt"),
        default="jsonl",
    )
    listen_parser.add_argument("--input-device")
    listen_parser.add_argument("--sample-rate", type=int, default=16_000)
    listen_parser.add_argument("--chunk-ms", type=int, default=100)
    listen_parser.add_argument("--duration", type=float)
    listen_parser.add_argument("--warmup-file", type=Path)
    listen_parser.add_argument("--ffmpeg", default="ffmpeg")
    listen_parser.add_argument(
        "--refinement-backend",
        choices=available("asr"),
        help="Optional second-pass ASR backend run after recording",
    )
    listen_parser.add_argument("--refinement-model")
    _add_backend_arguments(listen_parser)
    _add_segmentation_arguments(listen_parser)
    _add_postprocess_arguments(listen_parser)
    serve_parser = commands.add_parser("serve", help="Serve raw PCM transcription over WebSocket")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="Server log level written to stderr",
    )
    serve_parser.add_argument("--backend", default="transformers-whisper", choices=available("asr"))
    serve_parser.add_argument("--model")
    serve_parser.add_argument("--device", default="auto")
    serve_parser.add_argument("--language")
    serve_parser.add_argument("--compute-type")
    serve_parser.add_argument("--executable", help="Default executable for command backends")
    serve_parser.add_argument("--model-path", help="Default local model path")
    serve_parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Trusted server-side backend option; repeat as needed",
    )
    serve_parser.add_argument("--warmup-file", type=Path)
    serve_parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all default backend replicas before accepting connections",
    )
    serve_parser.add_argument(
        "--require-immutable-model-revision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject backends without an exact 40- or 64-character model commit",
    )
    serve_parser.add_argument("--ffmpeg", default="ffmpeg")
    serve_parser.add_argument(
        "--allow-backend",
        action="append",
        default=[],
        choices=available("asr"),
        help="Additional backend clients may request; repeat as needed",
    )
    serve_parser.add_argument(
        "--allow-model",
        action="append",
        default=[],
        help="Additional model identifier clients may request; repeat as needed",
    )
    serve_parser.add_argument(
        "--allow-language",
        action="append",
        default=[],
        help="Additional language clients may request; repeat as needed",
    )
    serve_parser.add_argument(
        "--allow-compute-type",
        action="append",
        default=[],
        help="Additional compute type clients may request; repeat as needed",
    )
    serve_parser.add_argument(
        "--allow-component",
        action="append",
        default=[],
        help="Alignment or diarization component clients may request",
    )
    serve_parser.add_argument("--allow-client-paths", action="store_true")
    serve_parser.add_argument("--allow-component-options", action="store_true")
    serve_parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding beyond localhost; deploy TLS/auth in front",
    )
    serve_auth = serve_parser.add_mutually_exclusive_group()
    serve_auth.add_argument(
        "--auth-token-env",
        help="Environment variable containing the required start-message auth token",
    )
    serve_auth.add_argument(
        "--auth-token-file",
        type=Path,
        help="Restricted file containing the required start-message auth token",
    )
    serve_parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Exact browser Origin allowed to open WebSockets; repeat as needed",
    )
    serve_parser.add_argument("--max-session-seconds", type=float, default=14_400.0)
    serve_parser.add_argument("--internal-chunk-ms", type=int, default=100)
    serve_parser.add_argument(
        "--max-control-message-bytes",
        type=int,
        default=64 * 1024,
        help="Maximum UTF-8 size of start and later JSON control messages",
    )
    serve_parser.add_argument(
        "--initialization-timeout",
        type=float,
        default=120.0,
        help="Seconds allowed for model and component initialization",
    )
    serve_parser.add_argument(
        "--worker-shutdown-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for a cancelled inference worker to stop",
    )
    serve_parser.add_argument(
        "--finalization-timeout",
        type=float,
        default=120.0,
        help="Seconds allowed for final ASR and post-processing after end",
    )
    serve_parser.add_argument(
        "--output-backpressure-timeout",
        type=float,
        default=5.0,
        help="Seconds a durable output event may wait for a stalled client",
    )
    serve_parser.add_argument(
        "--max-recovery-events",
        type=int,
        default=2_048,
        help="Maximum replayable events retained per in-process recovery session",
    )
    serve_parser.add_argument(
        "--max-recovery-event-kib",
        type=int,
        default=512,
        help="Maximum serialized size of one recoverable WebSocket event in KiB",
    )
    serve_parser.add_argument(
        "--max-recovery-events-mib",
        type=int,
        default=8,
        help="Maximum serialized recovery-event data retained per session in MiB",
    )
    serve_parser.add_argument(
        "--max-recovery-sessions",
        type=int,
        default=32,
        help="Maximum active or resumable recovery sessions retained per process",
    )
    serve_parser.add_argument(
        "--max-recovery-audio-mib",
        type=int,
        default=512,
        help="Maximum temporary recovery audio per session in MiB",
    )
    serve_parser.add_argument(
        "--max-recovery-total-mib",
        type=int,
        default=2_048,
        help="Maximum temporary recovery audio retained by the process in MiB",
    )
    serve_parser.add_argument(
        "--recovery-ttl-seconds",
        type=float,
        default=300.0,
        help="Seconds an inactive recovery session remains resumable",
    )
    serve_parser.add_argument(
        "--max-concurrent-sessions",
        type=int,
        default=32,
        help="Maximum accepted WebSocket sessions per process",
    )
    serve_parser.add_argument(
        "--start-timeout",
        type=float,
        default=10.0,
        help="Seconds allowed for the first start message",
    )
    serve_parser.add_argument(
        "--client-idle-timeout",
        type=float,
        default=60.0,
        help="Seconds allowed between client audio or control frames",
    )
    serve_parser.add_argument(
        "--shutdown-grace-timeout",
        type=float,
        default=30.0,
        help="Seconds allowed for active handlers to stop after SIGTERM",
    )
    serve_parser.add_argument(
        "--backend-replicas",
        type=int,
        default=1,
        help="Loaded model instances allowed per backend configuration (1-8)",
    )
    args = parser.parse_args()
    if args.command == "replay":
        return replay(args.source, args.output)
    if args.command == "validate-events":
        return validate_events(args.source)
    if args.command == "evaluate":
        return evaluate_files(args.reference, args.hypothesis, args)
    if args.command == "quality-gate":
        return quality_gate_files(args)
    if args.command == "release-gate":
        return release_gate(args)
    if args.command == "websocket-gate":
        return websocket_gate(args)
    if args.command == "deployment-rehearsal":
        return deployment_rehearsal(args)
    if args.command == "deployment-activate":
        return deployment_activation(args)
    if args.command == "production-gate":
        return production_gate(args)
    if args.command == "model-manifest":
        return model_manifest(args)
    if args.command == "host-profile":
        return host_profile(args)
    if args.command == "doctor":
        print(json.dumps(runtime_report(args.device), ensure_ascii=False, indent=2))
        return 0
    if args.command == "backends":
        print(json.dumps({
            kind: available(kind)
            for kind in (
                "asr",
                "vad",
                "alignment",
                "diarization",
                "online_diarization",
            )
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
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        auth_token = _authentication_token(args)
        policy = ServerPolicy(
            allowed_backends=frozenset({args.backend, *args.allow_backend}),
            allowed_models=frozenset(
                model for model in [args.model, *args.allow_model] if model
            ),
            allowed_languages=frozenset(
                language for language in [args.language, *args.allow_language] if language
            ),
            allowed_compute_types=frozenset(
                value for value in [args.compute_type, *args.allow_compute_type] if value
            ),
            allowed_components=frozenset(args.allow_component),
            allow_client_paths=args.allow_client_paths,
            allow_component_options=args.allow_component_options,
            allow_remote=args.allow_remote,
            auth_token=auth_token,
            max_session_seconds=args.max_session_seconds,
        )
        try:
            asyncio.run(serve(
            args.host, args.port, default_backend=args.backend,
            default_model=args.model, default_device=_resolved_device(args.device),
            default_language=args.language, default_compute_type=args.compute_type,
                default_executable=args.executable,
                default_model_path=args.model_path,
                default_backend_options=_extra_options(args.backend_option),
                warmup_file=args.warmup_file, ffmpeg=args.ffmpeg,
                policy=policy, internal_chunk_ms=args.internal_chunk_ms,
                max_control_message_bytes=args.max_control_message_bytes,
                initialization_timeout=args.initialization_timeout,
                finalization_timeout=args.finalization_timeout,
                worker_shutdown_timeout=args.worker_shutdown_timeout,
                output_backpressure_timeout=args.output_backpressure_timeout,
                max_recovery_events=args.max_recovery_events,
                max_recovery_event_bytes=args.max_recovery_event_kib * 1024,
                max_recovery_event_bytes_per_session=(
                    args.max_recovery_events_mib * 1024 * 1024
                ),
                max_recovery_sessions=args.max_recovery_sessions,
                max_recovery_audio_bytes=args.max_recovery_audio_mib * 1024 * 1024,
                max_recovery_total_bytes=args.max_recovery_total_mib * 1024 * 1024,
                recovery_ttl_seconds=args.recovery_ttl_seconds,
                max_concurrent_sessions=args.max_concurrent_sessions,
                start_timeout=args.start_timeout,
                client_idle_timeout=args.client_idle_timeout,
                shutdown_grace_timeout=args.shutdown_grace_timeout,
                backend_replicas=args.backend_replicas,
                preload=args.preload,
                require_immutable_revision=args.require_immutable_model_revision,
                allowed_origins=(None, *args.allow_origin),
            ))
        except KeyboardInterrupt:
            return 130
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
