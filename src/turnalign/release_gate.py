from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter

from .models import AudioChunk, TranscriptEvent
from .plugins import AsrBackend
from .resources import (
    is_immutable_model_revision,
    model_revision,
    observed_model_files,
)
from .session import transcribe_events
from .validation import EventStreamValidator


@dataclass(frozen=True, slots=True)
class ReleaseGateReport:
    status: str
    source_commit: str | None
    input_audio_sha256: str | None
    created_at: str
    validity_seconds: float
    backend: str
    model: str | None
    loaded_models: tuple[dict[str, object], ...]
    native_streaming: bool
    model_revision: str | None
    max_realtime_factor: float
    max_first_partial_seconds: float
    max_first_commit_seconds: float | None
    max_initialization_seconds: float
    min_audio_seconds: float
    min_commits: int
    require_partial: bool
    require_native_streaming: bool
    require_immutable_model_revision: bool
    require_local_model: bool
    initialization_seconds: float
    events: int
    partials: int
    commits: int
    replacements: int
    audio_seconds: float
    processing_seconds: float
    realtime_factor: float | None
    first_partial_seconds: float | None
    first_commit_seconds: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_release_gate(
    chunks: Iterable[AudioChunk],
    backend: AsrBackend,
    *,
    model: str | None = None,
    require_local_model: bool = False,
    validity_seconds: float = 86400.0,
    max_realtime_factor: float = 1.0,
    max_first_partial_seconds: float = 3.0,
    max_first_commit_seconds: float | None = None,
    max_initialization_seconds: float = 120.0,
    initialization_seconds: float = 0.0,
    min_audio_seconds: float = 10.0,
    min_commits: int = 1,
    require_partial: bool = True,
    require_native_streaming: bool = True,
    require_immutable_model_revision: bool = False,
    source_commit: str | None = None,
    input_audio_sha256: str | None = None,
    event_sink: Callable[[TranscriptEvent], None] | None = None,
) -> ReleaseGateReport:
    """Run a real backend and turn release expectations into a pass/fail result."""
    for name, value in (
        ("max_realtime_factor", max_realtime_factor),
        ("max_first_partial_seconds", max_first_partial_seconds),
        ("max_initialization_seconds", max_initialization_seconds),
        ("min_audio_seconds", min_audio_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if max_first_commit_seconds is not None and (
        isinstance(max_first_commit_seconds, bool)
        or not isinstance(max_first_commit_seconds, (int, float))
        or not math.isfinite(max_first_commit_seconds)
        or max_first_commit_seconds <= 0
    ):
        raise ValueError("max_first_commit_seconds must be finite and positive")
    if (
        isinstance(initialization_seconds, bool)
        or not isinstance(initialization_seconds, (int, float))
        or not math.isfinite(initialization_seconds)
        or initialization_seconds < 0
    ):
        raise ValueError("initialization_seconds must be finite and non-negative")
    if isinstance(min_commits, bool) or not isinstance(min_commits, int) or min_commits <= 0:
        raise ValueError("min_commits must be positive")
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, (int, float))
        or not math.isfinite(validity_seconds)
        or validity_seconds <= 0
    ):
        raise ValueError("validity_seconds must be finite and positive")

    validator = EventStreamValidator()
    started = perf_counter()
    first_partial_seconds: float | None = None
    first_commit_seconds: float | None = None
    counts = {"events": 0, "partials": 0, "commits": 0, "replacements": 0}
    protocol_failures: list[str] = []
    runtime_failure: str | None = None
    audio_seconds = 0.0
    events = transcribe_events(chunks, backend, live=True)
    try:
        for event in events:
            try:
                validator.accept(event)
            except ValueError as error:
                protocol_failures.append(f"invalid event stream: {error}")
            counts["events"] += 1
            if event.kind == "partial":
                counts["partials"] += 1
                if first_partial_seconds is None:
                    first_partial_seconds = perf_counter() - started
            elif event.kind == "commit":
                counts["commits"] += 1
                if first_commit_seconds is None:
                    first_commit_seconds = perf_counter() - started
            elif event.kind == "replace":
                counts["replacements"] += 1
            elif event.kind == "end":
                audio_seconds = float(event.metadata.get("audio_seconds") or event.end)
            if event_sink is not None:
                event_sink(event)
    except Exception as error:  # noqa: BLE001 - release gate is a reporting boundary
        runtime_failure = (
            f"backend execution failed: {type(error).__name__}: {error}"
        )
    finally:
        events.close()

    processing_seconds = perf_counter() - started
    realtime_factor = (
        processing_seconds / audio_seconds if audio_seconds > 0 else None
    )
    native_streaming = bool(backend.capabilities.streaming)
    revision = model_revision(backend)
    loaded_models = observed_model_files(backend)
    failures: list[str] = []
    failures.extend(protocol_failures)
    if runtime_failure is not None:
        failures.append(runtime_failure)
    if not validator.ended:
        failures.append("event stream did not end")
    if counts["commits"] < min_commits:
        failures.append(
            f"commit count {counts['commits']} is below required {min_commits}"
        )
    if audio_seconds < min_audio_seconds:
        failures.append(
            f"audio duration {audio_seconds:.3f}s is below required "
            f"{min_audio_seconds:.3f}s"
        )
    if require_native_streaming and not native_streaming:
        failures.append("backend does not declare native streaming")
    if require_immutable_model_revision and (
        not is_immutable_model_revision(backend)
    ):
        failures.append(
            "model revision is not pinned to an immutable 40- or 64-character "
            "commit hash"
        )
    if require_local_model and not loaded_models:
        failures.append("backend did not load retained local model files")
    if initialization_seconds > max_initialization_seconds:
        failures.append(
            "initialization latency "
            f"{initialization_seconds:.3f}s exceeds {max_initialization_seconds:.3f}s"
        )
    if require_partial and first_partial_seconds is None:
        failures.append("no partial event was emitted")
    elif (
        first_partial_seconds is not None
        and first_partial_seconds > max_first_partial_seconds
    ):
        failures.append(
            "first partial latency "
            f"{first_partial_seconds:.3f}s exceeds {max_first_partial_seconds:.3f}s"
        )
    if max_first_commit_seconds is not None:
        if first_commit_seconds is None:
            failures.append("no commit event was emitted for latency measurement")
        elif first_commit_seconds > max_first_commit_seconds:
            failures.append(
                "first commit latency "
                f"{first_commit_seconds:.3f}s exceeds "
                f"{max_first_commit_seconds:.3f}s"
            )
    if realtime_factor is None:
        failures.append("audio duration is zero or unavailable")
    elif realtime_factor > max_realtime_factor:
        failures.append(
            f"realtime factor {realtime_factor:.4f} exceeds {max_realtime_factor:.4f}"
        )

    return ReleaseGateReport(
        status="failed" if failures else "passed",
        source_commit=source_commit,
        input_audio_sha256=input_audio_sha256,
        created_at=(
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        ),
        validity_seconds=round(validity_seconds, 3),
        backend=backend.name,
        model=model,
        loaded_models=loaded_models,
        native_streaming=native_streaming,
        model_revision=revision,
        max_realtime_factor=max_realtime_factor,
        max_first_partial_seconds=max_first_partial_seconds,
        max_first_commit_seconds=max_first_commit_seconds,
        max_initialization_seconds=max_initialization_seconds,
        min_audio_seconds=min_audio_seconds,
        min_commits=min_commits,
        require_partial=require_partial,
        require_native_streaming=require_native_streaming,
        require_immutable_model_revision=require_immutable_model_revision,
        require_local_model=require_local_model,
        initialization_seconds=round(initialization_seconds, 3),
        events=counts["events"],
        partials=counts["partials"],
        commits=counts["commits"],
        replacements=counts["replacements"],
        audio_seconds=round(audio_seconds, 3),
        processing_seconds=round(processing_seconds, 3),
        realtime_factor=round(realtime_factor, 4) if realtime_factor is not None else None,
        first_partial_seconds=(
            round(first_partial_seconds, 3)
            if first_partial_seconds is not None
            else None
        ),
        first_commit_seconds=(
            round(first_commit_seconds, 3)
            if first_commit_seconds is not None
            else None
        ),
        failures=tuple(failures),
    )
