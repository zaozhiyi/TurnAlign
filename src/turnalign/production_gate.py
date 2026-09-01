from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODEL_REVISION_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_MAX_REPORT_BYTES = 2 * 1024 * 1024
REQUIRED_ARTIFACT_KINDS = frozenset({
    "dependency-lock",
    "host-profile",
    "model",
    "nginx-config",
    "service-unit",
    "wheel",
})


@dataclass(frozen=True, slots=True)
class EvidenceFile:
    name: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    kind: str
    name: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class ProductionGateReport:
    schema_version: int
    status: str
    source_commit: str
    release_report: EvidenceFile
    quality_report: EvidenceFile
    websocket_report: EvidenceFile
    artifacts: tuple[ArtifactEvidence, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.failures

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_json_report(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist a gate report without leaving a partial success file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_name = destination.name
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _regular_file(path: Path, *, report: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"evidence file is empty: {path}")
    if report and size > _MAX_REPORT_BYTES:
        raise ValueError(f"gate report exceeds {_MAX_REPORT_BYTES} bytes: {path}")


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _evidence(path: Path) -> EvidenceFile:
    digest, size = _digest(path)
    return EvidenceFile(path.name, digest, size)


def _load_report(path: Path) -> dict[str, object]:
    _regular_file(path, report=True)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid gate report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"gate report must contain one JSON object: {path}")
    return payload


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _nonnegative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _at_most(value: object, ceiling: object) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
    ):
        return False
    return math.isfinite(value) and value >= 0 and math.isfinite(ceiling) and value <= ceiling


def _at_least(value: object, floor: object) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or isinstance(floor, bool)
        or not isinstance(floor, (int, float))
    ):
        return False
    return math.isfinite(value) and value >= 0 and math.isfinite(floor) and value >= floor


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _integer_at_least(value: object, floor: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= floor


def _public_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        normalized = hostname.rstrip(".").lower()
        return (
            "." in normalized
            and normalized not in {"localhost"}
            and not normalized.endswith((".local", ".localhost", ".internal"))
        )
    return address.is_global


def _passed_report(name: str, report: dict[str, object], failures: list[str]) -> None:
    if report.get("status") != "passed":
        failures.append(f"{name} report did not pass")
    if report.get("failures", []) != []:
        failures.append(f"{name} report contains failures")


def _validate_release(report: dict[str, object], failures: list[str]) -> None:
    _passed_report("release", report, failures)
    if report.get("require_native_streaming") is not True:
        failures.append("release report did not require native streaming")
    if report.get("native_streaming") is not True:
        failures.append("release report did not observe native streaming")
    if report.get("require_partial") is not True:
        failures.append("release report did not require partial results")
    if report.get("require_immutable_model_revision") is not True:
        failures.append("release report did not require an immutable model revision")
    revision = report.get("model_revision")
    if not isinstance(revision, str) or _MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        failures.append("release report does not identify an immutable model revision")
    if not _positive_number(report.get("min_audio_seconds")):
        failures.append("release report has no positive minimum audio duration")
    if not _positive_integer(report.get("min_commits")):
        failures.append("release report has no positive minimum commit count")
    if not _positive_number(report.get("max_realtime_factor")):
        failures.append("release report has no positive real-time-factor ceiling")
    if not _positive_number(report.get("max_first_partial_seconds")):
        failures.append("release report has no positive first-partial ceiling")
    if not _positive_number(report.get("max_first_commit_seconds")):
        failures.append("release report has no positive first-commit ceiling")
    if not _positive_number(report.get("max_initialization_seconds")):
        failures.append("release report has no positive initialization ceiling")
    if not _at_most(report.get("realtime_factor"), report.get("max_realtime_factor")):
        failures.append("release real-time factor exceeds or lacks its ceiling")
    if not _at_most(
        report.get("first_partial_seconds"), report.get("max_first_partial_seconds")
    ):
        failures.append("release first-partial latency exceeds or lacks its ceiling")
    if not _at_most(
        report.get("first_commit_seconds"), report.get("max_first_commit_seconds")
    ):
        failures.append("release first-commit latency exceeds or lacks its ceiling")
    if not _at_most(
        report.get("initialization_seconds"), report.get("max_initialization_seconds")
    ):
        failures.append("release initialization latency exceeds or lacks its ceiling")
    if not _at_least(report.get("audio_seconds"), report.get("min_audio_seconds")):
        failures.append("release audio duration does not meet its minimum")
    minimum_commits = report.get("min_commits")
    commits = report.get("commits")
    if not isinstance(minimum_commits, int) or not _integer_at_least(commits, minimum_commits):
        failures.append("release commit count does not meet its minimum")


def _validate_quality(report: dict[str, object], failures: list[str]) -> None:
    _passed_report("quality", report, failures)
    maxima = (
        report.get("max_character_error_rate"),
        report.get("max_word_error_rate"),
        report.get("max_diarization_error_rate"),
        report.get("max_revision_updates_per_segment"),
    )
    if not any(value is not None for value in maxima):
        failures.append("quality report has no metric ceiling")
    elif any(value is not None and not _nonnegative_number(value) for value in maxima):
        failures.append("quality report contains an invalid metric ceiling")
    if not _positive_integer(report.get("min_reference_segments")):
        failures.append("quality report has no positive reference segment minimum")
    if not _positive_integer(report.get("min_reference_characters")):
        failures.append("quality report has no positive reference character minimum")
    if not _positive_number(report.get("min_reference_speech_seconds")):
        failures.append("quality report has no positive labelled-speech minimum")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        failures.append("quality report has no evaluation evidence")
        return
    for minimum, actual, label in (
        (report.get("min_reference_segments"), evaluation.get("reference_segments"), "segments"),
        (
            report.get("min_reference_characters"),
            evaluation.get("reference_characters"),
            "characters",
        ),
        (
            report.get("min_reference_speech_seconds"),
            evaluation.get("reference_speech_seconds"),
            "speech seconds",
        ),
    ):
        if not _at_least(actual, minimum):
            failures.append(f"quality reference {label} do not meet their minimum")
    for ceiling, metric, label in (
        (report.get("max_character_error_rate"), "character_error_rate", "CER"),
        (report.get("max_word_error_rate"), "word_error_rate", "WER"),
        (report.get("max_diarization_error_rate"), "diarization_error_rate", "speaker error"),
        (
            report.get("max_revision_updates_per_segment"),
            "revision_updates_per_segment",
            "revision rate",
        ),
    ):
        if ceiling is not None and not _at_most(evaluation.get(metric), ceiling):
            failures.append(f"quality {label} exceeds or lacks its ceiling")


def _validate_websocket(report: dict[str, object], failures: list[str]) -> None:
    _passed_report("websocket", report, failures)
    uri = report.get("uri")
    parsed = urlsplit(uri) if isinstance(uri, str) else None
    if parsed is None or parsed.scheme != "wss" or not _public_hostname(parsed.hostname):
        failures.append("websocket report was not run against a public wss:// endpoint")
    if report.get("recovery_probe_required") is not True:
        failures.append("websocket report did not require recovery verification")
    recovery = report.get("recovery_probe")
    if not isinstance(recovery, dict) or recovery.get("passed") is not True:
        failures.append("websocket recovery probe did not pass")
    if report.get("realtime_pacing") is not True:
        failures.append("websocket report did not use real-time pacing")
    sessions = report.get("sessions")
    if not _integer_at_least(sessions, 2):
        failures.append("websocket report did not exercise concurrent sessions")
    if report.get("failed_sessions") != 0:
        failures.append("websocket report contains failed sessions")
    if report.get("passed_sessions") != report.get("sessions"):
        failures.append("websocket report did not pass every session")
    if not _positive_number(report.get("max_ready_seconds")):
        failures.append("websocket report has no ready-time ceiling")
    if not _positive_number(report.get("max_total_seconds")):
        failures.append("websocket report has no total-time ceiling")
    if not _positive_integer(report.get("min_audio_acks_per_session")):
        failures.append("websocket report has no positive acknowledgement minimum")
    if report.get("max_dropped_partials_per_session") != 0:
        failures.append("websocket report did not forbid dropped partials")
    pauses = report.get("max_backpressure_pauses_per_session")
    if isinstance(pauses, bool) or not isinstance(pauses, int) or pauses < 0:
        failures.append("websocket report has no explicit backpressure-pause ceiling")
    if not _positive_number(report.get("audio_seconds_per_session")):
        failures.append("websocket report has no positive per-session audio duration")
    if not _at_most(report.get("ready_seconds_p95"), report.get("max_ready_seconds")):
        failures.append("websocket ready-time p95 exceeds or lacks its ceiling")
    if not _at_most(report.get("total_seconds_p95"), report.get("max_total_seconds")):
        failures.append("websocket total-time p95 exceeds or lacks its ceiling")
    results = report.get("results")
    if (
        not isinstance(results, list)
        or not isinstance(sessions, int)
        or len(results) != sessions
        or any(not isinstance(result, dict) or result.get("passed") is not True for result in results)
    ):
        failures.append("websocket report lacks passing per-session evidence")


def run_production_gate(
    release_path: Path,
    quality_path: Path,
    websocket_path: Path,
    *,
    source_commit: str,
    artifacts: list[tuple[str, Path]],
) -> ProductionGateReport:
    if _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a lowercase 40-character Git commit")

    release = _load_report(release_path)
    quality = _load_report(quality_path)
    websocket = _load_report(websocket_path)
    failures: list[str] = []
    _validate_release(release, failures)
    _validate_quality(quality, failures)
    _validate_websocket(websocket, failures)

    artifact_evidence = []
    kinds = set()
    for kind, path in artifacts:
        if kind not in REQUIRED_ARTIFACT_KINDS:
            raise ValueError(f"unsupported artifact kind: {kind}")
        _regular_file(path)
        digest, size = _digest(path)
        artifact_evidence.append(ArtifactEvidence(kind, path.name, digest, size))
        kinds.add(kind)
    for missing in sorted(REQUIRED_ARTIFACT_KINDS - kinds):
        failures.append(f"missing required artifact kind: {missing}")

    return ProductionGateReport(
        schema_version=1,
        status="failed" if failures else "passed",
        source_commit=source_commit,
        release_report=_evidence(release_path),
        quality_report=_evidence(quality_path),
        websocket_report=_evidence(websocket_path),
        artifacts=tuple(sorted(
            artifact_evidence,
            key=lambda item: (item.kind, item.name, item.sha256),
        )),
        failures=tuple(failures),
    )
