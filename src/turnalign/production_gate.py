from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import importlib.resources
import io
import ipaddress
import json
import math
import os
import platform
import re
import stat
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

from . import __version__
from .deployment_validation import validate_nginx_config, validate_systemd_service
from .jsonutil import strict_json_loads

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODEL_REVISION_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_REPORT_BYTES = 2 * 1024 * 1024
_MAX_SBOM_BYTES = 16 * 1024 * 1024
_MAX_LOCK_BYTES = 4 * 1024 * 1024
_MAX_MODEL_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_HOST_PROFILE_BYTES = 2 * 1024 * 1024
_MAX_DEPLOYMENT_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_WHEEL_BYTES = 64 * 1024 * 1024
_MAX_WHEEL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_WHEEL_ENTRIES = 4_096
_LOCK_REQUIREMENT_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)(?:\s|;|$)"
)
_EMBEDDED_REQUIREMENT_PATTERN = re.compile(
    r"\s[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9]"
)
_SHA256_OPTION_PATTERN = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})(?:\s|$)")
_BUILD_ONLY_SBOM_COMPONENTS = frozenset({
    "bandit",
    "build",
    "cyclonedx-bom",
    "mypy",
    "pip",
    "pip-audit",
    "ruff",
    "twine",
})
REQUIRED_ARTIFACT_KINDS = frozenset({
    "dependency-lock",
    "host-profile",
    "model",
    "model-manifest",
    "nginx-config",
    "quality-hypothesis",
    "quality-reference",
    "release-audio",
    "service-unit",
    "sbom",
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
class _EvidenceSnapshot:
    sha256: str
    size: int
    content: bytes | None


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
            json.dump(
                payload,
                destination,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
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


def _metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_evidence(path: Path) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        path_metadata = os.lstat(path)
        if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(
            path_metadata.st_mode
        ):
            raise ValueError(f"evidence must be a regular non-symlink file: {path}")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(
            f"evidence must be a regular non-symlink file: {path}"
        ) from error
    try:
        opened_metadata = os.fstat(descriptor)
        try:
            current_metadata = os.lstat(path)
        except OSError as error:
            raise ValueError(
                f"evidence changed while it was being opened: {path}"
            ) from error
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or (opened_metadata.st_dev, opened_metadata.st_ino)
            != (current_metadata.st_dev, current_metadata.st_ino)
        ):
            raise ValueError(f"evidence changed while it was being opened: {path}")
        return descriptor, opened_metadata
    except BaseException:
        os.close(descriptor)
        raise


def _snapshot_evidence(
    path: Path,
    *,
    capture_limit: int | None = None,
    hard_limit: int | None = None,
) -> _EvidenceSnapshot:
    descriptor, initial_metadata = _open_evidence(path)
    if initial_metadata.st_size <= 0:
        os.close(descriptor)
        raise ValueError(f"evidence file is empty: {path}")
    if hard_limit is not None and initial_metadata.st_size > hard_limit:
        os.close(descriptor)
        raise ValueError(f"gate report exceeds {hard_limit} bytes: {path}")

    digest = hashlib.sha256()
    size = 0
    captured = bytearray() if capture_limit is not None else None
    try:
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            for block in iter(lambda: source.read(1024 * 1024), b""):
                size += len(block)
                if hard_limit is not None and size > hard_limit:
                    raise ValueError(f"gate report exceeds {hard_limit} bytes: {path}")
                digest.update(block)
                if captured is not None:
                    if capture_limit is not None and size <= capture_limit:
                        captured.extend(block)
                    else:
                        captured = None
            final_metadata = os.fstat(source.fileno())
        try:
            current_metadata = os.lstat(path)
        except OSError as error:
            raise ValueError(
                f"evidence changed while it was being read: {path}"
            ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        size != initial_metadata.st_size
        or _metadata_signature(final_metadata) != _metadata_signature(initial_metadata)
        or (final_metadata.st_dev, final_metadata.st_ino)
        != (current_metadata.st_dev, current_metadata.st_ino)
    ):
        raise ValueError(f"evidence changed while it was being read: {path}")
    return _EvidenceSnapshot(
        sha256=digest.hexdigest(),
        size=size,
        content=bytes(captured) if captured is not None else None,
    )


def _load_report(path: Path) -> tuple[dict[str, object], EvidenceFile]:
    snapshot = _snapshot_evidence(
        path,
        capture_limit=_MAX_REPORT_BYTES,
        hard_limit=_MAX_REPORT_BYTES,
    )
    if snapshot.content is None:  # hard_limit makes this unreachable.
        raise ValueError(f"gate report exceeds {_MAX_REPORT_BYTES} bytes: {path}")
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid gate report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"gate report must contain one JSON object: {path}")
    return payload, EvidenceFile(path.name, snapshot.sha256, snapshot.size)


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


def _nonnegative_integer(value: object) -> bool:
    return _integer_at_least(value, 0)


def _valid_model_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.strip() == value
        and len(value) <= 512
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def create_model_manifest(
    model_id: str,
    model_revision: str,
    files: list[Path],
) -> dict[str, object]:
    """Hash immutable model files into the schema consumed by production-gate."""

    if not _valid_model_id(model_id):
        raise ValueError("model_id must be a non-empty bounded identifier")
    if _MODEL_REVISION_PATTERN.fullmatch(model_revision) is None:
        raise ValueError("model_revision must be an immutable 40- or 64-character hash")
    if not files:
        raise ValueError("at least one model file is required")
    entries = []
    names = set()
    for path in files:
        if path.name in names:
            raise ValueError(f"model files must have unique base names: {path.name}")
        names.add(path.name)
        snapshot = _snapshot_evidence(path)
        entries.append({
            "name": path.name,
            "sha256": snapshot.sha256,
            "bytes": snapshot.size,
        })
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "files": sorted(entries, key=lambda item: cast(str, item["name"])),
    }


def create_host_profile(
    source_commit: str | None,
    artifacts: list[tuple[str, Path]],
) -> dict[str, object]:
    """Capture host identity and bind every other retained production artifact."""

    system = platform.system()
    if system != "Linux":
        raise RuntimeError("host-profile must run on the Linux production host")
    runtime = _installed_runtime_identity(source_commit)
    bound_commit = runtime["turnalign_source_commit"]
    expected_kinds = REQUIRED_ARTIFACT_KINDS - {"host-profile"}
    evidence = []
    kinds = set()
    identities = set()
    for kind, path in artifacts:
        if kind not in expected_kinds:
            raise ValueError(f"unsupported host-profile artifact kind: {kind}")
        snapshot = _snapshot_evidence(path)
        identity = (kind, path.name)
        if identity in identities:
            raise ValueError(
                f"host-profile artifacts require unique kind/name pairs: {kind}={path.name}"
            )
        identities.add(identity)
        kinds.add(kind)
        evidence.append({
            "kind": kind,
            "name": path.name,
            "sha256": snapshot.sha256,
            "bytes": snapshot.size,
        })
    missing = expected_kinds - kinds
    if missing:
        raise ValueError(
            "host-profile is missing required artifact kinds: "
            + ", ".join(sorted(missing))
        )
    for kind in expected_kinds - {"model"}:
        if sum(item["kind"] == kind for item in evidence) != 1:
            raise ValueError(f"host-profile artifact kind must appear once: {kind}")
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count <= 0:
        raise RuntimeError("cannot determine the host logical CPU count")
    return {
        "schema_version": 2,
        "source_commit": bound_commit,
        "runtime": runtime,
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "logical_cpu_count": logical_cpu_count,
        },
        "artifacts": sorted(
            evidence,
            key=lambda item: (cast(str, item["kind"]), cast(str, item["name"])),
        ),
    }


def _installed_runtime_identity(source_commit: str | None = None) -> dict[str, str]:
    try:
        embedded_identity = importlib.resources.files("turnalign").joinpath(
            "_source_commit.txt"
        ).read_text(encoding="ascii")
    except (FileNotFoundError, UnicodeError, OSError) as error:
        raise ValueError(
            "host-profile requires a Wheel with a readable source identity"
        ) from error
    if (
        not embedded_identity.endswith("\n")
        or _COMMIT_PATTERN.fullmatch(embedded_identity[:-1]) is None
    ):
        raise ValueError(
            "host-profile requires a Wheel with a valid bound source identity"
        )
    embedded_commit = embedded_identity[:-1]
    if source_commit is not None and source_commit != embedded_commit:
        raise ValueError(
            "host-profile runtime source identity does not match the candidate"
        )
    release_prefix = f"/opt/turnalign/releases/{embedded_commit}/venv"
    python_executable = os.path.abspath(sys.executable)
    python_prefix = os.path.abspath(sys.prefix)
    if (
        python_prefix != release_prefix
        or python_executable != f"{release_prefix}/bin/python"
    ):
        raise ValueError(
            "host-profile must run from the candidate's versioned production "
            "environment"
        )
    return {
        "python_executable": python_executable,
        "python_prefix": python_prefix,
        "turnalign_source_commit": embedded_commit,
        "turnalign_version": __version__,
    }


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


def _package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _pypi_purl_identity(value: str) -> tuple[str, str] | None:
    if not value.startswith("pkg:pypi/"):
        return None
    identity = value.removeprefix("pkg:pypi/").split("?", 1)[0].split("#", 1)[0]
    name, separator, version = identity.rpartition("@")
    if not separator or not name or not version:
        return None
    return _package_name(unquote(name)), unquote(version)


def _passed_report(name: str, report: dict[str, object], failures: list[str]) -> None:
    if report.get("status") != "passed":
        failures.append(f"{name} report did not pass")
    if report.get("failures", []) != []:
        failures.append(f"{name} report contains failures")


def _validate_report_source(
    name: str,
    report: dict[str, object],
    source_commit: str,
    failures: list[str],
) -> None:
    if report.get("source_commit") != source_commit:
        failures.append(f"{name} report is not bound to source commit {source_commit}")


def _validate_release(report: dict[str, object], failures: list[str]) -> None:
    _passed_report("release", report, failures)
    backend = report.get("backend")
    if not isinstance(backend, str) or not backend or backend.strip() != backend:
        failures.append("release report does not identify a valid backend")
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

    counts = {
        name: report.get(name)
        for name in ("events", "partials", "commits", "replacements")
    }
    if not all(_nonnegative_integer(value) for value in counts.values()):
        failures.append("release report lacks complete typed event counts")
    else:
        event_count = cast(int, counts["events"])
        transcript_count = sum(
            cast(int, counts[name])
            for name in ("partials", "commits", "replacements")
        )
        if cast(int, counts["partials"]) < 1:
            failures.append("release event counts do not contain a partial result")
        if event_count < transcript_count + 1:
            failures.append("release event count is inconsistent with its typed events")

    processing_seconds = report.get("processing_seconds")
    if not _nonnegative_number(processing_seconds):
        failures.append("release report lacks a valid processing duration")
    elif _positive_number(report.get("audio_seconds")) and _nonnegative_number(
        report.get("realtime_factor")
    ):
        expected_realtime_factor = cast(float, processing_seconds) / cast(
            float, report["audio_seconds"]
        )
        if not math.isclose(
            cast(float, report["realtime_factor"]),
            expected_realtime_factor,
            rel_tol=0.001,
            abs_tol=0.0006,
        ):
            failures.append(
                "release real-time factor is inconsistent with its timing evidence"
            )

    if _nonnegative_number(processing_seconds):
        for field, label in (
            ("first_partial_seconds", "first-partial"),
            ("first_commit_seconds", "first-commit"),
        ):
            latency = report.get(field)
            if _nonnegative_number(latency) and cast(float, latency) > (
                cast(float, processing_seconds) + 0.001
            ):
                failures.append(
                    f"release {label} latency exceeds its processing duration"
                )


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
    revision = report.get("model_revision")
    if not isinstance(revision, str) or _MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        failures.append("quality report does not identify an immutable model revision")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        failures.append("quality report has no evaluation evidence")
        return

    normalization = evaluation.get("text_normalization")
    if not (
        isinstance(normalization, dict)
        and set(normalization) == {
            "unicode_form",
            "case_sensitive",
            "punctuation_sensitive",
        }
        and normalization.get("unicode_form") in {"none", "NFC", "NFKC"}
        and isinstance(normalization.get("case_sensitive"), bool)
        and isinstance(normalization.get("punctuation_sensitive"), bool)
    ):
        failures.append("quality report lacks a complete text-normalization policy")

    count_fields = (
        "reference_segments",
        "hypothesis_segments",
        "reference_characters",
        "reference_words",
        "reference_speakers",
    )
    if not all(_nonnegative_integer(evaluation.get(name)) for name in count_fields):
        failures.append("quality report lacks complete typed evaluation counts")
    else:
        reference_segments = cast(int, evaluation["reference_segments"])
        reference_characters = cast(int, evaluation["reference_characters"])
        if cast(int, evaluation["reference_speakers"]) > reference_segments:
            failures.append("quality speaker count exceeds its reference segment count")
        if cast(int, evaluation["reference_words"]) > reference_characters:
            failures.append("quality word count exceeds its reference character count")

    minimum_segments = report.get("min_reference_segments")
    if not isinstance(minimum_segments, int) or not _integer_at_least(
        evaluation.get("reference_segments"), minimum_segments
    ):
        failures.append("quality reference segments do not meet their minimum")
    minimum_characters = report.get("min_reference_characters")
    if not isinstance(minimum_characters, int) or not _integer_at_least(
        evaluation.get("reference_characters"), minimum_characters
    ):
        failures.append("quality reference characters do not meet their minimum")
    if not _at_least(
        evaluation.get("reference_speech_seconds"),
        report.get("min_reference_speech_seconds"),
    ):
        failures.append("quality reference speech seconds do not meet their minimum")

    for metric, label in (
        ("character_error_rate", "CER"),
        ("word_error_rate", "WER"),
        ("revision_updates_per_segment", "revision rate"),
        ("reference_speech_seconds", "reference speech duration"),
    ):
        if not _nonnegative_number(evaluation.get(metric)):
            failures.append(f"quality {label} is missing or invalid")
    diarization_error_rate = evaluation.get("diarization_error_rate")
    if diarization_error_rate is not None and not _nonnegative_number(
        diarization_error_rate
    ):
        failures.append("quality speaker error is invalid")
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
    if report.get("identity_consistent") is not True:
        failures.append("websocket report did not observe one consistent deployment identity")
    backend = report.get("backend")
    backend_implementation = report.get("backend_implementation")
    model = report.get("model")
    revision = report.get("model_revision")
    device = report.get("device")
    if not _valid_model_id(backend):
        failures.append("websocket report has no valid observed backend identity")
    if not _valid_model_id(backend_implementation):
        failures.append(
            "websocket report has no valid observed backend implementation identity"
        )
    if not _valid_model_id(model):
        failures.append("websocket report has no valid observed model identity")
    if not isinstance(revision, str) or _MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        failures.append("websocket report has no immutable observed model revision")
    if not _valid_model_id(device):
        failures.append("websocket report has no valid observed device identity")
    for field in ("language", "compute_type"):
        value = report.get(field)
        if value is not None and not _valid_model_id(value):
            failures.append(f"websocket report has an invalid observed {field} identity")
    uri = report.get("uri")
    parsed = urlsplit(uri) if isinstance(uri, str) else None
    if parsed is None or parsed.scheme != "wss" or not _public_hostname(parsed.hostname):
        failures.append("websocket report was not run against a public wss:// endpoint")
    if parsed is not None:
        try:
            _ = parsed.port
        except ValueError:
            failures.append("websocket report URI contains an invalid port")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            failures.append(
                "websocket report URI must not contain credentials, query strings "
                "or fragments"
            )
    if report.get("recovery_probe_required") is not True:
        failures.append("websocket report did not require recovery verification")
    recovery = report.get("recovery_probe")
    if not isinstance(recovery, dict) or recovery.get("passed") is not True:
        failures.append("websocket recovery probe did not pass")
    elif (
        recovery.get("backend") != backend
        or recovery.get("backend_implementation") != backend_implementation
        or recovery.get("model") != model
        or recovery.get("model_revision") != revision
        or recovery.get("device") != device
        or recovery.get("language") != report.get("language")
        or recovery.get("compute_type") != report.get("compute_type")
        or not _positive_number(recovery.get("disconnected_audio_seconds"))
        or not _integer_at_least(
            recovery.get("first_last_acknowledged_sequence"), 0
        )
        or not _integer_at_least(recovery.get("resumed_next_audio_sequence"), 0)
        or not _integer_at_least(recovery.get("final_acknowledged_sequence"), 0)
        or not _integer_at_least(recovery.get("final_buffered_bytes"), 0)
        or not _positive_integer(recovery.get("events"))
        or not _integer_at_least(recovery.get("commits"), 0)
        or not _positive_integer(recovery.get("audio_acks"))
        or recovery.get("failure") is not None
    ):
        failures.append("websocket recovery probe lacks complete typed evidence")
    else:
        first_acknowledged = cast(
            int, recovery["first_last_acknowledged_sequence"]
        )
        resumed_sequence = cast(int, recovery["resumed_next_audio_sequence"])
        final_acknowledged = cast(int, recovery["final_acknowledged_sequence"])
        if (
            resumed_sequence != first_acknowledged + 1
            or final_acknowledged <= first_acknowledged
            or recovery["final_buffered_bytes"] != 0
        ):
            failures.append(
                "websocket recovery probe has inconsistent sequence or buffer evidence"
            )
    if report.get("realtime_pacing") is not True:
        failures.append("websocket report did not use real-time pacing")
    sessions = report.get("sessions")
    if not _integer_at_least(sessions, 2):
        failures.append("websocket report did not exercise concurrent sessions")
    if not _integer_at_least(report.get("failed_sessions"), 0) or report.get(
        "failed_sessions"
    ) != 0:
        failures.append("websocket report contains failed sessions")
    if not _integer_at_least(report.get("passed_sessions"), 0) or report.get(
        "passed_sessions"
    ) != sessions:
        failures.append("websocket report did not pass every session")
    max_ready_seconds = report.get("max_ready_seconds")
    max_total_seconds = report.get("max_total_seconds")
    min_commits = report.get("min_commits_per_session")
    min_audio_acks = report.get("min_audio_acks_per_session")
    if not _positive_number(max_ready_seconds):
        failures.append("websocket report has no ready-time ceiling")
    if not _positive_number(max_total_seconds):
        failures.append("websocket report has no total-time ceiling")
    if not _integer_at_least(min_commits, 0):
        failures.append("websocket report has no valid commit minimum")
    if not _positive_integer(min_audio_acks):
        failures.append("websocket report has no positive acknowledgement minimum")
    if (
        not _integer_at_least(report.get("max_dropped_partials_per_session"), 0)
        or report.get("max_dropped_partials_per_session") != 0
    ):
        failures.append("websocket report did not forbid dropped partials")
    pauses = report.get("max_backpressure_pauses_per_session")
    if isinstance(pauses, bool) or not isinstance(pauses, int) or pauses < 0:
        failures.append("websocket report has no explicit backpressure-pause ceiling")
    if not _positive_number(report.get("audio_seconds_per_session")):
        failures.append("websocket report has no positive per-session audio duration")
    if not _at_most(report.get("ready_seconds_p95"), max_ready_seconds):
        failures.append("websocket ready-time p95 exceeds or lacks its ceiling")
    if not _at_most(report.get("total_seconds_p95"), max_total_seconds):
        failures.append("websocket total-time p95 exceeds or lacks its ceiling")
    results = report.get("results")
    if not isinstance(results, list) or not _integer_at_least(sessions, 2) or len(
        results
    ) != sessions:
        failures.append("websocket report lacks passing per-session evidence")
        return
    typed_sessions = cast(int, sessions)

    valid_results = all(
        isinstance(result, dict)
        and result.get("passed") is True
        and result.get("backend") == backend
        and result.get("backend_implementation") == backend_implementation
        and result.get("model") == model
        and result.get("model_revision") == revision
        and result.get("device") == device
        and result.get("language") == report.get("language")
        and result.get("compute_type") == report.get("compute_type")
        and _integer_at_least(result.get("session"), 1)
        and _positive_number(result.get("ready_seconds"))
        and _positive_number(result.get("total_seconds"))
        and _positive_integer(result.get("events"))
        and _integer_at_least(result.get("partials"), 0)
        and _integer_at_least(result.get("commits"), 0)
        and _positive_integer(result.get("audio_acks"))
        and _integer_at_least(result.get("last_acknowledged_sequence"), 0)
        and _integer_at_least(result.get("final_buffered_bytes"), 0)
        and _integer_at_least(result.get("backpressure_pauses"), 0)
        and _integer_at_least(result.get("dropped_partials"), 0)
        and result.get("failure") is None
        for result in results
    )
    if not valid_results:
        failures.append("websocket report lacks complete typed per-session evidence")
        return

    typed_results = [result for result in results if isinstance(result, dict)]
    if {result["session"] for result in typed_results} != set(
        range(1, typed_sessions + 1)
    ):
        failures.append("websocket report has duplicate or missing session identities")
    thresholds_are_typed = (
        _positive_number(max_ready_seconds)
        and _positive_number(max_total_seconds)
        and _integer_at_least(min_commits, 0)
        and _positive_integer(min_audio_acks)
        and _integer_at_least(pauses, 0)
    )
    if thresholds_are_typed:
        for result in typed_results:
            if (
                not _at_most(result["ready_seconds"], max_ready_seconds)
                or not _at_most(result["total_seconds"], max_total_seconds)
                or not _at_most(result["ready_seconds"], result["total_seconds"])
                or not _integer_at_least(result["commits"], cast(int, min_commits))
                or not _integer_at_least(
                    result["audio_acks"], cast(int, min_audio_acks)
                )
                or result["final_buffered_bytes"] != 0
                or not _at_most(result["backpressure_pauses"], pauses)
                or result["dropped_partials"] != 0
            ):
                failures.append("websocket per-session evidence violates its thresholds")
                break

    percentile_index = max(0, math.ceil(len(typed_results) * 0.95) - 1)
    expected_ready_p95 = sorted(result["ready_seconds"] for result in typed_results)[
        percentile_index
    ]
    expected_total_p95 = sorted(result["total_seconds"] for result in typed_results)[
        percentile_index
    ]
    if (
        report.get("ready_seconds_p95") != expected_ready_p95
        or report.get("total_seconds_p95") != expected_total_p95
    ):
        failures.append("websocket latency p95 does not match per-session evidence")

    for aggregate, field in (
        ("events", "events"),
        ("commits", "commits"),
        ("audio_acks", "audio_acks"),
        ("backpressure_pauses", "backpressure_pauses"),
        ("dropped_partials", "dropped_partials"),
    ):
        value = report.get(aggregate)
        if not _integer_at_least(value, 0) or value != sum(
            result[field] for result in typed_results
        ):
            failures.append(f"websocket aggregate {aggregate} does not match sessions")


def _validate_sbom(
    snapshot: _EvidenceSnapshot,
    failures: list[str],
) -> dict[str, str]:
    if snapshot.content is None:
        failures.append(f"SBOM exceeds {_MAX_SBOM_BYTES} bytes")
        return {}
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"SBOM is not strict JSON: {error}")
        return {}
    if not isinstance(payload, dict):
        failures.append("SBOM must contain one JSON object")
        return {}
    if payload.get("bomFormat") != "CycloneDX":
        failures.append("SBOM is not in CycloneDX format")
    if payload.get("specVersion") not in {"1.4", "1.5", "1.6", "1.7"}:
        failures.append("SBOM uses an unsupported CycloneDX specification version")
    metadata = payload.get("metadata")
    root = metadata.get("component") if isinstance(metadata, dict) else None
    root_ref = root.get("bom-ref") if isinstance(root, dict) else None
    if (
        not isinstance(root, dict)
        or root.get("name") != "turnalign"
        or not isinstance(root.get("version"), str)
        or not root.get("version")
        or not isinstance(root_ref, str)
        or not root_ref
    ):
        failures.append("SBOM does not identify the versioned TurnAlign root component")
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        failures.append("SBOM has no runtime components")
        return {}
    component_versions: dict[str, str] = {}
    websocket_refs = set()
    component_refs = set()
    for component in components:
        if not isinstance(component, dict):
            failures.append("SBOM contains a non-object component")
            continue
        name = component.get("name")
        version = component.get("version")
        purl = component.get("purl")
        component_ref = component.get("bom-ref")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or not isinstance(purl, str)
            or _pypi_purl_identity(purl) != (_package_name(name), version)
            or not isinstance(component_ref, str)
            or not component_ref
        ):
            failures.append("SBOM contains an unversioned or unidentifiable component")
            continue
        if component_ref in component_refs:
            failures.append(f"SBOM contains a duplicate component reference: {component_ref}")
        component_refs.add(component_ref)
        normalized_name = _package_name(name)
        previous_version = component_versions.get(normalized_name)
        if previous_version is not None and previous_version != version:
            failures.append(f"SBOM contains conflicting versions for {normalized_name}")
        component_versions[normalized_name] = version
        if normalized_name == "websockets":
            websocket_refs.add(component_ref)
    if "websockets" not in component_versions:
        failures.append("SBOM does not include the WebSocket server runtime dependency")
    build_only = sorted(_BUILD_ONLY_SBOM_COMPONENTS.intersection(component_versions))
    if build_only:
        failures.append(
            "SBOM includes build-only tooling in the production environment: "
            + ", ".join(build_only)
        )
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        failures.append("SBOM has no dependency graph")
    elif (
        not isinstance(root_ref, str)
        or not websocket_refs
        or not any(
            isinstance(item, dict)
            and item.get("ref") == root_ref
            and isinstance(item.get("dependsOn"), list)
            and bool(websocket_refs.intersection(item["dependsOn"]))
            for item in dependencies
        )
    ):
        failures.append("SBOM dependency graph does not link TurnAlign to websockets")
    return component_versions


def _validate_dependency_lock(
    snapshot: _EvidenceSnapshot,
    failures: list[str],
) -> dict[str, tuple[str, bool]]:
    if snapshot.content is None:
        failures.append(f"dependency lock exceeds {_MAX_LOCK_BYTES} bytes")
        return {}
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as error:
        failures.append(f"dependency lock is not UTF-8: {error}")
        return {}

    logical_lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if pending and line.startswith("#"):
                failures.append("dependency lock continuation is interrupted by a comment")
            continue
        if pending:
            if not line.startswith("--hash=sha256:"):
                failures.append("dependency lock continuation contains a second requirement")
            pending += " " + line
        else:
            pending = line
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        failures.append("dependency lock ends with an incomplete continuation")
        logical_lines.append(pending)

    requirements: dict[str, tuple[str, bool]] = {}
    allowed_directives = (
        "--extra-index-url ",
        "--find-links ",
        "--index-url ",
        "--no-binary ",
        "--only-binary ",
    )
    for line in logical_lines:
        if line.startswith("--trusted-host"):
            failures.append("dependency lock disables TLS verification with --trusted-host")
            continue
        if line.startswith(allowed_directives):
            continue
        if line.startswith(("-e ", "--editable ")) or any(
            token in line for token in (" @ ", "file:", "git+", "../")
        ):
            failures.append("dependency lock contains a mutable or local requirement")
            continue
        match = _LOCK_REQUIREMENT_PATTERN.match(line)
        if match is None:
            failures.append("dependency lock contains a requirement without an exact version")
            continue
        if _EMBEDDED_REQUIREMENT_PATTERN.search(line):
            failures.append("dependency lock combines multiple requirements in one entry")
            continue
        hashes = _SHA256_OPTION_PATTERN.findall(line)
        if not hashes:
            failures.append(f"dependency lock entry lacks a SHA-256 hash: {match.group(1)}")
        name = _package_name(match.group(1))
        version = match.group(2)
        conditional = ";" in line.split("--hash=", 1)[0]
        previous = requirements.get(name)
        if previous is not None and previous[0] != version:
            failures.append(f"dependency lock contains conflicting versions for {name}")
        requirements[name] = (version, conditional)

    if not requirements:
        failures.append("dependency lock has no pinned runtime requirements")
    if "websockets" not in requirements or requirements["websockets"][1]:
        failures.append("dependency lock does not unconditionally pin websockets")
    return requirements


def _validate_model_manifest(
    snapshot: _EvidenceSnapshot,
    model_revision: object,
    artifacts: list[ArtifactEvidence],
    failures: list[str],
) -> None:
    if snapshot.content is None:
        failures.append(f"model manifest exceeds {_MAX_MODEL_MANIFEST_BYTES} bytes")
        return
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"model manifest is not strict JSON: {error}")
        return
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "model_id",
        "model_revision",
        "files",
    }:
        failures.append("model manifest has an invalid top-level schema")
        return
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
    ):
        failures.append("model manifest has an unsupported schema version")
    model_id = payload.get("model_id")
    if not _valid_model_id(model_id):
        failures.append("model manifest does not identify a valid model")
    if payload.get("model_revision") != model_revision:
        failures.append("model manifest revision does not match the gate reports")

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        failures.append("model manifest has no model files")
        return
    manifest_files: list[tuple[str, str, int]] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"name", "sha256", "bytes"}:
            failures.append("model manifest contains an invalid file entry")
            return
        name = item.get("name")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not _positive_integer(size)
        ):
            failures.append("model manifest contains an invalid file identity")
            return
        manifest_files.append((name, digest, cast(int, size)))
    if len({name for name, _digest, _size in manifest_files}) != len(manifest_files):
        failures.append("model manifest contains duplicate file names")
        return

    actual_files = [(item.name, item.sha256, item.bytes) for item in artifacts]
    if len({name for name, _digest, _size in actual_files}) != len(actual_files):
        failures.append("model artifacts contain duplicate file names")
    elif sorted(manifest_files) != sorted(actual_files):
        failures.append("model manifest does not match the retained model artifacts")


def _validate_wheel(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    failures: list[str],
) -> str | None:
    if snapshot.content is None:
        failures.append(f"wheel exceeds {_MAX_WHEEL_BYTES} bytes")
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot.content)) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_WHEEL_ENTRIES:
                failures.append("wheel has an invalid number of archive entries")
                return None
            names = [item.filename for item in infos]
            if len(set(names)) != len(names):
                failures.append("wheel contains duplicate archive entries")
                return None
            total_size = 0
            contents: dict[str, bytes] = {}
            for info in infos:
                parts = info.filename.rstrip("/").split("/")
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    not info.filename
                    or info.filename.startswith("/")
                    or "\\" in info.filename
                    or "\x00" in info.filename
                    or any(part in {"", ".", ".."} for part in parts)
                    or info.flag_bits & 0x1
                    or stat.S_ISLNK(mode)
                ):
                    failures.append("wheel contains an unsafe archive entry")
                    return None
                if info.is_dir():
                    continue
                total_size += info.file_size
                if total_size > _MAX_WHEEL_UNCOMPRESSED_BYTES:
                    failures.append(
                        "wheel exceeds the uncompressed release-artifact limit"
                    )
                    return None
                contents[info.filename] = archive.read(info)
    except Exception as error:  # noqa: BLE001 - untrusted wheel is an evidence boundary
        failures.append(
            f"wheel is not a valid readable ZIP archive: {type(error).__name__}"
        )
        return None

    metadata_paths = [name for name in contents if name.endswith(".dist-info/METADATA")]
    wheel_paths = [name for name in contents if name.endswith(".dist-info/WHEEL")]
    record_paths = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if not (
        len(metadata_paths) == len(wheel_paths) == len(record_paths) == 1
        and metadata_paths[0].rsplit("/", 1)[0]
        == wheel_paths[0].rsplit("/", 1)[0]
        == record_paths[0].rsplit("/", 1)[0]
    ):
        failures.append("wheel does not contain one coherent dist-info directory")
        return None
    if "turnalign/__init__.py" not in contents:
        failures.append("wheel does not contain the TurnAlign package")
    source_identity = contents.get("turnalign/_source_commit.txt")
    if source_identity != f"{source_commit}\n".encode("ascii"):
        failures.append("wheel source commit does not match the production source commit")

    package_version: str | None = None
    try:
        metadata = BytesParser(policy=email_policy).parsebytes(contents[metadata_paths[0]])
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        if (
            len(names) != 1
            or _package_name(str(names[0])) != "turnalign"
            or len(versions) != 1
            or not str(versions[0]).strip()
        ):
            failures.append("wheel metadata does not identify one TurnAlign release")
        else:
            package_version = str(versions[0]).strip()

        wheel_metadata = BytesParser(policy=email_policy).parsebytes(
            contents[wheel_paths[0]]
        )
        if (
            wheel_metadata.get_all("Wheel-Version", []) != ["1.0"]
            or wheel_metadata.get_all("Root-Is-Purelib", []) != ["true"]
            or "py3-none-any" not in wheel_metadata.get_all("Tag", [])
        ):
            failures.append("wheel metadata is not the expected pure Python wheel")
    except (UnicodeError, ValueError, TypeError):
        failures.append("wheel contains invalid package metadata")

    entry_points = next(
        (
            payload
            for name, payload in contents.items()
            if name.endswith(".dist-info/entry_points.txt")
        ),
        None,
    )
    try:
        entry_point_config = configparser.ConfigParser(
            interpolation=None,
            strict=True,
        )
        if entry_points is None:
            raise ValueError("missing entry_points.txt")
        entry_point_config.read_string(entry_points.decode("utf-8"))
        valid_entry_point = (
            entry_point_config.get(
                "console_scripts",
                "turnalign",
                fallback=None,
            )
            == "turnalign.cli:main"
        )
    except (UnicodeDecodeError, configparser.Error, ValueError):
        valid_entry_point = False
    if not valid_entry_point:
        failures.append("wheel does not expose the TurnAlign console entry point")

    try:
        record_text = contents[record_paths[0]].decode("utf-8")
        rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except (UnicodeDecodeError, csv.Error):
        failures.append("wheel RECORD is not valid UTF-8 CSV")
        return None
    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or row[0] in recorded:
            failures.append("wheel RECORD contains malformed or duplicate entries")
            return None
        recorded[row[0]] = (row[1], row[2])
    if set(recorded) != set(contents):
        failures.append("wheel RECORD does not enumerate every archive file exactly once")
        return None
    for name, content in contents.items():
        digest, size = recorded[name]
        if name == record_paths[0]:
            if digest or size:
                failures.append("wheel RECORD must leave its own hash and size empty")
            continue
        expected_digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(
            b"="
        ).decode("ascii")
        if digest != f"sha256={expected_digest}" or size != str(len(content)):
            failures.append(f"wheel RECORD hash or size does not match: {name}")
            return None
    return package_version


def _validate_host_profile(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    wheel_version: str | None,
    artifacts: list[ArtifactEvidence],
    failures: list[str],
) -> None:
    if snapshot.content is None:
        failures.append(f"host profile exceeds {_MAX_HOST_PROFILE_BYTES} bytes")
        return
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"host profile is not strict JSON: {error}")
        return
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "source_commit",
        "runtime",
        "platform",
        "artifacts",
    }:
        failures.append("host profile has an invalid top-level schema")
        return
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
    ):
        failures.append("host profile has an unsupported schema version")
    if payload.get("source_commit") != source_commit:
        failures.append("host profile is not bound to the production source commit")

    runtime = payload.get("runtime")
    expected_prefix = f"/opt/turnalign/releases/{source_commit}/venv"
    expected_executable = f"{expected_prefix}/bin/python"
    if not (
        isinstance(runtime, dict)
        and set(runtime) == {
            "python_executable",
            "python_prefix",
            "turnalign_source_commit",
            "turnalign_version",
        }
        and runtime.get("python_executable") == expected_executable
        and runtime.get("python_prefix") == expected_prefix
        and runtime.get("turnalign_source_commit") == source_commit
        and isinstance(runtime.get("turnalign_version"), str)
        and bool(runtime["turnalign_version"])
        and runtime["turnalign_version"] == wheel_version
    ):
        failures.append(
            "host profile is not bound to the installed versioned Wheel runtime"
        )

    platform_data = payload.get("platform")
    text_fields = (
        "system",
        "release",
        "machine",
        "python_implementation",
        "python_version",
    )
    if not (
        isinstance(platform_data, dict)
        and set(platform_data) == {*text_fields, "logical_cpu_count"}
        and platform_data.get("system") == "Linux"
        and all(
            isinstance(platform_data.get(name), str)
            and bool(platform_data[name])
            and platform_data[name].strip() == platform_data[name]
            and len(platform_data[name]) <= 256
            and not any(
                ord(character) < 32 or ord(character) == 127
                for character in platform_data[name]
            )
            for name in text_fields
        )
        and _positive_integer(platform_data.get("logical_cpu_count"))
    ):
        failures.append("host profile lacks complete typed platform evidence")

    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        failures.append("host profile has no retained artifact evidence")
        return
    reported: list[tuple[str, str, str, int]] = []
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "kind",
            "name",
            "sha256",
            "bytes",
        }:
            failures.append("host profile contains an invalid artifact entry")
            return
        kind = item.get("kind")
        name = item.get("name")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(kind, str)
            or kind not in REQUIRED_ARTIFACT_KINDS - {"host-profile"}
            or not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not _positive_integer(size)
        ):
            failures.append("host profile contains an invalid artifact identity")
            return
        reported.append((kind, name, digest, cast(int, size)))
    if len({(kind, name) for kind, name, _digest, _size in reported}) != len(
        reported
    ):
        failures.append("host profile contains duplicate artifact identities")
        return
    actual = [
        (item.kind, item.name, item.sha256, item.bytes)
        for item in artifacts
        if item.kind != "host-profile"
    ]
    if sorted(reported) != sorted(actual):
        failures.append("host profile does not match the retained deployment artifacts")


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

    release, release_evidence = _load_report(release_path)
    quality, quality_evidence = _load_report(quality_path)
    websocket, websocket_evidence = _load_report(websocket_path)
    failures: list[str] = []
    _validate_release(release, failures)
    _validate_quality(quality, failures)
    _validate_websocket(websocket, failures)

    artifact_evidence = []
    evidence_by_kind: dict[str, list[ArtifactEvidence]] = {}
    kinds = set()
    artifact_paths: dict[str, list[Path]] = {}
    artifact_digests: dict[str, list[str]] = {}
    artifact_snapshots: dict[str, list[_EvidenceSnapshot]] = {}
    for kind, path in artifacts:
        if kind not in REQUIRED_ARTIFACT_KINDS:
            raise ValueError(f"unsupported artifact kind: {kind}")
        capture_limit = {
            "dependency-lock": _MAX_LOCK_BYTES,
            "host-profile": _MAX_HOST_PROFILE_BYTES,
            "model-manifest": _MAX_MODEL_MANIFEST_BYTES,
            "nginx-config": _MAX_DEPLOYMENT_CONFIG_BYTES,
            "sbom": _MAX_SBOM_BYTES,
            "wheel": _MAX_WHEEL_BYTES,
            "service-unit": _MAX_DEPLOYMENT_CONFIG_BYTES,
        }.get(kind)
        snapshot = _snapshot_evidence(path, capture_limit=capture_limit)
        evidence = ArtifactEvidence(kind, path.name, snapshot.sha256, snapshot.size)
        artifact_evidence.append(evidence)
        evidence_by_kind.setdefault(kind, []).append(evidence)
        kinds.add(kind)
        artifact_paths.setdefault(kind, []).append(path)
        artifact_digests.setdefault(kind, []).append(snapshot.sha256)
        artifact_snapshots.setdefault(kind, []).append(snapshot)
    for missing in sorted(REQUIRED_ARTIFACT_KINDS - kinds):
        failures.append(f"missing required artifact kind: {missing}")
    for kind in sorted(REQUIRED_ARTIFACT_KINDS - {"model"}):
        if len(artifact_paths.get(kind, [])) > 1:
            failures.append(f"artifact kind must appear exactly once: {kind}")

    sbom_snapshots = artifact_snapshots.get("sbom", [])
    lock_snapshots = artifact_snapshots.get("dependency-lock", [])
    if len(sbom_snapshots) == 1:
        sbom_components = _validate_sbom(sbom_snapshots[0], failures)
    else:
        sbom_components = {}
    if len(lock_snapshots) == 1:
        locked_requirements = _validate_dependency_lock(lock_snapshots[0], failures)
    else:
        locked_requirements = {}
    for name, (version, conditional) in locked_requirements.items():
        if conditional:
            continue
        if sbom_components.get(name) != version:
            failures.append(
                f"SBOM does not match locked runtime requirement: {name}=={version}"
            )
    for name, version in sbom_components.items():
        locked = locked_requirements.get(name)
        if locked is None:
            failures.append(f"SBOM contains an unlocked runtime component: {name}=={version}")
        elif locked[0] != version and locked[1]:
            failures.append(
                f"SBOM does not match locked runtime requirement: {name}=={locked[0]}"
            )

    wheel_version: str | None = None
    wheel_snapshots = artifact_snapshots.get("wheel", [])
    if len(wheel_snapshots) == 1:
        wheel_version = _validate_wheel(wheel_snapshots[0], source_commit, failures)

    model_manifest_snapshots = artifact_snapshots.get("model-manifest", [])
    if len(model_manifest_snapshots) == 1:
        _validate_model_manifest(
            model_manifest_snapshots[0],
            release.get("model_revision"),
            evidence_by_kind.get("model", []),
            failures,
        )
    host_profile_snapshots = artifact_snapshots.get("host-profile", [])
    if len(host_profile_snapshots) == 1:
        _validate_host_profile(
            host_profile_snapshots[0],
            source_commit,
            wheel_version,
            artifact_evidence,
            failures,
        )
    service_content: bytes | None = None
    service_snapshots = artifact_snapshots.get("service-unit", [])
    if len(service_snapshots) == 1:
        service_content = service_snapshots[0].content
        if service_content is None:
            failures.append(
                f"systemd service unit exceeds {_MAX_DEPLOYMENT_CONFIG_BYTES} bytes"
            )
        else:
            failures.extend(validate_systemd_service(service_content))
    nginx_snapshots = artifact_snapshots.get("nginx-config", [])
    if len(nginx_snapshots) == 1:
        nginx_content = nginx_snapshots[0].content
        websocket_uri = websocket.get("uri")
        if nginx_content is None:
            failures.append(
                f"Nginx configuration exceeds {_MAX_DEPLOYMENT_CONFIG_BYTES} bytes"
            )
        elif service_content is None:
            failures.append("Nginx cannot be checked without bounded systemd evidence")
        elif not isinstance(websocket_uri, str):
            failures.append("Nginx cannot be checked without a WebSocket report URI")
        else:
            failures.extend(
                validate_nginx_config(
                    nginx_content,
                    service_content,
                    websocket_uri,
                )
            )

    for name, report in (
        ("release", release),
        ("quality", quality),
        ("websocket", websocket),
    ):
        _validate_report_source(name, report, source_commit, failures)
    if release.get("model_revision") != quality.get("model_revision"):
        failures.append("quality and release reports identify different model revisions")
    if websocket.get("backend_implementation") != release.get("backend"):
        failures.append("websocket and release reports identify different backends")
    if websocket.get("model_revision") != release.get("model_revision"):
        failures.append("websocket and release reports identify different model revisions")
    for report, field, kind, label in (
        (release, "input_audio_sha256", "release-audio", "release audio"),
        (quality, "reference_sha256", "quality-reference", "quality reference"),
        (quality, "hypothesis_sha256", "quality-hypothesis", "quality hypothesis"),
    ):
        reported_digest = report.get(field)
        if (
            not isinstance(reported_digest, str)
            or _SHA256_PATTERN.fullmatch(reported_digest) is None
        ):
            failures.append(f"{label} report digest is missing or invalid")
        elif len(artifact_digests.get(kind, [])) == 1 and (
            artifact_digests[kind][0] != reported_digest
        ):
            failures.append(f"{label} report digest does not match its artifact")

    return ProductionGateReport(
        schema_version=1,
        status="failed" if failures else "passed",
        source_commit=source_commit,
        release_report=release_evidence,
        quality_report=quality_evidence,
        websocket_report=websocket_evidence,
        artifacts=tuple(sorted(
            artifact_evidence,
            key=lambda item: (item.kind, item.name, item.sha256),
        )),
        failures=tuple(failures),
    )
