from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .jsonutil import strict_json_loads

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODEL_REVISION_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_REPORT_BYTES = 2 * 1024 * 1024
_MAX_SBOM_BYTES = 16 * 1024 * 1024
_MAX_LOCK_BYTES = 4 * 1024 * 1024
_LOCK_REQUIREMENT_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)(?:\s|;|$)"
)
_EMBEDDED_REQUIREMENT_PATTERN = re.compile(
    r"\s[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9]"
)
_SHA256_OPTION_PATTERN = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})(?:\s|$)")
REQUIRED_ARTIFACT_KINDS = frozenset({
    "dependency-lock",
    "host-profile",
    "model",
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
    revision = report.get("model_revision")
    if not isinstance(revision, str) or _MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        failures.append("quality report does not identify an immutable model revision")
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
    kinds = set()
    artifact_paths: dict[str, list[Path]] = {}
    artifact_digests: dict[str, list[str]] = {}
    artifact_snapshots: dict[str, list[_EvidenceSnapshot]] = {}
    for kind, path in artifacts:
        if kind not in REQUIRED_ARTIFACT_KINDS:
            raise ValueError(f"unsupported artifact kind: {kind}")
        capture_limit = {
            "dependency-lock": _MAX_LOCK_BYTES,
            "sbom": _MAX_SBOM_BYTES,
        }.get(kind)
        snapshot = _snapshot_evidence(path, capture_limit=capture_limit)
        artifact_evidence.append(
            ArtifactEvidence(kind, path.name, snapshot.sha256, snapshot.size)
        )
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

    for name, report in (
        ("release", release),
        ("quality", quality),
        ("websocket", websocket),
    ):
        _validate_report_source(name, report, source_commit, failures)
    if release.get("model_revision") != quality.get("model_revision"):
        failures.append("quality and release reports identify different model revisions")
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
