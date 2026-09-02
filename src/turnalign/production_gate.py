from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import importlib.metadata
import importlib.resources
import io
import ipaddress
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import unquote, urlsplit

from . import __version__
from .deployment_validation import validate_nginx_config, validate_systemd_service
from .jsonutil import strict_json_loads

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODEL_REVISION_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRANSACTION_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
)
_DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_MAX_REPORT_BYTES = 2 * 1024 * 1024
_MAX_SBOM_BYTES = 16 * 1024 * 1024
_MAX_LOCK_BYTES = 4 * 1024 * 1024
_MAX_MODEL_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_MODEL_FILES = 65_536
_MAX_MODEL_BYTES = 64 * 1024 * 1024 * 1024
_MAX_HOST_PROFILE_BYTES = 2 * 1024 * 1024
_MAX_DEPLOYMENT_ACTIVATION_BYTES = 8 * 1024 * 1024
_MAX_ROLLBACK_REHEARSAL_BYTES = 8 * 1024 * 1024
_MAX_DEPLOYMENT_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_WHEEL_BYTES = 64 * 1024 * 1024
_MAX_WHEEL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_WHEEL_ENTRIES = 4_096
_MAX_DEPENDENCY_FILES = 250_000
_MAX_DEPENDENCY_BYTES = 128 * 1024 * 1024 * 1024
_MAX_REPORT_VALIDITY_SECONDS = 86_400.0
_MAX_DEPLOYMENT_STATE_VALIDITY_SECONDS = 300.0
_MAX_CLOCK_SKEW_SECONDS = 300.0
_DEPLOYMENT_TRANSACTION_PATH = Path(
    "/var/lib/turnalign-deployment/pending-activation.json"
)
_DEPLOYMENT_LOCK_PATH = Path("/run/lock/turnalign-deployment.lock")
_RELEASE_ROOT = Path("/opt/turnalign/releases")
_CURRENT_RELEASE_LINK = Path("/opt/turnalign/current")
_MODEL_EVIDENCE_ROOT = Path("/var/lib/turnalign/models")
_SERVICE_UNIT_ID = "/etc/systemd/system/turnalign.service"
_NGINX_CONFIG_ID = "/etc/nginx/conf.d/turnalign.conf"
_SERVICE_UNIT_PATH = Path(_SERVICE_UNIT_ID)
_NGINX_CONFIG_PATH = Path(_NGINX_CONFIG_ID)
_SYSTEMCTL_PATH = Path("/usr/bin/systemctl")
_NGINX_PATH = Path("/usr/sbin/nginx")
_EFFECTIVE_CONFIG_TIMEOUT_SECONDS = 15.0
_MAX_EFFECTIVE_CONFIG_OUTPUT_BYTES = 8 * 1024 * 1024
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
    "deployment-activation",
    "deployment-state",
    "dependency-lock",
    "host-profile",
    "model",
    "model-manifest",
    "nginx-config",
    "quality-hypothesis",
    "quality-reference",
    "release-audio",
    "rollback-rehearsal",
    "service-unit",
    "sbom",
    "websocket-probe-audio",
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
class _WheelIdentity:
    version: str
    package_files: tuple[EvidenceFile, ...]


@dataclass(frozen=True, slots=True)
class _DeploymentIdentity:
    boot_id: str
    previous_commit: str


@dataclass(frozen=True, slots=True)
class _HostProfileIdentity:
    boot_id: str
    installed_dependencies: dict[str, str]


@dataclass(frozen=True, slots=True)
class _DeploymentStateIdentity:
    boot_id: str
    active_commit: str
    pending_transaction: str | None


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
    """Atomically and durably persist a gate report on production POSIX hosts."""

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
        _fsync_directory(path.parent)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    """Persist a replaced directory entry before deployment state can advance."""

    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _root_owned_immutable(metadata: os.stat_result) -> bool:
    return metadata.st_uid == 0 and not stat.S_IMODE(metadata.st_mode) & 0o022


def _model_artifact_root(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    try:
        parent_paths = [str(path.resolve(strict=False).parent) for path in paths]
        return Path(os.path.commonpath(parent_paths))
    except (OSError, ValueError):
        return None


def _artifact_identity_name(
    kind: str,
    path: Path,
    model_root: Path | None = None,
) -> str:
    if kind != "model" or model_root is None:
        return path.name
    try:
        return path.resolve(strict=False).relative_to(model_root).as_posix()
    except ValueError:
        return path.name


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


def enumerate_model_files(model_root: Path) -> list[Path]:
    """Safely enumerate every regular file retained under a model root."""

    if (
        not model_root.is_absolute()
        or Path(os.path.normpath(str(model_root))) != model_root
    ):
        raise ValueError("model_root must be an absolute normalized path")
    try:
        resolved_root = model_root.resolve(strict=True)
        root_metadata = model_root.lstat()
    except OSError as error:
        raise ValueError(f"model_root is unavailable: {model_root}") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not resolved_root.is_dir():
        raise ValueError(f"model_root must be a non-symlink directory: {model_root}")
    immutable_required = (
        resolved_root == _MODEL_EVIDENCE_ROOT
        or resolved_root.is_relative_to(_MODEL_EVIDENCE_ROOT)
    )
    if immutable_required:
        for ancestor in tuple(reversed(resolved_root.parents)) + (resolved_root,):
            try:
                metadata = ancestor.lstat()
            except OSError as error:
                raise ValueError(
                    f"model root ancestor is unavailable: {ancestor}"
                ) from error
            if not stat.S_ISDIR(metadata.st_mode) or not _root_owned_immutable(metadata):
                raise ValueError(
                    "canonical model root and all ancestors must be root-owned "
                    "and not writable by group/others: "
                    f"{ancestor}"
                )

    entries: list[Path] = []
    total_size = 0
    for directory, directory_names, file_names in os.walk(
        resolved_root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or (immutable_required and not _root_owned_immutable(metadata))
            ):
                raise ValueError(f"model tree contains an unsafe directory: {child}")
        for name in file_names:
            child = directory_path / name
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or (immutable_required and not _root_owned_immutable(metadata))
            ):
                raise ValueError(f"model tree contains an unsafe file: {child}")
            if metadata.st_size <= 0:
                raise ValueError(f"model tree contains an empty file: {child}")
            total_size += metadata.st_size
            if len(entries) >= _MAX_MODEL_FILES or total_size > _MAX_MODEL_BYTES:
                raise ValueError("model tree exceeds production evidence limits")
            entries.append(child)
    if not entries:
        raise ValueError(f"model_root contains no regular files: {model_root}")
    return entries


def create_model_manifest(
    model_id: str,
    model_revision: str,
    model_root: Path,
) -> dict[str, object]:
    """Hash immutable model files into the schema consumed by production-gate."""

    if not _valid_model_id(model_id):
        raise ValueError("model_id must be a non-empty bounded identifier")
    if _MODEL_REVISION_PATTERN.fullmatch(model_revision) is None:
        raise ValueError("model_revision must be an immutable 40- or 64-character hash")
    files = enumerate_model_files(model_root)
    resolved_root = model_root.resolve(strict=True)
    entries = []
    for path in files:
        relative_path = path.relative_to(resolved_root).as_posix()
        snapshot = _snapshot_evidence(path)
        entries.append({
            "path": relative_path,
            "sha256": snapshot.sha256,
            "bytes": snapshot.size,
        })
    payload: dict[str, object] = {
        "schema_version": 2,
        "model_id": model_id,
        "model_revision": model_revision,
        "files": sorted(entries, key=lambda item: cast(str, item["path"])),
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_MODEL_MANIFEST_BYTES:
        raise ValueError(
            f"model manifest exceeds {_MAX_MODEL_MANIFEST_BYTES} encoded bytes"
        )
    return payload


def create_host_profile(
    source_commit: str | None,
    artifacts: list[tuple[str, Path]],
) -> dict[str, object]:
    """Capture host identity and bind every other retained production artifact."""

    system = platform.system()
    if system != "Linux":
        raise RuntimeError("host-profile must run on the Linux production host")
    lock_descriptor = _acquire_deployment_lock()
    try:
        _reject_pending_deployment_transaction()
        return _create_host_profile_locked(source_commit, artifacts, system)
    finally:
        os.close(lock_descriptor)


def create_deployment_state(validity_seconds: float = 300.0) -> dict[str, object]:
    """Capture the current live release and pending transaction on the target host."""

    system = platform.system()
    if system != "Linux":
        raise RuntimeError("deployment-state must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment-state must run as root")
    if (
        isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, (int, float))
        or not math.isfinite(validity_seconds)
        or validity_seconds <= 0
        or validity_seconds > _MAX_DEPLOYMENT_STATE_VALIDITY_SECONDS
    ):
        raise ValueError(
            "validity_seconds must be finite, positive, and no greater than 300"
        )
    lock_descriptor = _acquire_deployment_lock()
    try:
        active_commit = _active_release_commit()
        boot_id = _read_linux_boot_id()
        effective_configuration = _capture_effective_configuration()
        pending_transaction_id: str | None = None
        try:
            os.lstat(_DEPLOYMENT_TRANSACTION_PATH)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeError("cannot inspect pending deployment state") from error
        else:
            pending_payload = _load_report(_DEPLOYMENT_TRANSACTION_PATH)[0]
            raw_transaction_id = pending_payload.get("transaction_id")
            if (
                not isinstance(raw_transaction_id, str)
                or _TRANSACTION_ID_PATTERN.fullmatch(raw_transaction_id) is None
            ):
                raise RuntimeError("pending deployment transaction is invalid")
            pending_transaction_id = raw_transaction_id
        return {
            "schema_version": 2,
            "active_commit": active_commit,
            "pending_transaction_id": pending_transaction_id,
            "boot_id": boot_id,
            "effective_configuration": effective_configuration,
            "created_at": (
                datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            ),
            "validity_seconds": round(validity_seconds, 3),
        }
    finally:
        os.close(lock_descriptor)


def _acquire_deployment_lock() -> int:
    import fcntl

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            _DEPLOYMENT_LOCK_PATH,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
            0o600,
        )
    except OSError as error:
        raise RuntimeError("cannot securely open the deployment lock") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError("deployment lock must be a root-only regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another TurnAlign deployment operation is active"
            ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _reject_pending_deployment_transaction() -> None:
    try:
        os.lstat(_DEPLOYMENT_TRANSACTION_PATH)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RuntimeError("cannot inspect pending deployment state") from error
    else:
        raise RuntimeError(
            "host-profile refuses a pending deployment transaction; recover it first"
        )


def _create_host_profile_locked(
    source_commit: str | None,
    artifacts: list[tuple[str, Path]],
    system: str,
) -> dict[str, object]:
    runtime = _installed_runtime_identity(source_commit)
    bound_commit = runtime["turnalign_source_commit"]
    active_commit = _active_release_commit()
    if active_commit != bound_commit:
        raise ValueError(
            "host-profile runtime is not the active production release"
        )
    installed_distribution = _installed_distribution_identity(runtime)
    dependency_lock_path = next(
        path for kind, path in artifacts if kind == "dependency-lock"
    )
    installed_dependencies = _installed_dependency_identity(
        dependency_lock_path,
        runtime,
    )
    expected_kinds = REQUIRED_ARTIFACT_KINDS - {"host-profile"}
    evidence = []
    kinds = set()
    identities = set()
    configuration_snapshots: dict[str, _EvidenceSnapshot] = {}
    model_paths = [path for kind, path in artifacts if kind == "model"]
    model_root = _model_artifact_root(model_paths)
    for kind, path in artifacts:
        if kind not in expected_kinds:
            raise ValueError(f"unsupported host-profile artifact kind: {kind}")
        if kind == "service-unit" and Path(os.path.abspath(str(path))) != _SERVICE_UNIT_PATH:
            raise ValueError(
                "host-profile service-unit must use the active canonical systemd unit"
            )
        if kind == "nginx-config" and Path(os.path.abspath(str(path))) != _NGINX_CONFIG_PATH:
            raise ValueError(
                "host-profile nginx-config must use the active canonical Nginx config"
            )
        if kind in {"service-unit", "nginx-config"}:
            _require_root_owned_production_config(path)
            snapshot = _snapshot_evidence(
                path,
                capture_limit=_MAX_DEPLOYMENT_CONFIG_BYTES,
                hard_limit=_MAX_DEPLOYMENT_CONFIG_BYTES,
            )
            configuration_snapshots[kind] = snapshot
        else:
            snapshot = _snapshot_evidence(path)
        identity_name = _artifact_identity_name(kind, path, model_root)
        identity = (kind, identity_name)
        if identity in identities:
            raise ValueError(
                f"host-profile artifacts require unique kind/name pairs: {kind}={identity_name}"
            )
        identities.add(identity)
        kinds.add(kind)
        evidence.append({
            "kind": kind,
            "name": identity_name,
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
        "schema_version": 7,
        "source_commit": bound_commit,
        "active_commit": active_commit,
        "runtime": runtime,
        "installed_distribution": installed_distribution,
        "installed_dependencies": installed_dependencies,
        "platform": {
            "system": system,
            "boot_id": _read_linux_boot_id(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "logical_cpu_count": logical_cpu_count,
        },
        "effective_configuration": _capture_effective_configuration(
            service_snapshot=configuration_snapshots["service-unit"],
            nginx_snapshot=configuration_snapshots["nginx-config"],
        ),
        "artifacts": sorted(
            evidence,
            key=lambda item: (cast(str, item["kind"]), cast(str, item["name"])),
        ),
    }


def _require_root_owned_production_config(path: Path) -> None:
    ancestors = tuple(reversed(path.parents)) + (path,)
    for item in ancestors:
        try:
            metadata = os.lstat(item)
        except OSError as error:
            raise ValueError(
                f"production configuration is unavailable: {path}"
            ) from error
        expected_type = stat.S_ISREG if item == path else stat.S_ISDIR
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not expected_type(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError(
                "production configuration and all ancestors must be root-owned, "
                "non-symlink, correctly typed, and not writable by group/others: "
                f"{item}"
            )


def _run_effective_config_command(command: list[str]) -> tuple[bytes, bytes]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=_EFFECTIVE_CONFIG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"effective configuration command timed out: {command[0]}"
            ) from error
        stdout_size = stdout.tell()
        stderr_size = stderr.tell()
        if (
            stdout_size > _MAX_EFFECTIVE_CONFIG_OUTPUT_BYTES
            or stderr_size > _MAX_EFFECTIVE_CONFIG_OUTPUT_BYTES
        ):
            raise RuntimeError(
                f"effective configuration command output is too large: {command[0]}"
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"effective configuration command failed: {command[0]}"
            )
        stdout.seek(0)
        stderr.seek(0)
        return stdout.read(), stderr.read()


def _capture_systemd_effective_configuration(
    service_snapshot: _EvidenceSnapshot,
) -> dict[str, object]:
    stdout, _stderr = _run_effective_config_command([
        str(_SYSTEMCTL_PATH),
        "show",
        "turnalign.service",
        "--no-pager",
        "--property=FragmentPath",
        "--property=DropInPaths",
        "--property=NeedDaemonReload",
        "--property=ActiveState",
        "--property=SubState",
    ])
    try:
        lines = stdout.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("systemd effective configuration is not UTF-8") from error
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError("systemd effective configuration is malformed")
        fields[key] = value
    expected = {
        "FragmentPath": _SERVICE_UNIT_ID,
        "DropInPaths": "",
        "NeedDaemonReload": "no",
        "ActiveState": "active",
        "SubState": "running",
    }
    if fields != expected:
        raise RuntimeError(
            "systemd has not loaded the canonical active TurnAlign unit without drop-ins"
        )
    return {
        "fragment_path": fields["FragmentPath"],
        "drop_in_paths": [],
        "need_daemon_reload": False,
        "active_state": fields["ActiveState"],
        "sub_state": fields["SubState"],
        "sha256": service_snapshot.sha256,
        "bytes": service_snapshot.size,
    }


def _capture_nginx_effective_configuration(
    nginx_snapshot: _EvidenceSnapshot,
) -> dict[str, object]:
    if nginx_snapshot.content is None:
        raise RuntimeError("canonical Nginx configuration was not retained for capture")
    stdout, stderr = _run_effective_config_command([str(_NGINX_PATH), "-T"])
    if b"[warn]" in stderr.lower():
        raise RuntimeError("Nginx effective configuration contains warnings")
    marker = f"# configuration file {_NGINX_CONFIG_ID}:\n".encode()
    starts: list[int] = []
    cursor = 0
    while True:
        found = stdout.find(marker, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + len(marker)
    if len(starts) != 1:
        raise RuntimeError(
            "Nginx effective configuration must load the canonical TurnAlign file once"
        )
    content_start = starts[0] + len(marker)
    next_marker = stdout.find(b"\n# configuration file ", content_start)
    content_end = len(stdout) if next_marker < 0 else next_marker
    loaded_content = stdout[content_start:content_end]
    if (
        loaded_content != nginx_snapshot.content
        and loaded_content + b"\n" != nginx_snapshot.content
    ):
        raise RuntimeError(
            "Nginx effective configuration does not match the canonical TurnAlign file"
        )
    return {
        "configuration_path": _NGINX_CONFIG_ID,
        "loaded_occurrences": 1,
        "warning_free": True,
        "sha256": nginx_snapshot.sha256,
        "bytes": nginx_snapshot.size,
    }


def _capture_effective_configuration(
    *,
    service_snapshot: _EvidenceSnapshot | None = None,
    nginx_snapshot: _EvidenceSnapshot | None = None,
) -> dict[str, object]:
    if service_snapshot is None or nginx_snapshot is None:
        _require_root_owned_production_config(_SERVICE_UNIT_PATH)
        _require_root_owned_production_config(_NGINX_CONFIG_PATH)
        service_snapshot = _snapshot_evidence(
            _SERVICE_UNIT_PATH,
            capture_limit=_MAX_DEPLOYMENT_CONFIG_BYTES,
            hard_limit=_MAX_DEPLOYMENT_CONFIG_BYTES,
        )
        nginx_snapshot = _snapshot_evidence(
            _NGINX_CONFIG_PATH,
            capture_limit=_MAX_DEPLOYMENT_CONFIG_BYTES,
            hard_limit=_MAX_DEPLOYMENT_CONFIG_BYTES,
        )
    return {
        "systemd": _capture_systemd_effective_configuration(service_snapshot),
        "nginx": _capture_nginx_effective_configuration(nginx_snapshot),
    }


def _active_release_commit() -> str:
    release_root = _RELEASE_ROOT
    current_link = _CURRENT_RELEASE_LINK
    try:
        metadata = current_link.lstat()
        raw_target = os.readlink(current_link)
    except OSError as error:
        raise ValueError("host-profile cannot read the active release link") from error
    if (
        not stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != _required_deployment_owner()
    ):
        raise ValueError("host-profile active release link must be root-owned")
    target = Path(raw_target)
    if not target.is_absolute():
        raise ValueError("host-profile active release link must use an absolute target")
    normalized = Path(os.path.normpath(raw_target))
    commit = normalized.name
    if (
        normalized.parent != release_root
        or normalized != release_root / commit
        or _COMMIT_PATTERN.fullmatch(commit) is None
    ):
        raise ValueError("host-profile active release link is not canonical")
    try:
        target_metadata = normalized.lstat()
    except OSError as error:
        raise ValueError("host-profile active release directory is missing") from error
    if (
        not stat.S_ISDIR(target_metadata.st_mode)
        or stat.S_ISLNK(target_metadata.st_mode)
        or not _root_owned_immutable(target_metadata)
    ):
        raise ValueError("host-profile active release directory is unsafe or mutable")
    return commit


def _required_deployment_owner() -> int:
    return 0


def _read_linux_boot_id() -> str:
    """Read the current Linux boot identity without following a replacement link."""

    path = Path("/proc/sys/kernel/random/boot_id")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
    except OSError as error:
        raise RuntimeError("cannot securely read the Linux boot identity") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Linux boot identity is not a regular file")
        raw = os.read(descriptor, 65)
        if os.read(descriptor, 1):
            raise RuntimeError("Linux boot identity exceeds its expected size")
    finally:
        os.close(descriptor)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("Linux boot identity is not ASCII") from error
    if not text.endswith("\n"):
        raise RuntimeError("Linux boot identity is not newline-terminated")
    value = text[:-1]
    if _BOOT_ID_PATTERN.fullmatch(value) is None:
        raise RuntimeError("Linux boot identity has an invalid format")
    return value


def _installed_runtime_identity(source_commit: str | None = None) -> dict[str, str]:
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
        raise ValueError(
            "production host commands require Python -I -B before -m turnalign.cli"
        )
    try:
        package_resource = importlib.resources.files("turnalign")
        embedded_identity = package_resource.joinpath("_source_commit.txt").read_text(
            encoding="ascii"
        )
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
    expected_package_root = (
        Path(release_prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "turnalign"
    )
    package_root = os.path.realpath(os.path.abspath(str(package_resource)))
    expected_package = os.path.realpath(os.path.abspath(str(expected_package_root)))
    if (
        python_prefix != release_prefix
        or python_executable != f"{release_prefix}/bin/python"
        or package_root != expected_package
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


def _installed_distribution_identity(
    runtime: dict[str, str],
) -> dict[str, object]:
    """Hash the complete installed package tree used by the active interpreter."""

    try:
        distribution = importlib.metadata.distribution("turnalign")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "host-profile cannot locate the installed TurnAlign Wheel"
        ) from error
    if distribution.version != runtime["turnalign_version"]:
        raise ValueError("installed TurnAlign metadata version does not match the runtime")

    prefix = Path(runtime["python_prefix"])
    expected_root = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    try:
        distribution_root = Path(str(distribution.locate_file(""))).resolve(
            strict=True
        )
        expected_root = expected_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("host-profile cannot resolve the installed Wheel root") from error
    if distribution_root != expected_root:
        raise ValueError(
            "installed TurnAlign is not in the candidate versioned site-packages"
        )

    package_root = distribution_root / "turnalign"
    try:
        root_metadata = package_root.lstat()
    except OSError as error:
        raise ValueError("installed TurnAlign package directory is missing") from error
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or not _root_owned_immutable(root_metadata)
    ):
        raise ValueError("installed TurnAlign package must be a non-symlink directory")

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    total_size = 0
    try:
        for directory, directory_names, file_names in os.walk(
            package_root,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            for name in directory_names:
                child = directory_path / name
                metadata = child.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISDIR(metadata.st_mode)
                    or not _root_owned_immutable(metadata)
                ):
                    raise ValueError(
                        "installed TurnAlign contains an unsafe package directory"
                    )
            for name in file_names:
                path = directory_path / name
                relative = path.relative_to(distribution_root).as_posix()
                if relative in seen:
                    raise ValueError("installed TurnAlign contains duplicate package paths")
                seen.add(relative)
                metadata = path.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or not _root_owned_immutable(metadata)
                ):
                    raise ValueError(
                        "installed TurnAlign contains an unsafe or mutable package file"
                    )
                snapshot = _snapshot_evidence(path)
                total_size += snapshot.size
                if (
                    len(entries) >= _MAX_WHEEL_ENTRIES
                    or total_size > _MAX_WHEEL_UNCOMPRESSED_BYTES
                ):
                    raise ValueError("installed TurnAlign package exceeds evidence limits")
                entries.append({
                    "name": relative,
                    "sha256": snapshot.sha256,
                    "bytes": snapshot.size,
                })
    except OSError as error:
        raise ValueError(
            "cannot securely inspect the installed TurnAlign package"
        ) from error
    if not entries:
        raise ValueError("installed TurnAlign package contains no files")

    source_identity = next(
        (
            item
            for item in entries
            if item["name"] == "turnalign/_source_commit.txt"
        ),
        None,
    )
    expected_source_digest = hashlib.sha256(
        f"{runtime['turnalign_source_commit']}\n".encode("ascii")
    ).hexdigest()
    if (
        source_identity is None
        or source_identity["sha256"] != expected_source_digest
        or source_identity["bytes"] != 41
    ):
        raise ValueError(
            "installed TurnAlign package source identity changed during capture"
        )
    return {
        "name": "turnalign",
        "version": distribution.version,
        "root": str(distribution_root),
        "files": sorted(entries, key=lambda item: cast(str, item["name"])),
    }


def _installed_dependency_identity(
    lock_path: Path,
    runtime: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Hash installed runtime distributions for every pinned runtime requirement."""

    lock_snapshot = _snapshot_evidence(
        lock_path,
        capture_limit=_MAX_LOCK_BYTES,
        hard_limit=_MAX_LOCK_BYTES,
    )
    requirements = _dependency_lock_entries(lock_snapshot)
    prefix = Path(runtime["python_prefix"])
    site_packages = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    result: dict[str, dict[str, object]] = {}
    for name, (version, conditional) in sorted(requirements.items()):
        if conditional:
            continue
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"installed dependency is missing: {name}") from error
        if distribution.version != version:
            raise ValueError(
                f"installed dependency version does not match the lock: "
                f"{name}=={distribution.version} != {version}"
            )
        try:
            root = Path(str(distribution.locate_file(""))).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"installed dependency root is unavailable: {name}") from error
        if root != site_packages:
            raise ValueError(
                f"installed dependency is not in the candidate site-packages: {name}"
            )
        tree_digest = hashlib.sha256()
        file_count = 0
        total_size = 0
        seen: set[str] = set()
        for file_info in sorted(
            distribution.files or (), key=lambda item: item.as_posix()
        ):
            relative = file_info.as_posix()
            if not relative or relative.startswith(("..", "/")):
                continue
            if relative in seen:
                raise ValueError(
                    f"installed dependency contains duplicate files: {name}/{relative}"
                )
            seen.add(relative)
            path = site_packages / relative
            try:
                metadata = path.lstat()
            except OSError as error:
                raise ValueError(
                    f"installed dependency file is unavailable: {name}/{relative}"
                ) from error
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or not _root_owned_immutable(metadata)
            ):
                raise ValueError(
                    f"installed dependency contains an unsafe file: {name}/{relative}"
                )
            snapshot = _snapshot_evidence(path)
            file_count += 1
            total_size += snapshot.size
            if (
                file_count > _MAX_DEPENDENCY_FILES
                or total_size > _MAX_DEPENDENCY_BYTES
            ):
                raise ValueError(
                    f"installed dependency exceeds evidence limits: {name}"
                )
            tree_digest.update(json.dumps(
                [relative, snapshot.sha256, snapshot.size],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
            tree_digest.update(b"\n")
        if file_count == 0:
            raise ValueError(f"installed dependency has no content evidence: {name}")
        result[name] = {
            "name": name,
            "version": distribution.version,
            "root": str(site_packages),
            "file_count": file_count,
            "bytes": total_size,
            "sha256": tree_digest.hexdigest(),
        }
    return result


def _public_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        normalized = hostname.rstrip(".").lower()
        return (
            len(normalized) <= 253
            and "." in normalized
            and normalized not in {"localhost"}
            and not normalized.endswith((".local", ".localhost", ".internal"))
            and all(
                _DNS_LABEL_PATTERN.fullmatch(label) is not None
                for label in normalized.split(".")
            )
        )
    return address.is_global


def _package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _safe_relative_archive_path(value: str) -> bool:
    parts = value.split("/")
    return (
        bool(value)
        and not value.startswith("/")
        and "\\" not in value
        and "\x00" not in value
        and all(part not in {"", ".", ".."} for part in parts)
    )


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


def _validate_report_freshness(
    name: str,
    report: dict[str, object],
    failures: list[str],
    *,
    now: datetime | None = None,
    max_validity_seconds: float = _MAX_REPORT_VALIDITY_SECONDS,
) -> None:
    created = _utc_timestamp(report.get("created_at"))
    validity = report.get("validity_seconds")
    if created is None:
        failures.append(f"{name} report has no UTC creation timestamp")
    if (
        isinstance(validity, bool)
        or not isinstance(validity, (int, float))
        or not math.isfinite(validity)
        or validity <= 0
        or validity > max_validity_seconds
    ):
        failures.append(
            f"{name} report validity must be positive and no greater than "
            f"{max_validity_seconds:g} seconds"
        )
        return
    current = now or datetime.now(timezone.utc)
    if created is not None:
        if created - current > timedelta(seconds=_MAX_CLOCK_SKEW_SECONDS):
            failures.append(f"{name} report creation timestamp is in the future")
        elif current - created > timedelta(seconds=cast(float, validity)):
            failures.append(f"{name} report is stale and cannot be replayed as current")


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
    _validate_report_freshness("release", report, failures)
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
    if report.get("require_local_model") is not True:
        failures.append("release report did not require local model loading")
    model = report.get("model")
    if not _valid_model_id(model):
        failures.append("release report does not identify a valid model")
    loaded_models = report.get("loaded_models")
    if not isinstance(loaded_models, list) or not loaded_models or any(
        not isinstance(item, dict)
        or set(item) != {"path", "sha256", "bytes"}
        or not isinstance(item.get("path"), str)
        or not item["path"].startswith("/var/lib/turnalign/models/")
        or not isinstance(item.get("sha256"), str)
        or _SHA256_PATTERN.fullmatch(item["sha256"]) is None
        or not _positive_integer(item.get("bytes"))
        for item in loaded_models
    ):
        failures.append("release report has no bound loaded model evidence")
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
    _validate_report_freshness("quality", report, failures)
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
    if not _valid_model_id(report.get("model")):
        failures.append("quality report does not identify a valid model")
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
    _validate_report_freshness("websocket", report, failures)
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
    loaded_models = report.get("loaded_models")
    if not isinstance(loaded_models, list) or not loaded_models or any(
        not isinstance(item, dict)
        or set(item) != {"path", "sha256", "bytes"}
        or not isinstance(item.get("path"), str)
        or not item["path"].startswith("/var/lib/turnalign/models/")
        or not isinstance(item.get("sha256"), str)
        or _SHA256_PATTERN.fullmatch(item["sha256"]) is None
        or not _positive_integer(item.get("bytes"))
        for item in loaded_models
    ):
        failures.append("websocket report has no bound loaded model evidence")
    probe_sha256 = report.get("probe_audio_sha256")
    if not isinstance(probe_sha256, str) or _SHA256_PATTERN.fullmatch(probe_sha256) is None:
        failures.append("websocket report has no retained probe-audio digest")
    if (
        not _positive_integer(report.get("probe_audio_bytes"))
        or not _positive_number(report.get("probe_audio_rms"))
        or not _at_least(report.get("probe_audio_rms"), 1.0)
    ):
        failures.append("websocket probe audio is not non-silent and content-bound")
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
        or recovery.get("loaded_models") != report.get("loaded_models")
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
    if not _integer_at_least(min_commits, 1):
        failures.append("websocket report must require at least one commit per session")
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


def _dependency_lock_entries(
    snapshot: _EvidenceSnapshot,
) -> dict[str, tuple[str, bool]]:
    """Parse pinned requirements without appending production failures."""

    if snapshot.content is None:
        raise ValueError("dependency lock exceeds its capture limit")
    text = snapshot.content.decode("utf-8")
    logical_lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if pending:
            if not line.startswith("--hash=sha256:"):
                raise ValueError("dependency lock continuation is invalid")
            pending += " " + line
        else:
            pending = line
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        raise ValueError("dependency lock ends with an incomplete continuation")

    requirements: dict[str, tuple[str, bool]] = {}
    for line in logical_lines:
        if line.startswith(
            ("--extra-index-url ", "--find-links ", "--index-url ", "--no-binary ", "--only-binary ")
        ):
            continue
        if line.startswith(("-e ", "--editable ")) or any(
            token in line for token in (" @ ", "file:", "git+", "../")
        ):
            raise ValueError("dependency lock contains a mutable or local requirement")
        match = _LOCK_REQUIREMENT_PATTERN.match(line)
        if match is None:
            raise ValueError("dependency lock contains a requirement without an exact version")
        name = _package_name(match.group(1))
        version = match.group(2)
        conditional = ";" in line.split("--hash=", 1)[0]
        if conditional:
            raise ValueError(
                "dependency lock must be resolved for the production target "
                "and contain no environment markers"
            )
        previous = requirements.get(name)
        if previous is not None and previous[0] != version:
            raise ValueError(f"dependency lock contains conflicting versions for {name}")
        requirements[name] = (version, conditional)
    return requirements


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
    for line in logical_lines:
        if line.startswith("--trusted-host"):
            failures.append("dependency lock disables TLS verification with --trusted-host")
            continue
        if line.startswith(("--no-binary ", "--only-binary ")):
            continue
        index_directive = False
        for directive in ("--index-url ", "--extra-index-url "):
            if line.startswith(directive):
                raw_url = line.removeprefix(directive).strip()
                if urlsplit(raw_url).scheme != "https":
                    failures.append(
                        f"dependency lock {directive.strip()} must use HTTPS"
                    )
                index_directive = True
                break
        if index_directive:
            continue
        if line.startswith("--find-links "):
            raw_url = line.removeprefix("--find-links ").strip()
            if urlsplit(raw_url).scheme not in {"https", "file"}:
                failures.append(
                    "dependency lock --find-links must use HTTPS or a retained "
                    "immutable file directory"
                )
            elif urlsplit(raw_url).scheme == "file":
                failures.append(
                    "dependency lock must not use mutable local --find-links"
                )
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
        if conditional:
            failures.append(
                "dependency lock must be resolved for the production target "
                "and contain no environment markers"
            )
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
    expected_model_id: object,
    expected_loaded_models: object,
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
        or payload.get("schema_version") != 2
    ):
        failures.append("model manifest has an unsupported schema version")
    model_id = payload.get("model_id")
    if not _valid_model_id(model_id):
        failures.append("model manifest does not identify a valid model")
    # Gate reports describe paths on the Linux production host.  Parse those
    # paths with POSIX semantics even when the verifier itself runs elsewhere.
    model_relative = (
        PurePosixPath(cast(str, model_id)) if isinstance(model_id, str) else None
    )
    if (
        model_relative is None
        or model_relative.is_absolute()
        or any(part in {"", ".", ".."} for part in model_relative.parts)
    ):
        failures.append("model manifest model_id is not a safe retained path identity")
        model_relative = None
    if model_id != expected_model_id:
        failures.append("model manifest model_id does not match the deployed model identity")
    if payload.get("model_revision") != model_revision:
        failures.append("model manifest revision does not match the gate reports")

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        failures.append("model manifest has no model files")
        return
    manifest_files: list[tuple[str, str, int]] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            failures.append("model manifest contains an invalid file entry")
            return
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        relative_path = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or path.strip() != path
            or len(path) > 4_096
            or relative_path is None
            or relative_path.is_absolute()
            or relative_path.as_posix() != path
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not _positive_integer(size)
        ):
            failures.append("model manifest contains an invalid file identity")
            return
        manifest_files.append((path, digest, cast(int, size)))
    if len({path for path, _digest, _size in manifest_files}) != len(manifest_files):
        failures.append("model manifest contains duplicate file paths")
        return

    manifest_contents = sorted(
        (digest, size) for _path, digest, size in manifest_files
    )
    actual_contents = sorted((item.sha256, item.bytes) for item in artifacts)
    if manifest_contents != actual_contents:
        failures.append("model manifest does not match the retained model artifacts")
    if not isinstance(expected_loaded_models, list) or not expected_loaded_models:
        failures.append("model manifest is not bound to loaded runtime model evidence")
        return
    expected_model_root = (
        PurePosixPath("/var/lib/turnalign/models") / model_relative
        if model_relative is not None
        else None
    )
    loaded_files: list[tuple[str, object, object]] = []
    loaded_paths_valid = expected_model_root is not None
    for item in expected_loaded_models:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            loaded_paths_valid = False
            continue
        raw_loaded_path = cast(str, item["path"])
        loaded_path = PurePosixPath(raw_loaded_path)
        if (
            not loaded_path.is_absolute()
            or loaded_path.as_posix() != raw_loaded_path
            or any(part in {"", ".", ".."} for part in loaded_path.parts)
            or expected_model_root is None
            or loaded_path == expected_model_root
            or not loaded_path.is_relative_to(expected_model_root)
        ):
            loaded_paths_valid = False
            continue
        loaded_files.append((
            loaded_path.relative_to(expected_model_root).as_posix(),
            item.get("sha256"),
            item.get("bytes"),
        ))
    if (
        not loaded_paths_valid
        or len(loaded_files) != len(expected_loaded_models)
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes"}
            or not isinstance(item.get("path"), str)
            or not item["path"].startswith("/var/lib/turnalign/models/")
            or not isinstance(item.get("sha256"), str)
            or _SHA256_PATTERN.fullmatch(item["sha256"]) is None
            or not _positive_integer(item.get("bytes"))
            for item in expected_loaded_models
        )
        or sorted(loaded_files) != sorted(manifest_files)
    ):
        failures.append(
            "loaded runtime model evidence does not exactly match the retained model manifest"
        )


def _validate_wheel(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    failures: list[str],
) -> _WheelIdentity | None:
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
    if package_version is None:
        return None
    package_files = tuple(
        EvidenceFile(
            name=name,
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
        )
        for name, content in sorted(contents.items())
        if name.startswith("turnalign/")
    )
    if not package_files:
        failures.append("wheel does not contain any TurnAlign package files")
        return None
    return _WheelIdentity(package_version, package_files)


def _validate_host_profile(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    wheel_identity: _WheelIdentity | None,
    artifacts: list[ArtifactEvidence],
    failures: list[str],
) -> _HostProfileIdentity | None:
    if snapshot.content is None:
        failures.append(f"host profile exceeds {_MAX_HOST_PROFILE_BYTES} bytes")
        return None
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"host profile is not strict JSON: {error}")
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "source_commit",
        "active_commit",
        "runtime",
        "installed_distribution",
        "installed_dependencies",
        "platform",
        "effective_configuration",
        "artifacts",
    }:
        failures.append("host profile has an invalid top-level schema")
        return None
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 7
    ):
        failures.append("host profile has an unsupported schema version")
    if payload.get("source_commit") != source_commit:
        failures.append("host profile is not bound to the production source commit")
    if payload.get("active_commit") != source_commit:
        failures.append("host profile did not capture the active candidate release")

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
        and wheel_identity is not None
        and runtime["turnalign_version"] == wheel_identity.version
    ):
        failures.append(
            "host profile is not bound to the installed versioned Wheel runtime"
        )

    platform_data = payload.get("platform")
    installed_distribution = payload.get("installed_distribution")
    installed_file_payload = (
        installed_distribution.get("files")
        if isinstance(installed_distribution, dict)
        else None
    )
    expected_distribution_root: str | None = None
    if isinstance(platform_data, dict):
        python_version = platform_data.get("python_version")
        if isinstance(python_version, str):
            version_parts = python_version.split(".")
            if (
                len(version_parts) == 3
                and all(part.isdigit() for part in version_parts)
            ):
                expected_distribution_root = (
                    f"{expected_prefix}/lib/python{version_parts[0]}."
                    f"{version_parts[1]}/site-packages"
                )
    installed_files: list[EvidenceFile] = []
    installed_schema_valid = (
        isinstance(installed_distribution, dict)
        and set(installed_distribution) == {"name", "version", "root", "files"}
        and installed_distribution.get("name") == "turnalign"
        and wheel_identity is not None
        and installed_distribution.get("version") == wheel_identity.version
        and expected_distribution_root is not None
        and installed_distribution.get("root") == expected_distribution_root
        and isinstance(installed_file_payload, list)
        and bool(installed_file_payload)
    )
    if installed_schema_valid and isinstance(installed_file_payload, list):
        for item in installed_file_payload:
            if not (
                isinstance(item, dict)
                and set(item) == {"name", "sha256", "bytes"}
                and isinstance(item.get("name"), str)
                and item["name"].startswith("turnalign/")
                and _safe_relative_archive_path(item["name"])
                and isinstance(item.get("sha256"), str)
                and _SHA256_PATTERN.fullmatch(item["sha256"]) is not None
                and _nonnegative_integer(item.get("bytes"))
            ):
                installed_schema_valid = False
                break
            installed_files.append(
                EvidenceFile(
                    cast(str, item["name"]),
                    cast(str, item["sha256"]),
                    cast(int, item["bytes"]),
                )
            )
    if (
        not installed_schema_valid
        or len({item.name for item in installed_files}) != len(installed_files)
        or wheel_identity is None
        or tuple(sorted(installed_files, key=lambda item: item.name))
        != wheel_identity.package_files
    ):
        failures.append(
            "host profile installed package files do not exactly match the retained Wheel"
        )

    installed_dependencies = payload.get("installed_dependencies")
    installed_dependency_versions: dict[str, str] = {}
    if not isinstance(installed_dependencies, dict) or not installed_dependencies:
        failures.append("host profile has no installed runtime dependency evidence")
    else:
        for name, item in installed_dependencies.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(item, dict)
                or set(item)
                != {"name", "version", "root", "file_count", "bytes", "sha256"}
                or item.get("name") != name
                or not isinstance(item.get("version"), str)
                or not item.get("version")
                or expected_distribution_root is None
                or item.get("root") != expected_distribution_root
                or not _positive_integer(item.get("file_count"))
                or not _positive_integer(item.get("bytes"))
                or not isinstance(item.get("sha256"), str)
                or _SHA256_PATTERN.fullmatch(item["sha256"]) is None
            ):
                failures.append("host profile has invalid installed dependency evidence")
                installed_dependency_versions = {}
                break
            installed_dependency_versions[_package_name(name)] = cast(
                str, item["version"]
            )

    text_fields = (
        "system",
        "release",
        "machine",
        "python_implementation",
        "python_version",
    )
    if not (
        isinstance(platform_data, dict)
        and set(platform_data) == {*text_fields, "boot_id", "logical_cpu_count"}
        and platform_data.get("system") == "Linux"
        and isinstance(platform_data.get("boot_id"), str)
        and _BOOT_ID_PATTERN.fullmatch(platform_data["boot_id"]) is not None
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
        return None
    reported: list[tuple[str, str, str, int]] = []
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "kind",
            "name",
            "sha256",
            "bytes",
        }:
            failures.append("host profile contains an invalid artifact entry")
            return None
        kind = item.get("kind")
        name = item.get("name")
        digest = item.get("sha256")
        size = item.get("bytes")
        valid_name = (
            isinstance(name, str)
            and bool(name)
            and (
                Path(name).name == name
                if kind != "model"
                else (
                    not PurePosixPath(name).is_absolute()
                    and PurePosixPath(name).as_posix() == name
                    and all(part not in {"", ".", ".."} for part in PurePosixPath(name).parts)
                )
            )
        )
        if (
            not isinstance(kind, str)
            or kind not in REQUIRED_ARTIFACT_KINDS - {"host-profile"}
            or not valid_name
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not _positive_integer(size)
        ):
            failures.append("host profile contains an invalid artifact identity")
            return None
        if kind == "service-unit" and name != "turnalign.service":
            failures.append("host profile service-unit is not the canonical unit")
            return None
        if kind == "nginx-config" and name != "turnalign.conf":
            failures.append("host profile nginx-config is not the canonical config")
            return None
        reported.append((kind, cast(str, name), cast(str, digest), cast(int, size)))
    if len({(kind, name) for kind, name, _digest, _size in reported}) != len(
        reported
    ):
        failures.append("host profile contains duplicate artifact identities")
        return None
    actual = [
        (item.kind, item.name, item.sha256, item.bytes)
        for item in artifacts
        if item.kind != "host-profile"
    ]
    if sorted(reported) != sorted(actual):
        failures.append("host profile does not match the retained deployment artifacts")
    _validate_effective_configuration(
        "host profile",
        payload.get("effective_configuration"),
        artifacts,
        failures,
    )
    if (
        isinstance(platform_data, dict)
        and isinstance(platform_data.get("boot_id"), str)
        and _BOOT_ID_PATTERN.fullmatch(platform_data["boot_id"]) is not None
    ):
        return _HostProfileIdentity(
            boot_id=cast(str, platform_data["boot_id"]),
            installed_dependencies=installed_dependency_versions,
        )
    return None


def _validate_effective_configuration(
    label: str,
    payload: object,
    artifacts: list[ArtifactEvidence],
    failures: list[str],
) -> None:
    service_artifacts = [item for item in artifacts if item.kind == "service-unit"]
    nginx_artifacts = [item for item in artifacts if item.kind == "nginx-config"]
    if len(service_artifacts) != 1 or len(nginx_artifacts) != 1:
        failures.append(f"{label} cannot bind effective production configuration")
        return
    if not isinstance(payload, dict) or set(payload) != {"systemd", "nginx"}:
        failures.append(f"{label} has invalid effective configuration evidence")
        return
    systemd = payload.get("systemd")
    service = service_artifacts[0]
    if not (
        isinstance(systemd, dict)
        and set(systemd)
        == {
            "fragment_path",
            "drop_in_paths",
            "need_daemon_reload",
            "active_state",
            "sub_state",
            "sha256",
            "bytes",
        }
        and systemd.get("fragment_path") == _SERVICE_UNIT_ID
        and systemd.get("drop_in_paths") == []
        and systemd.get("need_daemon_reload") is False
        and systemd.get("active_state") == "active"
        and systemd.get("sub_state") == "running"
        and systemd.get("sha256") == service.sha256
        and systemd.get("bytes") == service.bytes
    ):
        failures.append(
            f"{label} does not prove the canonical active systemd unit without drop-ins"
        )
    nginx = payload.get("nginx")
    nginx_artifact = nginx_artifacts[0]
    if not (
        isinstance(nginx, dict)
        and set(nginx)
        == {
            "configuration_path",
            "loaded_occurrences",
            "warning_free",
            "sha256",
            "bytes",
        }
        and nginx.get("configuration_path") == _NGINX_CONFIG_ID
        and nginx.get("loaded_occurrences") == 1
        and nginx.get("warning_free") is True
        and nginx.get("sha256") == nginx_artifact.sha256
        and nginx.get("bytes") == nginx_artifact.bytes
    ):
        failures.append(
            f"{label} does not prove Nginx loaded the canonical configuration once"
        )


def _utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _validate_rehearsal_restart(
    payload: object,
    label: str,
    failures: list[str],
) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "restart_exit_code",
        "active_exit_code",
        "seconds",
        "failure",
    }:
        failures.append(f"{label} has invalid restart evidence")
        return
    if (
        payload.get("restart_exit_code") != 0
        or isinstance(payload.get("restart_exit_code"), bool)
        or payload.get("active_exit_code") != 0
        or isinstance(payload.get("active_exit_code"), bool)
        or not _nonnegative_number(payload.get("seconds"))
        or payload.get("failure") is not None
    ):
        failures.append(f"{label} did not restart an active service")


def _validate_rehearsal_readiness(
    payload: object,
    label: str,
    ready_uri: str,
    failures: list[str],
) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "uri",
        "status_code",
        "ready",
        "preloaded",
        "attempts",
        "seconds",
        "failure",
        "loaded_models",
    }:
        failures.append(f"{label} has invalid readiness evidence")
        return
    if (
        payload.get("uri") != ready_uri
        or payload.get("status_code") != 200
        or isinstance(payload.get("status_code"), bool)
        or payload.get("ready") is not True
        or payload.get("preloaded") is not True
        or not _positive_integer(payload.get("attempts"))
        or not _nonnegative_number(payload.get("seconds"))
        or payload.get("failure") is not None
    ):
        failures.append(
            f"{label} did not prove preloaded readiness"
        )
    loaded_models = payload.get("loaded_models")
    if (
        not isinstance(loaded_models, list)
        or not loaded_models
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes"}
            or not isinstance(item.get("path"), str)
            or not item["path"].startswith("/var/lib/turnalign/models/")
            or not isinstance(item.get("sha256"), str)
            or _SHA256_PATTERN.fullmatch(item["sha256"]) is None
            or not _positive_integer(item.get("bytes"))
            for item in loaded_models
        )
    ):
        failures.append(f"{label} did not retain loaded model evidence")


def _validate_rehearsal_phase(
    payload: object,
    *,
    phase: str,
    from_commit: str,
    target_commit: str,
    ready_uri: str,
    websocket_uri: str,
    release_root: str,
    failures: list[str],
    report_label: str = "rollback rehearsal",
) -> tuple[datetime | None, datetime | None, dict[str, object] | None]:
    label = f"{report_label} {phase}"
    if not isinstance(payload, dict) or set(payload) != {
        "name",
        "status",
        "from_commit",
        "target_commit",
        "target_path",
        "started_at",
        "activated_at",
        "completed_at",
        "activation_seconds",
        "restart",
        "readiness",
        "websocket_report",
        "failures",
    }:
        failures.append(f"{label} has an invalid phase schema")
        return None, None, None
    _passed_report(label, payload, failures)
    if (
        payload.get("name") != phase
        or payload.get("from_commit") != from_commit
        or payload.get("target_commit") != target_commit
        or payload.get("target_path") != f"{release_root}/{target_commit}"
    ):
        failures.append(f"{label} has an invalid transition")
    started = _utc_timestamp(payload.get("started_at"))
    activated = _utc_timestamp(payload.get("activated_at"))
    completed = _utc_timestamp(payload.get("completed_at"))
    if (
        started is None
        or activated is None
        or completed is None
        or not started <= activated <= completed
        or not _nonnegative_number(payload.get("activation_seconds"))
    ):
        failures.append(f"{label} has invalid activation timing")
    _validate_rehearsal_restart(payload.get("restart"), label, failures)
    _validate_rehearsal_readiness(
        payload.get("readiness"),
        label,
        ready_uri,
        failures,
    )
    websocket = payload.get("websocket_report")
    if not isinstance(websocket, dict):
        failures.append(f"{label} has no WebSocket gate report")
        websocket = None
    else:
        phase_failures: list[str] = []
        _validate_websocket(websocket, phase_failures)
        _validate_report_source(phase, websocket, target_commit, phase_failures)
        if websocket.get("uri") != websocket_uri:
            phase_failures.append("WebSocket report URI changed")
        failures.extend(
            f"{label}: {failure}"
            for failure in phase_failures
        )
    return started, completed, websocket


def _validate_deployment_state(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    artifacts: list[ArtifactEvidence],
    failures: list[str],
) -> _DeploymentStateIdentity | None:
    if snapshot.content is None:
        failures.append("deployment state exceeds its capture limit")
        return None
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"deployment state is not strict JSON: {error}")
        return None
    expected_fields = {
        "schema_version",
        "active_commit",
        "pending_transaction_id",
        "boot_id",
        "effective_configuration",
        "created_at",
        "validity_seconds",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        failures.append("deployment state has an invalid top-level schema")
        return None
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
    ):
        failures.append("deployment state has an unsupported schema version")
    active_commit = payload.get("active_commit")
    if active_commit != source_commit:
        failures.append(
            "live deployment state does not identify the candidate as active"
        )
    if payload.get("pending_transaction_id") is not None:
        failures.append(
            "live deployment state still records a pending activation transaction"
        )
    boot_id = payload.get("boot_id")
    if not isinstance(boot_id, str) or _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        failures.append("deployment state has no valid Linux boot identity")
        boot_id = None
    _validate_report_freshness(
        "deployment state",
        payload,
        failures,
        max_validity_seconds=_MAX_DEPLOYMENT_STATE_VALIDITY_SECONDS,
    )
    _validate_effective_configuration(
        "deployment state",
        payload.get("effective_configuration"),
        artifacts,
        failures,
    )
    if isinstance(boot_id, str):
        return _DeploymentStateIdentity(boot_id, str(active_commit), None)
    return None


def _validate_deployment_activation(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    candidate_websocket: dict[str, object],
    failures: list[str],
) -> _DeploymentIdentity | None:
    if snapshot.content is None:
        failures.append(
            "deployment activation exceeds "
            f"{_MAX_DEPLOYMENT_ACTIVATION_BYTES} bytes"
        )
        return None
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"deployment activation is not strict JSON: {error}")
        return None
    expected_fields = {
        "schema_version",
        "status",
        "candidate_commit",
        "previous_commit",
        "boot_id",
        "release_root",
        "current_link",
        "lock_path",
        "service",
        "systemctl",
        "ready_uri",
        "websocket_uri",
        "started_at",
        "completed_at",
        "initial_active_commit",
        "final_active_commit",
        "transaction_id",
        "transaction_path",
        "activation",
        "rollback",
        "failures",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        failures.append("deployment activation has an invalid top-level schema")
        return None
    _passed_report("deployment activation", payload, failures)
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
    ):
        failures.append("deployment activation has an unsupported schema version")
    transaction_id = payload.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None
        or payload.get("transaction_path")
        != "/var/lib/turnalign-deployment/pending-activation.json"
    ):
        failures.append("deployment activation has invalid transaction identity")
    previous_commit = payload.get("previous_commit")
    if (
        payload.get("candidate_commit") != source_commit
        or payload.get("final_active_commit") != source_commit
        or not isinstance(previous_commit, str)
        or _COMMIT_PATTERN.fullmatch(previous_commit) is None
        or previous_commit == source_commit
        or payload.get("initial_active_commit") != previous_commit
    ):
        failures.append(
            "deployment activation is not bound to one prior and candidate release"
        )
        previous_commit = ""
    boot_id = payload.get("boot_id")
    if not isinstance(boot_id, str) or _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        failures.append("deployment activation has no valid Linux boot identity")
        boot_id = None
    if (
        payload.get("release_root") != "/opt/turnalign/releases"
        or payload.get("current_link") != "/opt/turnalign/current"
        or payload.get("lock_path") != "/run/lock/turnalign-deployment.lock"
        or payload.get("service") != "turnalign.service"
        or payload.get("systemctl") != "/usr/bin/systemctl"
        or payload.get("ready_uri") != "http://127.0.0.1:8765/readyz"
    ):
        failures.append("deployment activation did not use the production layout")
    websocket_uri = candidate_websocket.get("uri")
    if not isinstance(websocket_uri, str) or payload.get("websocket_uri") != websocket_uri:
        failures.append("deployment activation did not probe the production WebSocket URI")
        websocket_uri = ""
    started = _utc_timestamp(payload.get("started_at"))
    completed = _utc_timestamp(payload.get("completed_at"))
    if started is None or completed is None or started > completed:
        failures.append("deployment activation has invalid overall timing")
    activation_started, activation_completed, deployed_websocket = (
        _validate_rehearsal_phase(
            payload.get("activation"),
            phase="activate",
            from_commit=previous_commit,
            target_commit=source_commit,
            ready_uri="http://127.0.0.1:8765/readyz",
            websocket_uri=websocket_uri,
            release_root="/opt/turnalign/releases",
            failures=failures,
            report_label="deployment activation",
        )
    )
    if payload.get("rollback") is not None:
        failures.append("passed deployment activation unexpectedly contains rollback evidence")
    if (
        started is not None
        and completed is not None
        and activation_started is not None
        and activation_completed is not None
        and not (
            started
            <= activation_started
            <= activation_completed
            <= completed
        )
    ):
        failures.append("deployment activation timestamps are out of order")
    if deployed_websocket is not None:
        identity_fields = (
            "backend",
            "backend_implementation",
            "model",
            "model_revision",
            "device",
            "language",
            "compute_type",
        )
        if any(
            deployed_websocket.get(field) != candidate_websocket.get(field)
            for field in identity_fields
        ):
            failures.append("deployment activation selected a different model identity")
    if isinstance(boot_id, str) and previous_commit:
        return _DeploymentIdentity(boot_id, previous_commit)
    return None


def _validate_rollback_rehearsal(
    snapshot: _EvidenceSnapshot,
    source_commit: str,
    candidate_websocket: dict[str, object],
    failures: list[str],
) -> _DeploymentIdentity | None:
    if snapshot.content is None:
        failures.append(
            "rollback rehearsal exceeds "
            f"{_MAX_ROLLBACK_REHEARSAL_BYTES} bytes"
        )
        return None
    try:
        payload = strict_json_loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"rollback rehearsal is not strict JSON: {error}")
        return None
    expected_fields = {
        "schema_version",
        "status",
        "candidate_commit",
        "previous_commit",
        "boot_id",
        "release_root",
        "current_link",
        "lock_path",
        "service",
        "systemctl",
        "ready_uri",
        "websocket_uri",
        "started_at",
        "completed_at",
        "initial_active_commit",
        "final_active_commit",
        "transaction_id",
        "transaction_path",
        "rollback",
        "restore",
        "failures",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        failures.append("rollback rehearsal has an invalid top-level schema")
        return None
    _passed_report("rollback rehearsal", payload, failures)
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
    ):
        failures.append("rollback rehearsal has an unsupported schema version")
    transaction_id = payload.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None
        or payload.get("transaction_path")
        != "/var/lib/turnalign-deployment/pending-activation.json"
    ):
        failures.append("rollback rehearsal has invalid transaction identity")
    previous_commit = payload.get("previous_commit")
    if (
        payload.get("candidate_commit") != source_commit
        or payload.get("initial_active_commit") != source_commit
        or payload.get("final_active_commit") != source_commit
        or not isinstance(previous_commit, str)
        or _COMMIT_PATTERN.fullmatch(previous_commit) is None
        or previous_commit == source_commit
    ):
        failures.append("rollback rehearsal is not bound to one prior and candidate release")
        previous_commit = ""
    boot_id = payload.get("boot_id")
    if not isinstance(boot_id, str) or _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        failures.append("rollback rehearsal has no valid Linux boot identity")
        boot_id = None
    if (
        payload.get("release_root") != "/opt/turnalign/releases"
        or payload.get("current_link") != "/opt/turnalign/current"
        or payload.get("lock_path") != "/run/lock/turnalign-deployment.lock"
        or payload.get("service") != "turnalign.service"
        or payload.get("systemctl") != "/usr/bin/systemctl"
        or payload.get("ready_uri") != "http://127.0.0.1:8765/readyz"
    ):
        failures.append("rollback rehearsal did not use the production activation layout")
    websocket_uri = candidate_websocket.get("uri")
    if not isinstance(websocket_uri, str) or payload.get("websocket_uri") != websocket_uri:
        failures.append("rollback rehearsal did not probe the production WebSocket URI")
        websocket_uri = ""
    started = _utc_timestamp(payload.get("started_at"))
    completed = _utc_timestamp(payload.get("completed_at"))
    if started is None or completed is None or started > completed:
        failures.append("rollback rehearsal has invalid overall timing")
    rollback_started, rollback_completed, _rollback_websocket = (
        _validate_rehearsal_phase(
            payload.get("rollback"),
            phase="rollback",
            from_commit=source_commit,
            target_commit=previous_commit,
            ready_uri="http://127.0.0.1:8765/readyz",
            websocket_uri=websocket_uri,
            release_root="/opt/turnalign/releases",
            failures=failures,
        )
    )
    restore_started, restore_completed, restored_websocket = (
        _validate_rehearsal_phase(
            payload.get("restore"),
            phase="restore",
            from_commit=previous_commit,
            target_commit=source_commit,
            ready_uri="http://127.0.0.1:8765/readyz",
            websocket_uri=websocket_uri,
            release_root="/opt/turnalign/releases",
            failures=failures,
        )
    )
    if (
        started is not None
        and completed is not None
        and rollback_started is not None
        and rollback_completed is not None
        and restore_started is not None
        and restore_completed is not None
        and not (
            started
            <= rollback_started
            <= rollback_completed
            <= restore_started
            <= restore_completed
            <= completed
        )
    ):
        failures.append("rollback rehearsal phase timestamps are out of order")
    if restored_websocket is not None:
        identity_fields = (
            "backend",
            "backend_implementation",
            "model",
            "model_revision",
            "device",
            "language",
            "compute_type",
        )
        if any(
            restored_websocket.get(field) != candidate_websocket.get(field)
            for field in identity_fields
        ):
            failures.append(
                "rollback rehearsal restored a different deployed model identity"
            )
    if isinstance(boot_id, str) and previous_commit:
        return _DeploymentIdentity(boot_id, previous_commit)
    return None


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
    model_root = _model_artifact_root(
        [path for kind, path in artifacts if kind == "model"]
    )
    for kind, path in artifacts:
        if kind not in REQUIRED_ARTIFACT_KINDS:
            raise ValueError(f"unsupported artifact kind: {kind}")
        capture_limit = {
            "deployment-activation": _MAX_DEPLOYMENT_ACTIVATION_BYTES,
            "deployment-state": _MAX_DEPLOYMENT_CONFIG_BYTES,
            "dependency-lock": _MAX_LOCK_BYTES,
            "host-profile": _MAX_HOST_PROFILE_BYTES,
            "model-manifest": _MAX_MODEL_MANIFEST_BYTES,
            "nginx-config": _MAX_DEPLOYMENT_CONFIG_BYTES,
            "rollback-rehearsal": _MAX_ROLLBACK_REHEARSAL_BYTES,
            "sbom": _MAX_SBOM_BYTES,
            "wheel": _MAX_WHEEL_BYTES,
            "service-unit": _MAX_DEPLOYMENT_CONFIG_BYTES,
            "websocket-probe-audio": 16 * 1024 * 1024,
        }.get(kind)
        snapshot = _snapshot_evidence(path, capture_limit=capture_limit)
        evidence = ArtifactEvidence(
            kind,
            _artifact_identity_name(kind, path, model_root),
            snapshot.sha256,
            snapshot.size,
        )
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

    wheel_identity: _WheelIdentity | None = None
    wheel_snapshots = artifact_snapshots.get("wheel", [])
    if len(wheel_snapshots) == 1:
        wheel_identity = _validate_wheel(wheel_snapshots[0], source_commit, failures)

    model_manifest_snapshots = artifact_snapshots.get("model-manifest", [])
    if len(model_manifest_snapshots) == 1:
        _validate_model_manifest(
            model_manifest_snapshots[0],
            release.get("model_revision"),
            release.get("model"),
            release.get("loaded_models"),
            evidence_by_kind.get("model", []),
            failures,
        )
    host_identity: _HostProfileIdentity | None = None
    host_profile_snapshots = artifact_snapshots.get("host-profile", [])
    if len(host_profile_snapshots) == 1:
        host_identity = _validate_host_profile(
            host_profile_snapshots[0],
            source_commit,
            wheel_identity,
            artifact_evidence,
            failures,
        )
    for name, (version, conditional) in locked_requirements.items():
        if conditional:
            continue
        if (
            host_identity is None
            or host_identity.installed_dependencies.get(name) != version
        ):
            failures.append(
                f"installed runtime dependency does not match the lock: {name}=={version}"
            )
    deployment_state_identity: _DeploymentStateIdentity | None = None
    deployment_state_snapshots = artifact_snapshots.get("deployment-state", [])
    if len(deployment_state_snapshots) == 1:
        deployment_state_identity = _validate_deployment_state(
            deployment_state_snapshots[0],
            source_commit,
            artifact_evidence,
            failures,
        )
    activation_identity: _DeploymentIdentity | None = None
    activation_snapshots = artifact_snapshots.get("deployment-activation", [])
    if len(activation_snapshots) == 1:
        activation_identity = _validate_deployment_activation(
            activation_snapshots[0],
            source_commit,
            websocket,
            failures,
        )
    rehearsal_identity: _DeploymentIdentity | None = None
    rehearsal_snapshots = artifact_snapshots.get("rollback-rehearsal", [])
    if len(rehearsal_snapshots) == 1:
        rehearsal_identity = _validate_rollback_rehearsal(
            rehearsal_snapshots[0],
            source_commit,
            websocket,
            failures,
        )
    if (
        host_identity is not None
        and activation_identity is not None
        and host_identity.boot_id != activation_identity.boot_id
    ):
        failures.append(
            "host profile and deployment activation identify different Linux boots"
        )
    if (
        host_identity is not None
        and rehearsal_identity is not None
        and host_identity.boot_id != rehearsal_identity.boot_id
    ):
        failures.append(
            "host profile and rollback rehearsal identify different Linux boots"
        )
    if (
        activation_identity is not None
        and rehearsal_identity is not None
        and activation_identity != rehearsal_identity
    ):
        failures.append(
            "deployment activation and rollback rehearsal identify different "
            "release pairs or Linux boots"
        )
    for identity in (host_identity, activation_identity, rehearsal_identity):
        if (
            identity is not None
            and deployment_state_identity is not None
            and identity.boot_id != deployment_state_identity.boot_id
        ):
            failures.append(
                "live deployment state identifies a different Linux boot"
            )
            break
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
    for label, report in (
        ("release", release),
        ("quality", quality),
        ("websocket", websocket),
    ):
        if report.get("model") != release.get("model"):
            failures.append(f"{label} report identifies a different model")
    if websocket.get("backend_implementation") != release.get("backend"):
        failures.append("websocket and release reports identify different backends")
    if websocket.get("model_revision") != release.get("model_revision"):
        failures.append("websocket and release reports identify different model revisions")
    if release.get("loaded_models") != websocket.get("loaded_models"):
        failures.append(
            "release and websocket evidence identify different loaded model files"
        )
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
    websocket_probe_digest = websocket.get("probe_audio_sha256")
    websocket_probe_evidence = evidence_by_kind.get("websocket-probe-audio", [])
    if (
        not isinstance(websocket_probe_digest, str)
        or _SHA256_PATTERN.fullmatch(websocket_probe_digest) is None
    ):
        failures.append("websocket probe audio digest is missing or invalid")
    elif len(websocket_probe_evidence) == 1 and (
        websocket_probe_evidence[0].sha256 != websocket_probe_digest
    ):
        failures.append("websocket probe audio does not match its retained artifact")
    if len(websocket_probe_evidence) == 1 and (
        websocket.get("probe_audio_bytes") != websocket_probe_evidence[0].bytes
    ):
        failures.append("websocket probe audio size does not match its retained artifact")

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
