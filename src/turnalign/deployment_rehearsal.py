from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .jsonutil import strict_json_loads
from .production_gate import _installed_runtime_identity, _read_linux_boot_id
from .websocket_gate import run_websocket_gate

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SERVICE_PATTERN = re.compile(r"[A-Za-z0-9_.@-]{1,128}\.service")
_MAX_IDENTITY_BYTES = 65
_MAX_READY_BYTES = 8 * 1024
_MAX_FAILURE_CHARACTERS = 1_024
_MAX_TRANSACTION_BYTES = 16 * 1024
_DEPLOYMENT_LOCK_PATH = Path("/run/lock/turnalign-deployment.lock")
_DEPLOYMENT_TRANSACTION_PATH = Path(
    "/var/lib/turnalign-deployment/pending-activation.json"
)
_DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TRANSACTION_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


@dataclass(frozen=True, slots=True)
class RehearsalProbeConfig:
    sessions: int = 2
    audio_seconds: float = 10.0
    sample_rate: int = 16_000
    channels: int = 1
    frame_ms: int = 100
    timeout: float = 120.0
    min_commits: int = 0
    min_audio_acks: int = 1
    max_dropped_partials: int = 0
    max_backpressure_pauses: int = 0
    max_ready_seconds: float = 30.0
    max_total_seconds: float = 180.0
    recovery_resume_timeout: float = 10.0
    backend: str | None = None
    model: str | None = None
    language: str | None = None
    compute_type: str | None = None
    probe_audio: Path | None = None


@dataclass(frozen=True, slots=True)
class ServiceRestartEvidence:
    restart_exit_code: int | None
    active_exit_code: int | None
    seconds: float
    failure: str | None


@dataclass(frozen=True, slots=True)
class ReadinessEvidence:
    uri: str
    status_code: int | None
    ready: bool
    preloaded: bool
    attempts: int
    seconds: float
    failure: str | None
    loaded_models: tuple[dict[str, object], ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.status_code == 200
            and self.ready
            and self.preloaded
            and self.failure is None
        )


@dataclass(frozen=True, slots=True)
class RehearsalPhaseReport:
    name: str
    status: str
    from_commit: str | None
    target_commit: str
    target_path: str
    started_at: str
    activated_at: str | None
    completed_at: str
    activation_seconds: float | None
    restart: ServiceRestartEvidence | None
    readiness: ReadinessEvidence | None
    websocket_report: dict[str, object] | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.failures


@dataclass(frozen=True, slots=True)
class DeploymentRehearsalReport:
    schema_version: int
    status: str
    candidate_commit: str
    previous_commit: str
    boot_id: str
    release_root: str
    current_link: str
    lock_path: str
    service: str
    systemctl: str
    ready_uri: str
    websocket_uri: str
    started_at: str
    completed_at: str
    initial_active_commit: str
    final_active_commit: str | None
    transaction_id: str
    transaction_path: str
    rollback: RehearsalPhaseReport
    restore: RehearsalPhaseReport
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.failures

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeploymentActivationReport:
    schema_version: int
    status: str
    candidate_commit: str
    previous_commit: str
    boot_id: str
    release_root: str
    current_link: str
    lock_path: str
    service: str
    systemctl: str
    ready_uri: str
    websocket_uri: str
    started_at: str
    completed_at: str
    initial_active_commit: str
    final_active_commit: str | None
    transaction_id: str
    transaction_path: str
    activation: RehearsalPhaseReport
    rollback: RehearsalPhaseReport | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.failures

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PendingDeploymentTransaction:
    schema_version: int
    transaction_id: str
    operation: str
    previous_commit: str
    candidate_commit: str
    boot_id: str
    release_root: str
    current_link: str
    service: str
    systemctl: str
    ready_uri: str
    websocket_uri: str
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeploymentRecoveryReport:
    schema_version: int
    status: str
    transaction_id: str
    transaction_path: str
    operation: str
    recovery_commit: str
    candidate_commit: str
    previous_commit: str
    original_boot_id: str
    recovery_boot_id: str
    release_root: str
    current_link: str
    lock_path: str
    service: str
    systemctl: str
    ready_uri: str
    websocket_uri: str
    started_at: str
    completed_at: str
    initial_active_commit: str | None
    final_active_commit: str | None
    recovery: RehearsalPhaseReport
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed" and not self.failures

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _seconds(started: float) -> float:
    return round(max(0.0, time.monotonic() - started), 3)


def _failure_text(error: BaseException | str) -> str:
    if isinstance(error, str):
        message = error
    else:
        message = f"{type(error).__name__}: {error}"
    cleaned = "".join(
        character if ord(character) >= 32 and ord(character) != 127 else " "
        for character in message
    ).strip()
    return cleaned[:_MAX_FAILURE_CHARACTERS] or "operation failed"


def _validate_layout(release_root: Path, current_link: Path) -> None:
    if not release_root.is_absolute() or not current_link.is_absolute():
        raise ValueError("release root and current link must be absolute paths")
    if (
        Path(os.path.normpath(str(release_root))) != release_root
        or Path(os.path.normpath(str(current_link))) != current_link
    ):
        raise ValueError("deployment paths must be normalized")
    if current_link.parent != release_root.parent or current_link.name != "current":
        raise ValueError("current link must be the release-root sibling named current")


def _validate_ready_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        _ = parsed.port
    except ValueError as error:
        raise ValueError("ready URI must use a valid loopback IP address") from error
    if (
        parsed.scheme != "http"
        or not address.is_loopback
        or parsed.path != "/readyz"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("ready URI must be a credential-free loopback HTTP /readyz URL")


def _public_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_global
    except ValueError:
        normalized = hostname.rstrip(".").lower()
        return (
            len(normalized) <= 253
            and "." in normalized
            and normalized != "localhost"
            and not normalized.endswith((".local", ".localhost", ".internal"))
            and all(
                _DNS_LABEL_PATTERN.fullmatch(label) is not None
                for label in normalized.split(".")
            )
        )


def _validate_websocket_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("WebSocket URI contains an invalid port") from error
    if (
        parsed.scheme != "wss"
        or not _public_hostname(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "deployment operations require a credential-free public wss:// endpoint"
        )


def _parse_loaded_models(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise TypeError("readiness response has invalid loaded_models")
    entries = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("readiness response has an invalid loaded model entry")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(path, str)
            or not path.startswith("/var/lib/turnalign/models/")
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError("readiness response has an invalid loaded model entry")
        entries.append({"path": path, "sha256": digest, "bytes": size})
    return tuple(entries)


def _required_release_owner() -> int:
    return 0


def _release_entry_is_immutable(metadata: os.stat_result) -> bool:
    return (
        metadata.st_uid == _required_release_owner()
        and not stat.S_IMODE(metadata.st_mode) & 0o022
    )


def _validate_symlink_target(path: Path, release: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved_release = release.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"release contains an unresolved symbolic link: {path}") from error
    internal_target = resolved.is_relative_to(resolved_release)
    if resolved.is_dir() and not internal_target:
        raise ValueError(
            f"release symbolic link points to an external directory: {path}"
        )
    target_parents: list[Path] = []
    if internal_target and resolved != resolved_release:
        for parent in resolved.parents:
            target_parents.append(parent)
            if parent == resolved_release:
                break
    elif not internal_target:
        target_parents.extend(resolved.parents)
    candidates = [resolved, *target_parents]
    for candidate in reversed(candidates):
        try:
            metadata = os.lstat(candidate)
        except OSError as error:
            raise ValueError(
                f"cannot securely inspect release link target: {path}"
            ) from error
        if candidate == resolved:
            valid_type = stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(
                metadata.st_mode
            )
        else:
            valid_type = stat.S_ISDIR(metadata.st_mode)
        if (
            not valid_type
            or stat.S_ISLNK(metadata.st_mode)
            or not _release_entry_is_immutable(metadata)
        ):
            raise ValueError(
                f"release symbolic link target is unsafe or mutable: {path}"
            )


def _validate_release_tree(release: Path) -> None:
    def fail_walk(error: OSError) -> None:
        raise error

    try:
        for directory, directory_names, file_names in os.walk(
            release,
            topdown=True,
            onerror=fail_walk,
            followlinks=False,
        ):
            directory_path = Path(directory)
            for name in (*directory_names, *file_names):
                path = directory_path / name
                metadata = os.lstat(path)
                if stat.S_ISLNK(metadata.st_mode):
                    if metadata.st_uid != _required_release_owner():
                        raise ValueError(
                            f"release contains a non-root-owned symbolic link: {path}"
                        )
                    _validate_symlink_target(path, release)
                    continue
                if not (
                    stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISREG(metadata.st_mode)
                ) or not _release_entry_is_immutable(metadata):
                    raise ValueError(
                        f"release contains an unsafe or mutable entry: {path}"
                    )
    except OSError as error:
        raise ValueError("cannot securely inspect the complete release tree") from error


def _read_release_identity(path: Path) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
    except OSError as error:
        raise ValueError(f"cannot securely read release identity: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != _required_release_owner()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError("release identity must be root-owned and immutable")
        raw = os.read(descriptor, _MAX_IDENTITY_BYTES)
        if os.read(descriptor, 1):
            raise ValueError("release identity exceeds its expected size")
    finally:
        os.close(descriptor)
    try:
        identity = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("release identity must be ASCII") from error
    if not identity.endswith("\n") or _COMMIT_PATTERN.fullmatch(identity[:-1]) is None:
        raise ValueError("release identity is not a bound source commit")
    return identity[:-1]


def _validate_release_directory(release_root: Path, commit: str) -> Path:
    release = release_root / commit
    for label, path in (
        ("deployment root", release_root.parent),
        ("release root", release_root),
        ("release", release),
    ):
        try:
            metadata = os.lstat(path)
        except OSError as error:
            raise ValueError(f"{label} directory is unavailable: {path}") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or not _release_entry_is_immutable(metadata)
        ):
            raise ValueError(f"{label} must be a root-owned immutable directory")
    _validate_release_tree(release)
    identities = list(
        (release / "venv" / "lib").glob(
            "python*/site-packages/turnalign/_source_commit.txt"
        )
    )
    if len(identities) != 1 or _read_release_identity(identities[0]) != commit:
        raise ValueError(f"release directory is not bound to source commit {commit}")
    python = release / "venv" / "bin" / "python"
    try:
        python_metadata = os.stat(python)
    except OSError as error:
        raise ValueError(f"release has no executable Python runtime: {python}") from error
    if (
        not stat.S_ISREG(python_metadata.st_mode)
        or not _release_entry_is_immutable(python_metadata)
        or not stat.S_IMODE(python_metadata.st_mode) & 0o111
    ):
        raise ValueError(f"release has no executable Python runtime: {python}")
    return release


def _active_commit(release_root: Path, current_link: Path) -> str:
    try:
        metadata = os.lstat(current_link)
        raw_target = os.readlink(current_link)
    except OSError as error:
        raise ValueError("current release link is unavailable") from error
    if not stat.S_ISLNK(metadata.st_mode):
        raise ValueError("current release path must be a symbolic link")
    target = Path(raw_target)
    if not target.is_absolute():
        raise ValueError("current release link must use an absolute target")
    normalized = Path(os.path.normpath(raw_target))
    if normalized.parent != release_root:
        raise ValueError("current release link points outside the release root")
    commit = normalized.name
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("current release link target is not a source commit")
    if normalized != release_root / commit:
        raise ValueError("current release link target is not canonical")
    return commit


def _fsync_directory(path: Path) -> None:
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


def _required_transaction_owner() -> int:
    return 0


def _validate_transaction_directory(path: Path) -> None:
    parent = path.parent
    if not path.is_absolute() or Path(os.path.normpath(str(path))) != path:
        raise ValueError("deployment transaction path must be absolute and normalized")
    try:
        metadata = os.lstat(parent)
    except FileNotFoundError:
        try:
            os.mkdir(parent, 0o700)
        except OSError as error:
            raise RuntimeError(
                "cannot create the root-only deployment transaction directory"
            ) from error
        metadata = os.lstat(parent)
        _fsync_directory(parent.parent)
    except OSError as error:
        raise RuntimeError(
            "cannot inspect the deployment transaction directory"
        ) from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != _required_transaction_owner()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise RuntimeError(
            "deployment transaction directory must be a root-only regular directory"
        )


def _transaction_from_payload(payload: object) -> PendingDeploymentTransaction:
    common_fields = {
        "schema_version",
        "transaction_id",
        "previous_commit",
        "candidate_commit",
        "boot_id",
        "release_root",
        "current_link",
        "service",
        "systemctl",
        "ready_uri",
        "websocket_uri",
        "created_at",
    }
    if not isinstance(payload, dict):
        raise TypeError("deployment transaction has an invalid schema")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in {1, 2}:
        raise ValueError("deployment transaction has an unsupported schema version")
    expected_fields = (
        common_fields if schema_version == 1 else common_fields | {"operation"}
    )
    if set(payload) != expected_fields:
        raise ValueError("deployment transaction has an invalid schema")
    transaction_id = payload.get("transaction_id")
    operation = "activation" if schema_version == 1 else payload.get("operation")
    previous_commit = payload.get("previous_commit")
    candidate_commit = payload.get("candidate_commit")
    boot_id = payload.get("boot_id")
    string_fields = {
        field: payload.get(field)
        for field in (
            "release_root",
            "current_link",
            "service",
            "systemctl",
            "ready_uri",
            "websocket_uri",
            "created_at",
        )
    }
    if (
        not isinstance(transaction_id, str)
        or _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None
        or operation not in {"activation", "rehearsal"}
        or not isinstance(previous_commit, str)
        or _COMMIT_PATTERN.fullmatch(previous_commit) is None
        or not isinstance(candidate_commit, str)
        or _COMMIT_PATTERN.fullmatch(candidate_commit) is None
        or candidate_commit == previous_commit
        or not isinstance(boot_id, str)
        or _BOOT_ID_PATTERN.fullmatch(boot_id) is None
        or any(not isinstance(value, str) for value in string_fields.values())
    ):
        raise ValueError("deployment transaction contains invalid identities")
    release_root = Path(str(string_fields["release_root"]))
    current_link = Path(str(string_fields["current_link"]))
    _validate_layout(release_root, current_link)
    _validate_ready_uri(str(string_fields["ready_uri"]))
    _validate_websocket_uri(str(string_fields["websocket_uri"]))
    service = str(string_fields["service"])
    if _SERVICE_PATTERN.fullmatch(service) is None:
        raise ValueError("deployment transaction contains an invalid service")
    if string_fields["systemctl"] != "/usr/bin/systemctl":
        raise ValueError("deployment transaction contains an invalid systemctl path")
    try:
        created_at = datetime.fromisoformat(
            str(string_fields["created_at"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("deployment transaction has an invalid timestamp") from error
    if created_at.tzinfo is None:
        raise ValueError("deployment transaction timestamp must include a timezone")
    return PendingDeploymentTransaction(
        schema_version=2,
        transaction_id=transaction_id,
        operation=str(operation),
        previous_commit=previous_commit,
        candidate_commit=candidate_commit,
        boot_id=boot_id,
        release_root=str(release_root),
        current_link=str(current_link),
        service=service,
        systemctl=str(string_fields["systemctl"]),
        ready_uri=str(string_fields["ready_uri"]),
        websocket_uri=str(string_fields["websocket_uri"]),
        created_at=str(string_fields["created_at"]),
    )


def _read_pending_transaction() -> PendingDeploymentTransaction:
    path = _DEPLOYMENT_TRANSACTION_PATH
    _validate_transaction_directory(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
    except OSError as error:
        raise RuntimeError("no recoverable deployment transaction exists") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != _required_transaction_owner()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > _MAX_TRANSACTION_BYTES
        ):
            raise RuntimeError(
                "deployment transaction must be a bounded root-only regular file"
            )
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        chunks = []
        remaining = _MAX_TRANSACTION_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(remaining, 4 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        final_identity = (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        )
        if (
            len(raw) > _MAX_TRANSACTION_BYTES
            or len(raw) != metadata.st_size
            or final_identity != identity
        ):
            raise RuntimeError(
                "deployment transaction changed while being read or exceeds its limit"
            )
    finally:
        os.close(descriptor)
    try:
        payload = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("deployment transaction is not strict JSON") from error
    try:
        return _transaction_from_payload(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(str(error)) from error


def _write_pending_transaction(transaction: PendingDeploymentTransaction) -> None:
    path = _DEPLOYMENT_TRANSACTION_PATH
    _validate_transaction_directory(path)
    _reject_pending_transaction()
    payload = (
        json.dumps(
            transaction.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_TRANSACTION_BYTES:
        raise RuntimeError("deployment transaction exceeds its size limit")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
            0o600,
        )
    except OSError as error:
        raise RuntimeError("cannot create the deployment transaction") from error
    installed = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("deployment transaction write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        installed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    _fsync_directory(path.parent)


def _remove_pending_transaction(transaction_id: str) -> None:
    transaction = _read_pending_transaction()
    if transaction.transaction_id != transaction_id:
        raise RuntimeError("deployment transaction identity changed")
    try:
        _DEPLOYMENT_TRANSACTION_PATH.unlink()
    except OSError as error:
        raise RuntimeError("cannot remove the deployment transaction") from error
    _fsync_directory(_DEPLOYMENT_TRANSACTION_PATH.parent)


def _reject_pending_transaction() -> None:
    try:
        os.lstat(_DEPLOYMENT_TRANSACTION_PATH)
    except FileNotFoundError:
        return
    except OSError as error:
        raise RuntimeError("cannot inspect the deployment transaction") from error
    raise RuntimeError("a pending deployment transaction must be recovered first")


def _activate_release(release_root: Path, current_link: Path, commit: str) -> None:
    target = release_root / commit
    temporary = current_link.parent / f".current-{commit}-{secrets.token_hex(8)}"
    created = False
    try:
        os.symlink(str(target), temporary)
        created = True
        _fsync_directory(current_link.parent)
        os.replace(temporary, current_link)
        created = False
        _fsync_directory(current_link.parent)
    finally:
        if created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if _active_commit(release_root, current_link) != commit:
        raise RuntimeError("atomic release activation did not select its target")


def _command(
    arguments: list[str],
    *,
    timeout: float,
) -> tuple[int | None, str | None]:
    with tempfile.TemporaryFile() as diagnostics:
        try:
            completed = subprocess.run(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=diagnostics,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"command exceeded {timeout:.3f}s timeout"
        diagnostics.flush()
        diagnostics.seek(0, os.SEEK_END)
        length = diagnostics.tell()
        diagnostics.seek(max(0, length - 8 * 1024))
        tail = diagnostics.read(8 * 1024).decode("utf-8", errors="replace").strip()
    if completed.returncode == 0:
        return 0, None
    if length > 8 * 1024:
        tail = f"[earlier output truncated] {tail}"
    return completed.returncode, _failure_text(tail or "command failed")


def _restart_service(
    systemctl: Path,
    service: str,
    *,
    timeout: float,
) -> ServiceRestartEvidence:
    started = time.monotonic()
    restart_code, failure = _command(
        [str(systemctl), "restart", service],
        timeout=timeout,
    )
    active_code: int | None = None
    if restart_code == 0:
        active_code, active_failure = _command(
            [str(systemctl), "is-active", "--quiet", service],
            timeout=timeout,
        )
        failure = active_failure
    return ServiceRestartEvidence(
        restart_exit_code=restart_code,
        active_exit_code=active_code,
        seconds=_seconds(started),
        failure=failure,
    )


def _wait_readiness(
    uri: str,
    *,
    timeout: float,
    interval: float,
) -> ReadinessEvidence:
    started = time.monotonic()
    deadline = started + timeout
    attempts = 0
    status_code: int | None = None
    last_failure = "readiness deadline expired"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    while True:
        attempts += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        request = urllib.request.Request(
            uri,
            headers={"Connection": "close", "User-Agent": "turnalign-deployment/1"},
        )
        try:
            with opener.open(
                request,
                timeout=min(5.0, remaining),
            ) as response:
                status_code = response.status
                raw = response.read(_MAX_READY_BYTES + 1)
                if len(raw) > _MAX_READY_BYTES:
                    raise ValueError("readiness response exceeds 8 KiB")
                payload = strict_json_loads(raw.decode("utf-8"))
                if (
                    status_code == 200
                    and isinstance(payload, dict)
                    and payload.get("status") == "ok"
                    and payload.get("ready") is True
                    and payload.get("preloaded") is True
                ):
                    loaded_models = _parse_loaded_models(
                        payload.get("loaded_models")
                    )
                    if not loaded_models:
                        raise ValueError(
                            "readiness endpoint did not report loaded model evidence"
                        )
                    return ReadinessEvidence(
                        uri=uri,
                        status_code=200,
                        ready=True,
                        preloaded=True,
                        attempts=attempts,
                        seconds=_seconds(started),
                        failure=None,
                        loaded_models=loaded_models,
                    )
                last_failure = "readiness endpoint did not confirm preloaded readiness"
        except (
            OSError,
            UnicodeDecodeError,
            ValueError,
            urllib.error.URLError,
        ) as error:
            if isinstance(error, urllib.error.HTTPError):
                status_code = error.code
            last_failure = _failure_text(error)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    return ReadinessEvidence(
        uri=uri,
        status_code=status_code,
        ready=False,
        preloaded=False,
        attempts=attempts,
        seconds=_seconds(started),
        failure=last_failure,
        loaded_models=(),
    )


async def _exercise_phase(
    name: str,
    target_commit: str,
    *,
    release_root: Path,
    current_link: Path,
    systemctl: Path,
    service: str,
    ready_uri: str,
    websocket_uri: str,
    restart_timeout: float,
    readiness_timeout: float,
    readiness_interval: float,
    probe: RehearsalProbeConfig,
    auth_token: str | None,
) -> RehearsalPhaseReport:
    started_at = _utc_timestamp()
    from_commit: str | None = None
    activated_at: str | None = None
    activation_seconds: float | None = None
    restart: ServiceRestartEvidence | None = None
    readiness: ReadinessEvidence | None = None
    websocket_report: dict[str, object] | None = None
    failures: list[str] = []
    try:
        from_commit = _active_commit(release_root, current_link)
        activation_started = time.monotonic()
        _activate_release(release_root, current_link, target_commit)
        activation_seconds = _seconds(activation_started)
        activated_at = _utc_timestamp()
        restart = _restart_service(
            systemctl,
            service,
            timeout=restart_timeout,
        )
        if restart.failure is not None:
            failures.append(f"service restart failed: {restart.failure}")
        else:
            readiness = _wait_readiness(
                ready_uri,
                timeout=readiness_timeout,
                interval=readiness_interval,
            )
            if not readiness.passed:
                failures.append(
                    f"preloaded readiness failed: {readiness.failure or 'not ready'}"
                )
            else:
                report = await run_websocket_gate(
                    websocket_uri,
                    sessions=probe.sessions,
                    audio_seconds=probe.audio_seconds,
                    sample_rate=probe.sample_rate,
                    channels=probe.channels,
                    frame_ms=probe.frame_ms,
                    timeout=probe.timeout,
                    min_commits=probe.min_commits,
                    min_audio_acks=probe.min_audio_acks,
                    max_dropped_partials=probe.max_dropped_partials,
                    max_backpressure_pauses=probe.max_backpressure_pauses,
                    max_ready_seconds=probe.max_ready_seconds,
                    max_total_seconds=probe.max_total_seconds,
                    realtime=True,
                    backend=probe.backend,
                    model=probe.model,
                    language=probe.language,
                    compute_type=probe.compute_type,
                    auth_token=auth_token,
                    verify_recovery=True,
                    recovery_resume_timeout=probe.recovery_resume_timeout,
                    source_commit=target_commit,
                    probe_audio_path=probe.probe_audio,
                )
                websocket_report = report.to_dict()
                if not report.passed:
                    failures.append("public WebSocket gate did not pass")
    except BaseException as error:  # deployment boundary; restore still follows
        if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise
        failures.append(_failure_text(error))
    return RehearsalPhaseReport(
        name=name,
        status="failed" if failures else "passed",
        from_commit=from_commit,
        target_commit=target_commit,
        target_path=str(release_root / target_commit),
        started_at=started_at,
        activated_at=activated_at,
        completed_at=_utc_timestamp(),
        activation_seconds=activation_seconds,
        restart=restart,
        readiness=readiness,
        websocket_report=websocket_report,
        failures=tuple(failures),
    )


async def _run_deployment_rehearsal_locked(
    previous_commit: str,
    candidate_commit: str,
    websocket_uri: str,
    *,
    release_root: Path = Path("/opt/turnalign/releases"),
    current_link: Path = Path("/opt/turnalign/current"),
    service: str = "turnalign.service",
    systemctl: Path = Path("/usr/bin/systemctl"),
    ready_uri: str = "http://127.0.0.1:8765/readyz",
    restart_timeout: float = 120.0,
    readiness_timeout: float = 300.0,
    readiness_interval: float = 1.0,
    probe: RehearsalProbeConfig | None = None,
    auth_token: str | None = None,
) -> DeploymentRehearsalReport:
    """Atomically roll back, probe, restore, and probe one Linux release pair."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment rehearsal must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment rehearsal must run as root")
    for label, commit in (
        ("previous", previous_commit),
        ("candidate", candidate_commit),
    ):
        if _COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(f"{label} commit must be a lowercase 40-character hash")
    if previous_commit == candidate_commit:
        raise ValueError("previous and candidate commits must be different")
    _validate_layout(release_root, current_link)
    _validate_ready_uri(ready_uri)
    _validate_websocket_uri(websocket_uri)
    if _SERVICE_PATTERN.fullmatch(service) is None:
        raise ValueError("service must be a bounded systemd .service unit name")
    if systemctl != Path("/usr/bin/systemctl"):
        raise ValueError("deployment rehearsal requires /usr/bin/systemctl")
    for label, value in (
        ("restart_timeout", restart_timeout),
        ("readiness_timeout", readiness_timeout),
        ("readiness_interval", readiness_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{label} must be positive")
    selected_probe = probe or RehearsalProbeConfig()
    if selected_probe.backend is None or selected_probe.model is None:
        raise ValueError("deployment rehearsal requires an explicit backend and model")
    if selected_probe.probe_audio is None:
        raise ValueError("deployment rehearsal requires an explicit probe-audio WAV")
    if selected_probe.min_commits < 1:
        raise ValueError("deployment rehearsal requires at least one commit per session")

    runtime = _installed_runtime_identity(candidate_commit)
    if runtime["turnalign_source_commit"] != candidate_commit:
        raise RuntimeError("candidate runtime identity changed during preflight")
    _validate_release_directory(release_root, previous_commit)
    _validate_release_directory(release_root, candidate_commit)
    initial_active_commit = _active_commit(release_root, current_link)
    if initial_active_commit != candidate_commit:
        raise ValueError("candidate must be active before the rollback rehearsal")
    boot_id = _read_linux_boot_id()
    started_at = _utc_timestamp()
    transaction = PendingDeploymentTransaction(
        schema_version=2,
        transaction_id=secrets.token_hex(32),
        operation="rehearsal",
        previous_commit=previous_commit,
        candidate_commit=candidate_commit,
        boot_id=boot_id,
        release_root=str(release_root),
        current_link=str(current_link),
        service=service,
        systemctl=str(systemctl),
        ready_uri=ready_uri,
        websocket_uri=websocket_uri,
        created_at=started_at,
    )
    _write_pending_transaction(transaction)

    restore: RehearsalPhaseReport | None = None
    try:
        rollback = await _exercise_phase(
            "rollback",
            previous_commit,
            release_root=release_root,
            current_link=current_link,
            systemctl=systemctl,
            service=service,
            ready_uri=ready_uri,
            websocket_uri=websocket_uri,
            restart_timeout=restart_timeout,
            readiness_timeout=readiness_timeout,
            readiness_interval=readiness_interval,
            probe=selected_probe,
            auth_token=auth_token,
        )
    finally:
        try:
            # A cancellation or operator interrupt must still attempt to return
            # the service to the reviewed candidate before control leaves.
            restore = await _exercise_phase(
                "restore",
                candidate_commit,
                release_root=release_root,
                current_link=current_link,
                systemctl=systemctl,
                service=service,
                ready_uri=ready_uri,
                websocket_uri=websocket_uri,
                restart_timeout=restart_timeout,
                readiness_timeout=readiness_timeout,
                readiness_interval=readiness_interval,
                probe=selected_probe,
                auth_token=auth_token,
            )
        finally:
            # If the structured restore phase was interrupted before activation,
            # make one final synchronous switch-and-restart attempt. The command
            # still fails because this fallback has no complete probe evidence.
            try:
                candidate_active = (
                    _active_commit(release_root, current_link) == candidate_commit
                )
            except ValueError:
                candidate_active = False
            restart_incomplete = (
                restore is None
                or restore.restart is None
                or restore.restart.failure is not None
            )
            if not candidate_active or restart_incomplete:
                if not candidate_active:
                    _activate_release(release_root, current_link, candidate_commit)
                _restart_service(
                    systemctl,
                    service,
                    timeout=restart_timeout,
                )
    if restore is None:
        raise RuntimeError("candidate restore did not produce deployment evidence")
    failures: list[str] = []
    if not rollback.passed:
        failures.append("rollback phase did not pass")
    if rollback.from_commit != candidate_commit:
        failures.append("rollback phase did not start from the candidate")
    if not restore.passed:
        failures.append("candidate restore phase did not pass")
    if restore.from_commit != previous_commit:
        failures.append("restore phase did not start from the preceding release")
    try:
        final_active_commit = _active_commit(release_root, current_link)
    except ValueError as error:
        final_active_commit = None
        failures.append(_failure_text(error))
    if final_active_commit != candidate_commit:
        failures.append("candidate is not active after the rehearsal")
    try:
        if _read_linux_boot_id() != boot_id:
            failures.append("host rebooted during the deployment rehearsal")
    except RuntimeError as error:
        failures.append(_failure_text(error))
    return DeploymentRehearsalReport(
        schema_version=2,
        status="failed" if failures else "passed",
        candidate_commit=candidate_commit,
        previous_commit=previous_commit,
        boot_id=boot_id,
        release_root=str(release_root),
        current_link=str(current_link),
        lock_path=str(_DEPLOYMENT_LOCK_PATH),
        service=service,
        systemctl=str(systemctl),
        ready_uri=ready_uri,
        websocket_uri=websocket_uri,
        started_at=started_at,
        completed_at=_utc_timestamp(),
        initial_active_commit=initial_active_commit,
        final_active_commit=final_active_commit,
        transaction_id=transaction.transaction_id,
        transaction_path=str(_DEPLOYMENT_TRANSACTION_PATH),
        rollback=rollback,
        restore=restore,
        failures=tuple(failures),
    )


async def _run_deployment_activation_locked(
    previous_commit: str,
    candidate_commit: str,
    websocket_uri: str,
    *,
    release_root: Path = Path("/opt/turnalign/releases"),
    current_link: Path = Path("/opt/turnalign/current"),
    service: str = "turnalign.service",
    systemctl: Path = Path("/usr/bin/systemctl"),
    ready_uri: str = "http://127.0.0.1:8765/readyz",
    restart_timeout: float = 120.0,
    readiness_timeout: float = 300.0,
    readiness_interval: float = 1.0,
    probe: RehearsalProbeConfig | None = None,
    auth_token: str | None = None,
) -> DeploymentActivationReport:
    """Activate and probe a candidate, leaving its marker for report finalization."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment activation must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment activation must run as root")
    for label, commit in (
        ("previous", previous_commit),
        ("candidate", candidate_commit),
    ):
        if _COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(f"{label} commit must be a lowercase 40-character hash")
    if previous_commit == candidate_commit:
        raise ValueError("previous and candidate commits must be different")
    _validate_layout(release_root, current_link)
    _validate_ready_uri(ready_uri)
    _validate_websocket_uri(websocket_uri)
    if _SERVICE_PATTERN.fullmatch(service) is None:
        raise ValueError("service must be a bounded systemd .service unit name")
    if systemctl != Path("/usr/bin/systemctl"):
        raise ValueError("deployment activation requires /usr/bin/systemctl")
    for label, value in (
        ("restart_timeout", restart_timeout),
        ("readiness_timeout", readiness_timeout),
        ("readiness_interval", readiness_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{label} must be positive")
    selected_probe = probe or RehearsalProbeConfig()
    if selected_probe.backend is None or selected_probe.model is None:
        raise ValueError("deployment activation requires an explicit backend and model")
    if selected_probe.probe_audio is None:
        raise ValueError("deployment activation requires an explicit probe-audio WAV")
    if selected_probe.min_commits < 1:
        raise ValueError("deployment activation requires at least one commit per session")

    runtime = _installed_runtime_identity(candidate_commit)
    if runtime["turnalign_source_commit"] != candidate_commit:
        raise RuntimeError("candidate runtime identity changed during preflight")
    _validate_release_directory(release_root, previous_commit)
    _validate_release_directory(release_root, candidate_commit)
    initial_active_commit = _active_commit(release_root, current_link)
    if initial_active_commit != previous_commit:
        raise ValueError("preceding release must be active before candidate activation")
    boot_id = _read_linux_boot_id()
    started_at = _utc_timestamp()
    transaction = PendingDeploymentTransaction(
        schema_version=2,
        transaction_id=secrets.token_hex(32),
        operation="activation",
        previous_commit=previous_commit,
        candidate_commit=candidate_commit,
        boot_id=boot_id,
        release_root=str(release_root),
        current_link=str(current_link),
        service=service,
        systemctl=str(systemctl),
        ready_uri=ready_uri,
        websocket_uri=websocket_uri,
        created_at=started_at,
    )
    _write_pending_transaction(transaction)

    activation: RehearsalPhaseReport | None = None
    rollback: RehearsalPhaseReport | None = None
    activation_failures: list[str] = []
    rollback_required = False
    try:
        activation = await _exercise_phase(
            "activate",
            candidate_commit,
            release_root=release_root,
            current_link=current_link,
            systemctl=systemctl,
            service=service,
            ready_uri=ready_uri,
            websocket_uri=websocket_uri,
            restart_timeout=restart_timeout,
            readiness_timeout=readiness_timeout,
            readiness_interval=readiness_interval,
            probe=selected_probe,
            auth_token=auth_token,
        )
        if activation.passed:
            try:
                if _active_commit(release_root, current_link) != candidate_commit:
                    activation_failures.append(
                        "candidate is not active after successful activation"
                    )
            except ValueError as error:
                activation_failures.append(_failure_text(error))
            try:
                if _read_linux_boot_id() != boot_id:
                    activation_failures.append(
                        "host rebooted during candidate activation"
                    )
            except RuntimeError as error:
                activation_failures.append(_failure_text(error))
    finally:
        rollback_required = (
            activation is None or not activation.passed or bool(activation_failures)
        )
        if rollback_required:
            try:
                rollback = await _exercise_phase(
                    "activation-rollback",
                    previous_commit,
                    release_root=release_root,
                    current_link=current_link,
                    systemctl=systemctl,
                    service=service,
                    ready_uri=ready_uri,
                    websocket_uri=websocket_uri,
                    restart_timeout=restart_timeout,
                    readiness_timeout=readiness_timeout,
                    readiness_interval=readiness_interval,
                    probe=selected_probe,
                    auth_token=auth_token,
                )
            finally:
                try:
                    previous_active = (
                        _active_commit(release_root, current_link) == previous_commit
                    )
                except ValueError:
                    previous_active = False
                restart_incomplete = (
                    rollback is None
                    or rollback.restart is None
                    or rollback.restart.failure is not None
                )
                if not previous_active or restart_incomplete:
                    if not previous_active:
                        _activate_release(release_root, current_link, previous_commit)
                    _restart_service(systemctl, service, timeout=restart_timeout)
    if activation is None:
        raise RuntimeError("candidate activation did not produce deployment evidence")

    failures: list[str] = []
    if not activation.passed:
        failures.append("candidate activation phase did not pass")
    failures.extend(activation_failures)
    if activation.from_commit != previous_commit:
        failures.append("candidate activation did not start from the preceding release")
    if rollback_required:
        if rollback is None or not rollback.passed:
            failures.append("failed activation did not complete a verified rollback")
        elif rollback.from_commit != candidate_commit:
            failures.append("activation rollback did not start from the candidate")
    try:
        final_active_commit = _active_commit(release_root, current_link)
    except ValueError as error:
        final_active_commit = None
        failures.append(_failure_text(error))
    expected_final = previous_commit if rollback_required else candidate_commit
    if final_active_commit != expected_final:
        failures.append("deployment activation ended on the wrong release")
    try:
        if _read_linux_boot_id() != boot_id:
            failures.append("host rebooted during deployment activation")
    except RuntimeError as error:
        failures.append(_failure_text(error))
    return DeploymentActivationReport(
        schema_version=2,
        status="failed" if failures else "passed",
        candidate_commit=candidate_commit,
        previous_commit=previous_commit,
        boot_id=boot_id,
        release_root=str(release_root),
        current_link=str(current_link),
        lock_path=str(_DEPLOYMENT_LOCK_PATH),
        service=service,
        systemctl=str(systemctl),
        ready_uri=ready_uri,
        websocket_uri=websocket_uri,
        started_at=started_at,
        completed_at=_utc_timestamp(),
        initial_active_commit=initial_active_commit,
        final_active_commit=final_active_commit,
        transaction_id=transaction.transaction_id,
        transaction_path=str(_DEPLOYMENT_TRANSACTION_PATH),
        activation=activation,
        rollback=rollback,
        failures=tuple(failures),
    )


async def _run_deployment_recovery_locked(
    *,
    restart_timeout: float = 120.0,
    readiness_timeout: float = 300.0,
    readiness_interval: float = 1.0,
    probe: RehearsalProbeConfig | None = None,
    auth_token: str | None = None,
) -> DeploymentRecoveryReport:
    """Restore and probe the safe release recorded by an interrupted operation."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment recovery must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment recovery must run as root")
    for label, value in (
        ("restart_timeout", restart_timeout),
        ("readiness_timeout", readiness_timeout),
        ("readiness_interval", readiness_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{label} must be positive")
    selected_probe = probe or RehearsalProbeConfig()
    if selected_probe.backend is None or selected_probe.model is None:
        raise ValueError("deployment recovery requires an explicit backend and model")
    if selected_probe.probe_audio is None:
        raise ValueError("deployment recovery requires an explicit probe-audio WAV")
    if selected_probe.min_commits < 1:
        raise ValueError("deployment recovery requires at least one commit per session")

    transaction = _read_pending_transaction()
    runtime = _installed_runtime_identity(transaction.candidate_commit)
    if runtime["turnalign_source_commit"] != transaction.candidate_commit:
        raise RuntimeError("candidate runtime identity changed during recovery")
    release_root = Path(transaction.release_root)
    current_link = Path(transaction.current_link)
    systemctl = Path(transaction.systemctl)
    _validate_release_directory(release_root, transaction.previous_commit)
    _validate_release_directory(release_root, transaction.candidate_commit)
    try:
        initial_active_commit = _active_commit(release_root, current_link)
    except ValueError:
        initial_active_commit = None
    if initial_active_commit not in {
        transaction.previous_commit,
        transaction.candidate_commit,
    }:
        raise RuntimeError(
            "pending deployment transaction does not match the active release"
        )
    recovery_boot_id = _read_linux_boot_id()
    started_at = _utc_timestamp()
    recovery_commit = (
        transaction.previous_commit
        if transaction.operation == "activation"
        else transaction.candidate_commit
    )
    recovery = await _exercise_phase(
        f"{transaction.operation}-crash-recovery",
        recovery_commit,
        release_root=release_root,
        current_link=current_link,
        systemctl=systemctl,
        service=transaction.service,
        ready_uri=transaction.ready_uri,
        websocket_uri=transaction.websocket_uri,
        restart_timeout=restart_timeout,
        readiness_timeout=readiness_timeout,
        readiness_interval=readiness_interval,
        probe=selected_probe,
        auth_token=auth_token,
    )
    failures: list[str] = []
    if not recovery.passed:
        failures.append("interrupted deployment recovery phase did not pass")
    if recovery.from_commit != initial_active_commit:
        failures.append("deployment recovery observed an inconsistent active release")
    try:
        final_active_commit = _active_commit(release_root, current_link)
    except ValueError as error:
        final_active_commit = None
        failures.append(_failure_text(error))
    if final_active_commit != recovery_commit:
        failures.append("deployment recovery did not restore its recorded safe release")
    try:
        if _read_linux_boot_id() != recovery_boot_id:
            failures.append("host rebooted during deployment recovery")
    except RuntimeError as error:
        failures.append(_failure_text(error))
    return DeploymentRecoveryReport(
        schema_version=2,
        status="failed" if failures else "passed",
        transaction_id=transaction.transaction_id,
        transaction_path=str(_DEPLOYMENT_TRANSACTION_PATH),
        operation=transaction.operation,
        recovery_commit=recovery_commit,
        candidate_commit=transaction.candidate_commit,
        previous_commit=transaction.previous_commit,
        original_boot_id=transaction.boot_id,
        recovery_boot_id=recovery_boot_id,
        release_root=transaction.release_root,
        current_link=transaction.current_link,
        lock_path=str(_DEPLOYMENT_LOCK_PATH),
        service=transaction.service,
        systemctl=transaction.systemctl,
        ready_uri=transaction.ready_uri,
        websocket_uri=transaction.websocket_uri,
        started_at=started_at,
        completed_at=_utc_timestamp(),
        initial_active_commit=initial_active_commit,
        final_active_commit=final_active_commit,
        recovery=recovery,
        failures=tuple(failures),
    )


def _acquire_deployment_lock() -> int:
    import fcntl

    path = _DEPLOYMENT_LOCK_PATH
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            path,
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
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError("deployment lock must be a root-only regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another TurnAlign deployment operation is active") from error
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


async def run_deployment_rehearsal(
    previous_commit: str,
    candidate_commit: str,
    websocket_uri: str,
    *,
    release_root: Path = Path("/opt/turnalign/releases"),
    current_link: Path = Path("/opt/turnalign/current"),
    service: str = "turnalign.service",
    systemctl: Path = Path("/usr/bin/systemctl"),
    ready_uri: str = "http://127.0.0.1:8765/readyz",
    restart_timeout: float = 120.0,
    readiness_timeout: float = 300.0,
    readiness_interval: float = 1.0,
    probe: RehearsalProbeConfig | None = None,
    auth_token: str | None = None,
) -> DeploymentRehearsalReport:
    """Serialize and execute one rollback/restore rehearsal on the target host."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment rehearsal must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment rehearsal must run as root")
    lock_descriptor = _acquire_deployment_lock()
    try:
        _reject_pending_transaction()
        return await _run_deployment_rehearsal_locked(
            previous_commit,
            candidate_commit,
            websocket_uri,
            release_root=release_root,
            current_link=current_link,
            service=service,
            systemctl=systemctl,
            ready_uri=ready_uri,
            restart_timeout=restart_timeout,
            readiness_timeout=readiness_timeout,
            readiness_interval=readiness_interval,
            probe=probe,
            auth_token=auth_token,
        )
    finally:
        os.close(lock_descriptor)


async def run_deployment_activation(
    previous_commit: str,
    candidate_commit: str,
    websocket_uri: str,
    *,
    release_root: Path = Path("/opt/turnalign/releases"),
    current_link: Path = Path("/opt/turnalign/current"),
    service: str = "turnalign.service",
    systemctl: Path = Path("/usr/bin/systemctl"),
    ready_uri: str = "http://127.0.0.1:8765/readyz",
    restart_timeout: float = 120.0,
    readiness_timeout: float = 300.0,
    readiness_interval: float = 1.0,
    probe: RehearsalProbeConfig | None = None,
    auth_token: str | None = None,
) -> DeploymentActivationReport:
    """Serialize and transactionally activate one candidate release."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment activation must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment activation must run as root")
    lock_descriptor = _acquire_deployment_lock()
    try:
        return await _run_deployment_activation_locked(
            previous_commit,
            candidate_commit,
            websocket_uri,
            release_root=release_root,
            current_link=current_link,
            service=service,
            systemctl=systemctl,
            ready_uri=ready_uri,
            restart_timeout=restart_timeout,
            readiness_timeout=readiness_timeout,
            readiness_interval=readiness_interval,
            probe=probe,
            auth_token=auth_token,
        )
    finally:
        os.close(lock_descriptor)


async def run_deployment_recovery(
    *,
    restart_timeout: float = 120.0,
    readiness_timeout: float = 300.0,
    readiness_interval: float = 1.0,
    probe: RehearsalProbeConfig | None = None,
    auth_token: str | None = None,
) -> DeploymentRecoveryReport:
    """Recover an interrupted deployment operation and retain its marker."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment recovery must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment recovery must run as root")
    lock_descriptor = _acquire_deployment_lock()
    try:
        return await _run_deployment_recovery_locked(
            restart_timeout=restart_timeout,
            readiness_timeout=readiness_timeout,
            readiness_interval=readiness_interval,
            probe=probe,
            auth_token=auth_token,
        )
    finally:
        os.close(lock_descriptor)


def finalize_deployment_transaction(
    transaction_id: str,
    expected_active_commit: str,
) -> None:
    """Remove a durable marker only after its report is safely persisted."""

    if platform.system() != "Linux":
        raise RuntimeError("deployment finalization must run on the Linux production host")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("deployment finalization must run as root")
    if _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None:
        raise ValueError("transaction ID must be a lowercase 64-character hash")
    if _COMMIT_PATTERN.fullmatch(expected_active_commit) is None:
        raise ValueError("expected active commit must be a lowercase 40-character hash")
    lock_descriptor = _acquire_deployment_lock()
    try:
        transaction = _read_pending_transaction()
        if transaction.transaction_id != transaction_id:
            raise RuntimeError("deployment transaction identity changed")
        permitted_commits = (
            {transaction.previous_commit, transaction.candidate_commit}
            if transaction.operation == "activation"
            else {transaction.candidate_commit}
        )
        if expected_active_commit not in permitted_commits:
            raise RuntimeError("deployment finalization selected an unrelated release")
        active_commit = _active_commit(
            Path(transaction.release_root),
            Path(transaction.current_link),
        )
        if active_commit != expected_active_commit:
            raise RuntimeError("deployment finalization observed the wrong active release")
        _remove_pending_transaction(transaction_id)
    finally:
        os.close(lock_descriptor)
