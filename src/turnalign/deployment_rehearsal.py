from __future__ import annotations

import asyncio
import ipaddress
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
_DEPLOYMENT_LOCK_PATH = Path("/run/lock/turnalign-deployment.lock")
_DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


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
    rollback: RehearsalPhaseReport
    restore: RehearsalPhaseReport
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
            "rollback rehearsal requires a credential-free public wss:// endpoint"
        )


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
            or metadata.st_uid != 0
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
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError(f"{label} must be a root-owned immutable directory")
    identities = list(
        (release / "venv" / "lib").glob(
            "python*/site-packages/turnalign/_source_commit.txt"
        )
    )
    if len(identities) != 1 or _read_release_identity(identities[0]) != commit:
        raise ValueError(f"release directory is not bound to source commit {commit}")
    python = release / "venv" / "bin" / "python"
    if not python.exists() or not python.is_file() or not os.access(python, os.X_OK):
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
            headers={"Connection": "close", "User-Agent": "turnalign-rehearsal/1"},
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
                    return ReadinessEvidence(
                        uri=uri,
                        status_code=200,
                        ready=True,
                        preloaded=True,
                        attempts=attempts,
                        seconds=_seconds(started),
                        failure=None,
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
        schema_version=1,
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
        rollback=rollback,
        restore=restore,
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
