from __future__ import annotations

import math
import re
import shlex
from collections import defaultdict

_UNIT_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(ms|s|min|h)?")
_INTEGER_PATTERN = re.compile(r"\d+")


def _parse_unit(content: bytes) -> dict[str, dict[str, list[str]]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("systemd unit is not UTF-8") from error
    if "\x00" in text or any(
        ord(character) < 32 and character not in {"\n", "\r", "\t"}
        for character in text
    ):
        raise ValueError("systemd unit contains control characters")

    logical_lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        fragment = raw_line.strip() if pending else raw_line
        continued = fragment.rstrip().endswith("\\")
        fragment = fragment.rstrip()
        if continued:
            fragment = fragment[:-1].rstrip()
        pending = f"{pending} {fragment.lstrip()}".strip() if pending else fragment
        if not continued:
            logical_lines.append(pending)
            pending = ""
    if pending:
        raise ValueError("systemd unit ends with an incomplete continuation")

    parsed: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    section: str | None = None
    for line in logical_lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            if not section or any(character.isspace() for character in section):
                raise ValueError("systemd unit contains an invalid section")
            continue
        if section is None or "=" not in line:
            raise ValueError("systemd unit contains a directive outside a section")
        key, value = line.split("=", 1)
        key = key.strip()
        if _UNIT_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("systemd unit contains an invalid directive name")
        parsed[section][key].append(value.strip())
    return {name: dict(values) for name, values in parsed.items()}


def _one(
    service: dict[str, list[str]],
    key: str,
    failures: list[str],
) -> str | None:
    values = service.get(key, [])
    if len(values) != 1:
        failures.append(f"systemd service must define {key} exactly once")
        return None
    return values[0]


def _enabled(value: str | None) -> bool:
    return value is not None and value.casefold() in {"1", "yes", "true", "on"}


def _duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    match = _DURATION_PATTERN.fullmatch(value)
    if match is None:
        return None
    amount = float(match.group(1))
    multiplier = {None: 1.0, "ms": 0.001, "s": 1.0, "min": 60.0, "h": 3600.0}[
        match.group(2)
    ]
    seconds = amount * multiplier
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _option_values(tokens: list[str], option: str) -> list[str | None]:
    values: list[str | None] = []
    for index, token in enumerate(tokens):
        if token == option:
            values.append(tokens[index + 1] if index + 1 < len(tokens) else None)
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def _one_option(
    tokens: list[str],
    option: str,
    failures: list[str],
) -> str | None:
    values = _option_values(tokens, option)
    if (
        len(values) != 1
        or values[0] is None
        or values[0] == ""
        or values[0].startswith("-")
    ):
        failures.append(f"systemd ExecStart must define {option} exactly once")
        return None
    return values[0]


def _positive_integer(value: str | None) -> int | None:
    if value is None or _INTEGER_PATTERN.fullmatch(value) is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _positive_number(value: str | None) -> float | None:
    if value is None or re.fullmatch(r"\d+(?:\.\d+)?", value) is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def validate_systemd_service(content: bytes) -> tuple[str, ...]:
    """Validate the retained Linux service unit's production safety invariants."""

    try:
        unit = _parse_unit(content)
    except ValueError as error:
        return (str(error),)
    failures: list[str] = []
    service = unit.get("Service")
    if service is None:
        return ("systemd unit has no Service section",)

    for key in (
        "NoNewPrivileges",
        "PrivateTmp",
        "PrivateDevices",
        "ProtectHome",
        "ProtectControlGroups",
        "ProtectKernelLogs",
        "ProtectKernelModules",
        "ProtectKernelTunables",
        "ProtectClock",
        "RestrictRealtime",
        "RestrictSUIDSGID",
        "LockPersonality",
    ):
        if not _enabled(_one(service, key, failures)):
            failures.append(f"systemd service must enable {key}")
    if _one(service, "ProtectSystem", failures) != "strict":
        failures.append("systemd service must use ProtectSystem=strict")

    user = _one(service, "User", failures)
    group = _one(service, "Group", failures)
    if not user or user.casefold() in {"0", "root"}:
        failures.append("systemd service must run as a non-root user")
    if not group or group.casefold() in {"0", "root"}:
        failures.append("systemd service must run as a non-root group")
    if _one(service, "UMask", failures) not in {"0077", "077"}:
        failures.append("systemd service must use a private UMask")
    if _one(service, "CapabilityBoundingSet", failures) != "":
        failures.append("systemd service must clear its capability bounding set")
    if _one(service, "AmbientCapabilities", failures) != "":
        failures.append("systemd service must clear ambient capabilities")
    if _one(service, "Type", failures) not in {"simple", "exec"}:
        failures.append("systemd service must use a foreground process type")
    if _one(service, "Restart", failures) != "on-failure":
        failures.append("systemd service must restart after runtime failures")
    if _duration_seconds(_one(service, "RestartSec", failures)) is None:
        failures.append("systemd service must use a positive restart delay")
    if _one(service, "KillSignal", failures) != "SIGTERM":
        failures.append("systemd service must initiate graceful SIGTERM shutdown")
    if _one(service, "KillMode", failures) != "mixed":
        failures.append("systemd service must bound the full process group on shutdown")
    working_directory = _one(service, "WorkingDirectory", failures)
    if working_directory is None or not working_directory.startswith("/"):
        failures.append("systemd service must use an absolute working directory")
    for key in ("StateDirectory", "CacheDirectory"):
        value = _one(service, key, failures)
        if value is None or not value or "/" in value or value in {".", ".."}:
            failures.append(f"systemd service must define a managed {key}")
    if _positive_integer(_one(service, "LimitNOFILE", failures)) is None:
        failures.append("systemd service must set a positive file-descriptor limit")
    if _positive_integer(_one(service, "TasksMax", failures)) is None:
        failures.append("systemd service must set a positive task limit")
    for key in ("StandardOutput", "StandardError"):
        if _one(service, key, failures) != "journal":
            failures.append(f"systemd service must send {key} to the journal")

    address_families = _one(service, "RestrictAddressFamilies", failures)
    if address_families is None or set(address_families.split()) != {
        "AF_UNIX",
        "AF_INET",
        "AF_INET6",
    }:
        failures.append("systemd service must restrict address families")
    if _one(service, "IPAddressDeny", failures) != "any":
        failures.append("systemd service must deny non-allowlisted network traffic")
    if _one(service, "IPAddressAllow", failures) != "localhost":
        failures.append("systemd service must allow only loopback network traffic")

    credentials = service.get("LoadCredential", [])
    if len(credentials) != 1 or not credentials[0].startswith("auth-token:/"):
        failures.append("systemd service must load one root-managed auth-token credential")
    if service.get("EnvironmentFile"):
        failures.append("systemd service must not load a mutable environment file")
    for environment in service.get("Environment", []):
        try:
            assignments = shlex.split(environment, posix=True)
        except ValueError:
            failures.append("systemd service contains an invalid Environment directive")
            continue
        if any(
            re.search(
                r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL)",
                item.split("=", 1)[0],
                re.IGNORECASE,
            )
            for item in assignments
            if "=" in item
        ):
            failures.append("systemd service must not expose secrets in its environment")

    exec_start = _one(service, "ExecStart", failures)
    try:
        tokens = shlex.split(exec_start, posix=True) if exec_start is not None else []
    except ValueError:
        tokens = []
    if (
        len(tokens) < 2
        or not tokens[0].startswith("/")
        or tokens[0].rsplit("/", 1)[-1] != "turnalign"
        or tokens[1] != "serve"
    ):
        failures.append("systemd ExecStart must invoke an absolute TurnAlign serve command")
        return tuple(failures)
    if "--allow-remote" in tokens:
        failures.append("systemd service must not enable TurnAlign remote binding")
    if tokens.count("--preload") != 1:
        failures.append("systemd service must preload model replicas")
    if tokens.count("--require-immutable-model-revision") != 1:
        failures.append("systemd service must require an immutable model revision")

    host = _one_option(tokens, "--host", failures)
    if host not in {"127.0.0.1", "::1"}:
        failures.append("systemd service must bind TurnAlign only to loopback")
    port = _positive_integer(_one_option(tokens, "--port", failures))
    if port is None or port > 65_535:
        failures.append("systemd service must use a valid positive port")
    auth_file = _one_option(tokens, "--auth-token-file", failures)
    if auth_file not in {
        "${CREDENTIALS_DIRECTORY}/auth-token",
        "%d/auth-token",
    }:
        failures.append("systemd service must read auth from its credential directory")
    warmup = _one_option(tokens, "--warmup-file", failures)
    if warmup is None or not warmup.startswith("/"):
        failures.append("systemd service must use an absolute warm-up file")
    for option in ("--backend", "--device"):
        _one_option(tokens, option, failures)

    integer_options = {
        option: _positive_integer(_one_option(tokens, option, failures))
        for option in (
            "--backend-replicas",
            "--max-concurrent-sessions",
            "--max-recovery-sessions",
            "--max-recovery-audio-mib",
            "--max-recovery-total-mib",
        )
    }
    recovery_audio_mib = integer_options["--max-recovery-audio-mib"]
    recovery_total_mib = integer_options["--max-recovery-total-mib"]
    if any(value is None for value in integer_options.values()):
        failures.append("systemd service has invalid positive capacity limits")
    elif (
        recovery_audio_mib is not None
        and recovery_total_mib is not None
        and recovery_audio_mib > recovery_total_mib
    ):
        failures.append("systemd recovery audio limit exceeds its total limit")

    numeric_options = {
        option: _positive_number(_one_option(tokens, option, failures))
        for option in (
            "--max-session-seconds",
            "--recovery-ttl-seconds",
            "--initialization-timeout",
            "--client-idle-timeout",
            "--finalization-timeout",
            "--worker-shutdown-timeout",
            "--shutdown-grace-timeout",
        )
    }
    max_session_seconds = numeric_options["--max-session-seconds"]
    if any(value is None for value in numeric_options.values()):
        failures.append("systemd service has invalid positive lifecycle limits")
    elif max_session_seconds is not None and recovery_audio_mib is not None:
        maximum_pcm_bytes = max_session_seconds * 32_000
        recovery_bytes = recovery_audio_mib * 1024 * 1024
        if maximum_pcm_bytes > recovery_bytes:
            failures.append("systemd session duration exceeds recoverable audio capacity")

    stop_timeout = _duration_seconds(_one(service, "TimeoutStopSec", failures))
    start_timeout = _duration_seconds(_one(service, "TimeoutStartSec", failures))
    if numeric_options["--shutdown-grace-timeout"] is not None and (
        stop_timeout is None
        or stop_timeout <= numeric_options["--shutdown-grace-timeout"]
    ):
        failures.append("systemd stop timeout must exceed TurnAlign shutdown grace")
    if numeric_options["--initialization-timeout"] is not None and (
        start_timeout is None
        or start_timeout <= numeric_options["--initialization-timeout"]
    ):
        failures.append("systemd start timeout must exceed TurnAlign initialization")

    prestarts = service.get("ExecStartPre", [])
    if not any("${CREDENTIALS_DIRECTORY}/auth-token" in item and " -s " in item for item in prestarts):
        failures.append("systemd service must fail closed on an empty auth credential")
    if warmup is not None and not any(warmup in item and " -r " in item for item in prestarts):
        failures.append("systemd service must fail closed on an unreadable warm-up file")
    return tuple(failures)
