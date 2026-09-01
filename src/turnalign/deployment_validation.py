from __future__ import annotations

import ipaddress
import math
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

_UNIT_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(ms|s|min|h)?")
_INTEGER_PATTERN = re.compile(r"\d+")


@dataclass(slots=True)
class _NginxNode:
    name: str
    args: tuple[str, ...]
    children: list[_NginxNode] | None = None


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
    managed_directories: dict[str, str | None] = {}
    for key in ("StateDirectory", "CacheDirectory"):
        value = _one(service, key, failures)
        managed_directories[key] = value
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
        len(tokens) < 7
        or not tokens[0].startswith("/")
        or tokens[:7] != [
            "/opt/turnalign/current/venv/bin/python",
            "-I",
            "-B",
            "-u",
            "-m",
            "turnalign.cli",
            "serve",
        ]
    ):
        failures.append(
            "systemd ExecStart must invoke TurnAlign through the isolated "
            "versioned Python runtime"
        )
        return tuple(failures)
    if "--allow-remote" in tokens:
        failures.append("systemd service must not enable TurnAlign remote binding")
    if tokens.count("--preload") != 1:
        failures.append("systemd service must preload model replicas")
    if tokens.count("--require-local-model") != 1:
        failures.append("systemd service must require retained local model files")
    if "--no-require-local-model" in tokens:
        failures.append("systemd service must not disable retained local model files")
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
    backend = _one_option(tokens, "--backend", failures)
    for option in ("--device", "--model"):
        _one_option(tokens, option, failures)
    if backend in {"funasr", "funasr-streaming"}:
        revisions = [
            value.removeprefix("model_revision=")
            for value in _option_values(tokens, "--backend-option")
            if isinstance(value, str) and value.startswith("model_revision=")
        ]
        if len(revisions) != 1 or re.fullmatch(
            r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})",
            revisions[0] if revisions else "",
        ) is None:
            failures.append(
                "systemd FunASR service must pin one immutable model_revision"
            )
    model_path_value = _one_option(tokens, "--model-path", failures)
    if model_path_value is not None:
        model_path = PurePosixPath(model_path_value)
        model_root = PurePosixPath("/var/lib/turnalign/models")
        if (
            not model_path.is_absolute()
            or str(model_path) != model_path_value
            or model_path == model_root
            or model_root not in model_path.parents
            or any(part in {".", ".."} for part in model_path.parts)
        ):
            failures.append(
                "systemd service model path must be a canonical child of "
                "/var/lib/turnalign/models"
            )
        state_name = managed_directories.get("StateDirectory")
        if isinstance(state_name, str) and "/" not in state_name:
            writable_state = PurePosixPath("/var/lib") / state_name
            if writable_state == model_path or writable_state in model_path.parents:
                failures.append(
                    "systemd service must not place model files under its writable "
                    "StateDirectory"
                )

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
            "--backend-cancel-timeout",
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


def _nginx_tokens(content: bytes) -> list[str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Nginx configuration is not UTF-8") from error
    if "\x00" in text or any(
        ord(character) < 32 and character not in {"\n", "\r", "\t"}
        for character in text
    ):
        raise ValueError("Nginx configuration contains control characters")

    tokens: list[str] = []
    current: list[str] = []
    token_started = False
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if escaped:
            current.append(character)
            token_started = True
            escaped = False
        elif character == "\\":
            escaped = True
            token_started = True
        elif quote is not None:
            if character == quote:
                quote = None
            else:
                current.append(character)
                token_started = True
        elif character in {"'", '"'}:
            quote = character
            token_started = True
        elif character == "#":
            if token_started:
                tokens.append("".join(current))
                current = []
                token_started = False
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
        elif character.isspace():
            if token_started:
                tokens.append("".join(current))
                current = []
                token_started = False
        elif character in {"{", "}", ";"}:
            if token_started:
                tokens.append("".join(current))
                current = []
                token_started = False
            tokens.append(character)
        else:
            current.append(character)
            token_started = True
        index += 1
    if escaped or quote is not None:
        raise ValueError("Nginx configuration has an unterminated quote or escape")
    if token_started:
        tokens.append("".join(current))
    if len(tokens) > 100_000:
        raise ValueError("Nginx configuration contains too many tokens")
    return tokens


def _parse_nginx(content: bytes) -> list[_NginxNode]:
    roots: list[_NginxNode] = []
    current = roots
    stack: list[list[_NginxNode]] = []
    words: list[str] = []
    for token in _nginx_tokens(content):
        if token == ";":
            if not words:
                raise ValueError("Nginx configuration contains an empty directive")
            current.append(_NginxNode(words[0], tuple(words[1:])))
            words = []
        elif token == "{":
            if not words:
                raise ValueError("Nginx configuration contains an unnamed block")
            children: list[_NginxNode] = []
            node = _NginxNode(words[0], tuple(words[1:]), children)
            current.append(node)
            stack.append(current)
            current = children
            words = []
        elif token == "}":
            if words or not stack:
                raise ValueError("Nginx configuration has mismatched block syntax")
            current = stack.pop()
        else:
            words.append(token)
    if words or stack:
        raise ValueError("Nginx configuration ends with an incomplete directive or block")
    return roots


def _children(nodes: list[_NginxNode], name: str) -> list[_NginxNode]:
    return [node for node in nodes if node.name == name]


def _walk(nodes: list[_NginxNode]) -> list[_NginxNode]:
    result: list[_NginxNode] = []
    pending = list(reversed(nodes))
    while pending:
        node = pending.pop()
        result.append(node)
        if node.children is not None:
            pending.extend(reversed(node.children))
    return result


def _exact_directive(
    nodes: list[_NginxNode],
    name: str,
    args: tuple[str, ...],
) -> bool:
    matches = _children(nodes, name)
    return len(matches) == 1 and matches[0].children is None and matches[0].args == args


def _location(server: list[_NginxNode], path: str) -> list[_NginxNode] | None:
    matches = [
        node
        for node in _children(server, "location")
        if node.args == ("=", path) and node.children is not None
    ]
    return matches[0].children if len(matches) == 1 else None


def _exact_header(nodes: list[_NginxNode], name: str, value: str) -> bool:
    matches = [
        node
        for node in _children(nodes, "proxy_set_header")
        if node.args and node.args[0].casefold() == name.casefold()
    ]
    return len(matches) == 1 and matches[0].args == (name, value)


def _parameter(args: tuple[str, ...], name: str) -> str | None:
    prefix = f"{name}="
    values = [value.removeprefix(prefix) for value in args if value.startswith(prefix)]
    return values[0] if len(values) == 1 and values[0] else None


def _systemd_endpoint_and_deadlines(
    content: bytes,
) -> tuple[str, int, dict[str, float]] | None:
    try:
        service = _parse_unit(content).get("Service", {})
        exec_start = service.get("ExecStart", [])
        if len(exec_start) != 1:
            return None
        tokens = shlex.split(exec_start[0], posix=True)
        host_values = _option_values(tokens, "--host")
        port_values = _option_values(tokens, "--port")
        if len(host_values) != 1 or len(port_values) != 1:
            return None
        host = host_values[0]
        port = _positive_integer(port_values[0])
        if host is None or port is None:
            return None
        deadlines = {}
        for name in (
            "initialization-timeout",
            "client-idle-timeout",
            "finalization-timeout",
        ):
            values = _option_values(tokens, f"--{name}")
            if len(values) != 1 or (value := _positive_number(values[0])) is None:
                return None
            deadlines[name] = value
        return host, port, deadlines
    except (ValueError, TypeError):
        return None


def _listen_port(args: tuple[str, ...]) -> int | None:
    if not args:
        return None
    address = args[0]
    if address.isdecimal():
        return int(address)
    try:
        return urlsplit(f"//{address}").port
    except ValueError:
        return None


def _upstream_endpoint(value: str) -> tuple[str, int] | None:
    try:
        parsed = urlsplit(f"//{value}")
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if host is None or port is None:
        return None
    try:
        if not ipaddress.ip_address(host).is_loopback:
            return None
    except ValueError:
        return None
    return host, port


def _nginx_duration(value: str) -> float | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)?", value)
    if match is None:
        return None
    amount = float(match.group(1))
    multiplier = {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[
        match.group(2)
    ]
    seconds = amount * multiplier
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def validate_nginx_config(
    content: bytes,
    systemd_content: bytes,
    websocket_uri: str,
) -> tuple[str, ...]:
    """Validate retained TLS proxy semantics against the probed service topology."""

    try:
        roots = _parse_nginx(content)
    except ValueError as error:
        return (str(error),)
    failures: list[str] = []
    if any(node.name == "include" for node in _walk(roots)):
        failures.append("Nginx evidence must be self-contained without include directives")
    http_blocks = [
        node
        for node in _children(roots, "http")
        if not node.args and node.children is not None
    ]
    if len(http_blocks) > 1:
        failures.append("Nginx evidence contains multiple HTTP contexts")
        http_context = roots
    elif http_blocks:
        http_context = http_blocks[0].children or []
    else:
        http_context = roots

    try:
        public = urlsplit(websocket_uri)
        public_host = public.hostname
        public_port = public.port or 443
        public_path = public.path or "/"
    except ValueError:
        return ("Nginx validator received an invalid WebSocket URI",)
    if public_host is None:
        return ("Nginx validator received a WebSocket URI without a host",)

    servers = [
        node
        for node in _children(http_context, "server")
        if node.children is not None
        and any(
            child.name == "server_name" and public_host in child.args
            for child in node.children
        )
    ]
    if len(servers) != 1:
        return ("Nginx configuration must contain one server for the probed host",)
    server = servers[0].children
    if server is None:
        return ("Nginx probed-host server block is empty",)
    server_names = _children(server, "server_name")
    if len(server_names) != 1 or server_names[0].args != (public_host,):
        failures.append("Nginx server_name must exactly match the probed host")
    listens = _children(server, "listen")
    if not listens or not any(
        _listen_port(node.args) == public_port and "ssl" in node.args[1:]
        for node in listens
    ):
        failures.append("Nginx must listen with TLS on the probed public port")
    if any("ssl" not in node.args[1:] for node in listens):
        failures.append("Nginx probed-host server must not expose a plaintext listener")
    for directive in ("ssl_certificate", "ssl_certificate_key"):
        values = _children(server, directive)
        if (
            len(values) != 1
            or len(values[0].args) != 1
            or not values[0].args[0].startswith("/")
        ):
            failures.append(f"Nginx must define one absolute {directive}")
    certificates = _children(server, "ssl_certificate")
    certificate_keys = _children(server, "ssl_certificate_key")
    if (
        len(certificates) == 1
        and len(certificate_keys) == 1
        and certificates[0].args == certificate_keys[0].args
    ):
        failures.append("Nginx certificate and private-key paths must be distinct")
    if not _exact_directive(server, "ssl_protocols", ("TLSv1.2", "TLSv1.3")):
        failures.append("Nginx must allow only TLSv1.2 and TLSv1.3")
    if not _exact_directive(server, "ssl_session_tickets", ("off",)):
        failures.append("Nginx must disable TLS session tickets")
    if not _exact_directive(server, "server_tokens", ("off",)):
        failures.append("Nginx must disable server version tokens")

    metrics = _location(server, "/metrics")
    if metrics is None or not _exact_directive(metrics, "return", ("404",)) or _children(
        metrics, "proxy_pass"
    ):
        failures.append("Nginx must keep metrics private with an exact 404 location")
    for path in ("/healthz", "/readyz"):
        location = _location(server, path)
        if location is None or not _exact_directive(
            location,
            "proxy_pass",
            (f"http://turnalign_backend{path}",),
        ):
            failures.append(f"Nginx must proxy the exact {path} endpoint")

    websocket = _location(server, public_path)
    if websocket is None:
        failures.append("Nginx has no exact location for the probed WebSocket path")
        return tuple(failures)
    for name, args, label in (
        ("proxy_http_version", ("1.1",), "HTTP/1.1 proxying"),
        ("proxy_buffering", ("off",), "response buffering disablement"),
        ("proxy_request_buffering", ("off",), "request buffering disablement"),
        ("proxy_next_upstream", ("off",), "upstream retry disablement"),
    ):
        matches = [node for node in _children(websocket, name) if node.args == args]
        if len(matches) != 1:
            failures.append(f"Nginx WebSocket location lacks exact {label}")
    for name, value, label in (
        ("Upgrade", "$http_upgrade", "Upgrade forwarding"),
        ("Host", "$host", "sanitized Host forwarding"),
        (
            "X-Forwarded-For",
            "$proxy_add_x_forwarded_for",
            "forwarded client addressing",
        ),
        ("X-Forwarded-Proto", "https", "HTTPS forwarding"),
    ):
        if not _exact_header(websocket, name, value):
            failures.append(f"Nginx WebSocket location lacks exact {label}")
    connection_headers = [
        node
        for node in _children(websocket, "proxy_set_header")
        if node.args and node.args[0].casefold() == "connection"
    ]
    if (
        len(connection_headers) != 1
        or len(connection_headers[0].args) != 2
        or connection_headers[0].args[0] != "Connection"
    ):
        failures.append("Nginx WebSocket location has no unambiguous Connection header")
        connection_variable = None
    else:
        connection_variable = connection_headers[0].args[1]

    request_limits = _children(websocket, "limit_req")
    request_zone = (
        _parameter(request_limits[0].args, "zone")
        if len(request_limits) == 1
        else None
    )
    request_burst = (
        _parameter(request_limits[0].args, "burst")
        if len(request_limits) == 1
        else None
    )
    if (
        request_zone is None
        or request_burst is None
        or _positive_integer(request_burst) is None
        or len(request_limits[0].args) != 3
        or request_limits[0].args.count("nodelay") != 1
    ):
        failures.append("Nginx WebSocket handshake rate limit is missing")
    else:
        request_zone_definitions = [
            node
            for node in _children(http_context, "limit_req_zone")
            if len(node.args) >= 2
            and node.args[1].startswith(f"zone={request_zone}:")
        ]
        if len(request_zone_definitions) != 1 or not (
            len(request_zone_definitions[0].args) == 3
            and request_zone_definitions[0].args[0] == "$binary_remote_addr"
            and re.fullmatch(
                r"[1-9]\d*[kKmMgG]",
                request_zone_definitions[0].args[1].split(":", 1)[1],
            )
            and request_zone_definitions[0].args[2].startswith("rate=")
            and re.fullmatch(
                r"[1-9]\d*r/s",
                request_zone_definitions[0].args[2].removeprefix("rate="),
            )
        ):
            failures.append("Nginx WebSocket handshake rate-limit zone is missing")

    connection_limits = _children(websocket, "limit_conn")
    connection_zone = (
        connection_limits[0].args[0]
        if len(connection_limits) == 1 and len(connection_limits[0].args) == 2
        else None
    )
    if (
        connection_zone is None
        or _positive_integer(connection_limits[0].args[1]) is None
    ):
        failures.append("Nginx WebSocket connection limit is missing")
    else:
        connection_zone_definitions = [
            node
            for node in _children(http_context, "limit_conn_zone")
            if len(node.args) >= 2
            and node.args[1].startswith(f"zone={connection_zone}:")
        ]
        if len(connection_zone_definitions) != 1 or not (
            len(connection_zone_definitions[0].args) == 2
            and connection_zone_definitions[0].args[0] == "$binary_remote_addr"
            and re.fullmatch(
                r"[1-9]\d*[kKmMgG]",
                connection_zone_definitions[0].args[1].split(":", 1)[1],
            )
        ):
            failures.append("Nginx WebSocket connection-limit zone is missing")

    proxy_passes = _children(websocket, "proxy_pass")
    if len(proxy_passes) != 1 or len(proxy_passes[0].args) != 1:
        failures.append("Nginx WebSocket location must define one upstream")
        upstream_name = None
    else:
        try:
            parsed_upstream = urlsplit(proxy_passes[0].args[0])
            upstream_name = (
                parsed_upstream.hostname
                if parsed_upstream.scheme == "http"
                and parsed_upstream.username is None
                and parsed_upstream.password is None
                and parsed_upstream.port is None
                and not parsed_upstream.path
                and not parsed_upstream.query
                and not parsed_upstream.fragment
                else None
            )
        except ValueError:
            upstream_name = None
        if upstream_name is None:
            failures.append("Nginx WebSocket upstream must be one plain HTTP origin")
    upstream_blocks = [
        node
        for node in _children(http_context, "upstream")
        if node.args == (upstream_name,) and node.children is not None
    ]
    endpoints: list[tuple[str, int]] = []
    if len(upstream_blocks) != 1:
        failures.append("Nginx WebSocket upstream block is missing or ambiguous")
    else:
        upstream_servers = _children(upstream_blocks[0].children or [], "server")
        if len(upstream_servers) != 1 or len(upstream_servers[0].args) != 1:
            failures.append("Nginx upstream must contain exactly one TCP server")
        else:
            endpoint = _upstream_endpoint(upstream_servers[0].args[0])
            if endpoint is None:
                failures.append("Nginx upstream must contain only a loopback TCP server")
            else:
                endpoints.append(endpoint)

    maps = [
        node
        for node in _children(http_context, "map")
        if len(node.args) == 2
        and node.args[0] == "$http_upgrade"
        and node.args[1] == connection_variable
        and node.children is not None
    ]
    if len(maps) != 1 or not (
        _exact_directive(maps[0].children or [], "default", ("upgrade",))
        and _exact_directive(maps[0].children or [], "", ("close",))
    ):
        failures.append("Nginx Connection header map is missing or unsafe")

    service = _systemd_endpoint_and_deadlines(systemd_content)
    if service is None:
        failures.append("Nginx cannot be checked against systemd lifecycle evidence")
    else:
        service_host, service_port, deadlines = service
        if endpoints and any(
            host != service_host or port != service_port for host, port in endpoints
        ):
            failures.append("Nginx upstream does not match the systemd listen endpoint")
        read_values = _children(websocket, "proxy_read_timeout")
        send_values = _children(websocket, "proxy_send_timeout")
        read_timeout = (
            _nginx_duration(read_values[0].args[0])
            if len(read_values) == 1 and len(read_values[0].args) == 1
            else None
        )
        send_timeout = (
            _nginx_duration(send_values[0].args[0])
            if len(send_values) == 1 and len(send_values[0].args) == 1
            else None
        )
        if read_timeout is None or read_timeout <= max(deadlines.values()):
            failures.append("Nginx read timeout must exceed every application phase")
        if send_timeout is None or send_timeout <= deadlines["client-idle-timeout"]:
            failures.append("Nginx send timeout must exceed the client idle deadline")
    return tuple(failures)
