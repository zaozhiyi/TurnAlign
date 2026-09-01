from __future__ import annotations

import hmac
import ipaddress
import math
from dataclasses import dataclass, field
from typing import Any

from .model_pool import BackendPoolCapacityError
from .recovery import RecoveryCapacityError, RecoveryConflictError

AUTH_TOKEN_MAX_BYTES = 8 * 1024


class ServerBusyError(RuntimeError):
    pass


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_auth_token(token: object) -> str:
    if not isinstance(token, str) or not token:
        raise ValueError("authentication token must be a non-empty string")
    if "\r" in token or "\n" in token or "\x00" in token:
        raise ValueError("authentication token must contain one non-empty token")
    try:
        encoded = token.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("authentication token must contain valid UTF-8 text") from error
    if len(encoded) > AUTH_TOKEN_MAX_BYTES:
        raise ValueError(
            f"authentication token exceeds {AUTH_TOKEN_MAX_BYTES} bytes"
        )
    return token


def _auth_tokens_equal(supplied: str, expected: str) -> bool:
    try:
        supplied_bytes = supplied.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(supplied_bytes, expected.encode("utf-8"))


def _validate_policy_name(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if (
        not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _request_name(
    request: dict[str, Any],
    key: str,
    default: str | None,
) -> str | None:
    raw = request.get(key)
    if raw is None:
        return (
            _validate_policy_name(default, label=f"default {key}")
            if default is not None
            else None
        )
    return _validate_policy_name(raw, label=key)


@dataclass(frozen=True, slots=True)
class ServerPolicy:
    """Explicit trust boundary for WebSocket-controlled configuration."""

    allowed_backends: frozenset[str] = field(default_factory=frozenset)
    allowed_models: frozenset[str] = field(default_factory=frozenset)
    allowed_languages: frozenset[str] = field(default_factory=frozenset)
    allowed_compute_types: frozenset[str] = field(default_factory=frozenset)
    allowed_components: frozenset[str] = field(default_factory=frozenset)
    allow_client_paths: bool = False
    allow_component_options: bool = False
    allow_remote: bool = False
    auth_token: str | None = None
    max_session_seconds: float = 14_400.0
    redact_errors: bool = True

    def __post_init__(self) -> None:
        for name in (
            "allowed_backends",
            "allowed_models",
            "allowed_languages",
            "allowed_compute_types",
            "allowed_components",
        ):
            values = getattr(self, name)
            if not isinstance(values, frozenset):
                raise TypeError(f"{name} must be a frozenset")
            for value in values:
                _validate_policy_name(value, label=f"{name} entry")
        for name in (
            "allow_client_paths",
            "allow_component_options",
            "allow_remote",
            "redact_errors",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if (
            isinstance(self.max_session_seconds, bool)
            or not isinstance(self.max_session_seconds, (int, float))
        ):
            raise TypeError("max_session_seconds must be a number")
        if not math.isfinite(self.max_session_seconds) or self.max_session_seconds <= 0:
            raise ValueError("max_session_seconds must be finite and positive")
        if self.auth_token is not None:
            validate_auth_token(self.auth_token)

    @classmethod
    def defaults(cls, backend: str, model: str | None = None) -> ServerPolicy:
        return cls(
            allowed_backends=frozenset({backend}),
            allowed_models=frozenset({model}) if model else frozenset(),
        )

    def validate_bind(self, host: str) -> None:
        if not self.allow_remote and not _is_loopback(host):
            raise ValueError("non-loopback WebSocket binding requires allow_remote=True")

    def validate_defaults(
        self,
        *,
        backend: str,
        model: str | None,
        language: str | None,
        compute_type: str | None,
    ) -> None:
        default_backend = _validate_policy_name(backend, label="default backend")
        if default_backend not in self.allowed_backends:
            raise ValueError("default backend is not allowed by server policy")
        for label, value in (
            ("default model", model),
            ("default language", language),
            ("default compute type", compute_type),
        ):
            if value is not None:
                _validate_policy_name(value, label=label)

    def validate_start(
        self,
        request: dict[str, Any],
        *,
        default_backend: str,
        default_model: str | None,
        default_language: str | None = None,
        default_compute_type: str | None = None,
    ) -> tuple[str, str | None, str | None, str | None]:
        if self.auth_token is not None:
            supplied = request.get("auth")
            if not isinstance(supplied, str) or not _auth_tokens_equal(
                supplied,
                self.auth_token,
            ):
                raise PermissionError("authentication failed")

        backend = _request_name(request, "backend", default_backend)
        if backend is None:
            raise ValueError("backend must not be empty")
        if backend not in self.allowed_backends:
            raise ValueError(f"backend {backend!r} is not allowed by server policy")

        model = _request_name(request, "model", default_model)
        if model != default_model and (model is None or model not in self.allowed_models):
            raise ValueError("requested model is not allowed by server policy")

        language = _request_name(request, "language", default_language)
        if language != default_language and (
            language is None or language not in self.allowed_languages
        ):
            raise ValueError("requested language is not allowed by server policy")

        compute_type = _request_name(request, "compute_type", default_compute_type)
        if compute_type != default_compute_type and (
            compute_type is None or compute_type not in self.allowed_compute_types
        ):
            raise ValueError("requested compute type is not allowed by server policy")

        if not self.allow_client_paths:
            forbidden = [key for key in ("executable", "model_path") if request.get(key)]
            if forbidden:
                raise ValueError(
                    f"client-controlled paths are disabled: {', '.join(forbidden)}"
                )

        components = set()
        for key in ("aligner", "diarizer", "online_diarizer"):
            component = _request_name(request, key, None)
            if component is not None:
                components.add(component)
        if not components.issubset(self.allowed_components):
            raise ValueError("requested component is not allowed by server policy")
        for key in (
            "aligner_options",
            "diarizer_options",
            "online_diarizer_options",
        ):
            value = request.get(key)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"{key} must be a JSON object")
            if value and not self.allow_component_options:
                raise ValueError("client-controlled component options are disabled")
        return backend, model, language, compute_type

    def public_error(self, error: BaseException) -> dict[str, object]:
        if isinstance(error, PermissionError):
            return {"type": "error", "code": "unauthorized", "message": "authentication failed"}
        if isinstance(error, (
            ServerBusyError,
            BackendPoolCapacityError,
            RecoveryCapacityError,
        )):
            return {
                "type": "error",
                "code": "server_busy",
                "message": "server session capacity reached; retry later",
            }
        if isinstance(error, RecoveryConflictError):
            return {
                "type": "error",
                "code": "session_conflict",
                "message": "recovery session is still active; retry later",
            }
        if isinstance(error, TimeoutError):
            return {
                "type": "error",
                "code": "timeout",
                "message": "session timed out",
            }
        if not self.redact_errors:
            return {"type": "error", "code": "session_error", "message": str(error)}
        if isinstance(error, (TypeError, ValueError, KeyError, LookupError)):
            return {
                "type": "error",
                "code": "invalid_request",
                "message": "request rejected by server policy or protocol",
            }
        return {
            "type": "error",
            "code": "session_error",
            "message": "session failed; inspect trusted server logs",
        }
