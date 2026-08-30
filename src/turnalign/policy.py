from __future__ import annotations

import hmac
import ipaddress
from dataclasses import dataclass, field
from typing import Any


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class ServerPolicy:
    """Explicit trust boundary for WebSocket-controlled configuration."""

    allowed_backends: frozenset[str] = field(default_factory=frozenset)
    allowed_models: frozenset[str] = field(default_factory=frozenset)
    allowed_components: frozenset[str] = field(default_factory=frozenset)
    allow_client_paths: bool = False
    allow_component_options: bool = False
    allow_remote: bool = False
    auth_token: str | None = None
    max_session_seconds: float = 14_400.0
    redact_errors: bool = True

    def __post_init__(self) -> None:
        if self.max_session_seconds <= 0:
            raise ValueError("max_session_seconds must be positive")

    @classmethod
    def defaults(cls, backend: str, model: str | None = None) -> ServerPolicy:
        return cls(
            allowed_backends=frozenset({backend}),
            allowed_models=frozenset({model}) if model else frozenset(),
        )

    def validate_bind(self, host: str) -> None:
        if not self.allow_remote and not _is_loopback(host):
            raise ValueError("non-loopback WebSocket binding requires allow_remote=True")

    def validate_start(
        self,
        request: dict[str, Any],
        *,
        default_backend: str,
        default_model: str | None,
    ) -> tuple[str, str | None]:
        if self.auth_token is not None:
            supplied = request.get("auth")
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.auth_token):
                raise PermissionError("authentication failed")

        backend = str(request.get("backend") or default_backend)
        if backend not in self.allowed_backends:
            raise ValueError(f"backend {backend!r} is not allowed by server policy")

        model_value = request.get("model")
        model = str(model_value) if model_value else default_model
        if model != default_model and (model is None or model not in self.allowed_models):
            raise ValueError("requested model is not allowed by server policy")

        if not self.allow_client_paths:
            forbidden = [key for key in ("executable", "model_path") if request.get(key)]
            if forbidden:
                raise ValueError(
                    f"client-controlled paths are disabled: {', '.join(forbidden)}"
                )

        components = {
            str(value)
            for value in (
                request.get("aligner"),
                request.get("diarizer"),
                request.get("online_diarizer"),
            )
            if value
        }
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
        return backend, model

    def public_error(self, error: BaseException) -> dict[str, str]:
        if isinstance(error, PermissionError):
            return {"type": "error", "code": "unauthorized", "message": "authentication failed"}
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
