from __future__ import annotations

import json


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON number: {value}")


def strict_json_loads(source: str | bytes) -> object:
    """Parse standards-compliant JSON while preserving key uniqueness."""
    return json.loads(
        source,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def strict_json_object(source: str | bytes, *, label: str) -> dict[str, object]:
    payload = strict_json_loads(source)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain one JSON object")
    return payload
