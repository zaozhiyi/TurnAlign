from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .plugins import AsrConfig


ENTRY_POINT_GROUPS = {
    "asr": "turnalign.backends",
    "vad": "turnalign.vad",
    "alignment": "turnalign.alignment",
    "diarization": "turnalign.diarization",
}

BUILTIN_ASR = {
    "faster-whisper": "turnalign.backends.faster_whisper:FasterWhisperBackend",
    "funasr": "turnalign.backends.funasr:FunAsrBackend",
    "glm-asr": "turnalign.backends.transformers:GlmAsrBackend",
    "transformers-whisper": "turnalign.backends.transformers:TransformersWhisperBackend",
    "whisper-cpp": "turnalign.backends.whisper_cpp:WhisperCppBackend",
}


def _import_string(value: str):
    module_name, attribute = value.split(":", 1)
    module = __import__(module_name, fromlist=[attribute])
    return getattr(module, attribute)


def discover(kind: str) -> dict[str, Any]:
    """Return lazy plugin entry points for one component kind."""
    if kind not in ENTRY_POINT_GROUPS:
        raise ValueError(f"unknown plugin kind: {kind}")
    group = ENTRY_POINT_GROUPS[kind]
    return {item.name: item for item in entry_points(group=group)}


def load(kind: str, name: str):
    if kind == "asr" and name in BUILTIN_ASR:
        return _import_string(BUILTIN_ASR[name])
    plugins = discover(kind)
    if name not in plugins:
        available = ", ".join(sorted(plugins)) or "none"
        raise LookupError(f"plugin {name!r} not found for {kind}; available: {available}")
    return plugins[name].load()


def available(kind: str) -> list[str]:
    names = set(discover(kind))
    names.discard("jsonl")  # legacy replay adapter, not an audio backend factory
    if kind == "asr":
        names.update(BUILTIN_ASR)
    return sorted(names)


def create_asr(name: str, config: AsrConfig):
    implementation = load("asr", name)
    factory = getattr(implementation, "create", None)
    if callable(factory):
        return factory(config)
    return implementation(config)


def create_component(kind: str, name: str, options: dict[str, Any] | None = None):
    """Create a VAD, alignment, or diarization plugin from keyword options."""
    if kind == "asr":
        raise ValueError("use create_asr for ASR backends")
    implementation = load(kind, name)
    factory = getattr(implementation, "create", None)
    if callable(factory):
        return factory(options or {})
    return implementation(**(options or {}))
