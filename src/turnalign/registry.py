from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from .hints import AsrHints
from .plugins import AsrConfig

ENTRY_POINT_GROUPS = {
    "asr": "turnalign.backends",
    "vad": "turnalign.vad",
    "alignment": "turnalign.alignment",
    "diarization": "turnalign.diarization",
    "online_diarization": "turnalign.online_diarization",
}

BUILTIN_ASR = {
    "faster-whisper": "turnalign.backends.faster_whisper:FasterWhisperBackend",
    "funasr": "turnalign.backends.funasr:FunAsrBackend",
    "funasr-streaming": "turnalign.backends.funasr_streaming:FunAsrStreamingBackend",
    "glm-asr": "turnalign.backends.transformers:GlmAsrBackend",
    "transformers-whisper": "turnalign.backends.transformers:TransformersWhisperBackend",
    "whisper-cpp": "turnalign.backends.whisper_cpp:WhisperCppBackend",
}

BUILTIN_COMPONENTS = {
    "vad": {
        "energy": "turnalign.components.energy_vad:EnergyVadBackend",
        "fsmn-vad": "turnalign.components.funasr:FsmnVadBackend",
    },
    "alignment": {
        "paraformer": "turnalign.components.funasr:ParaformerAlignmentBackend",
    },
    "diarization": {
        "campp": "turnalign.components.funasr:CamppDiarizationBackend",
    },
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
    if name in BUILTIN_COMPONENTS.get(kind, {}):
        return _import_string(BUILTIN_COMPONENTS[kind][name])
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
    names.update(BUILTIN_COMPONENTS.get(kind, {}))
    return sorted(names)


def _validate_hints(name: str, implementation, hints: AsrHints) -> None:
    capabilities = getattr(implementation, "capabilities", None)
    if capabilities is not None:
        if hints.hotwords and not capabilities.hotwords:
            raise ValueError(f"ASR backend {name!r} does not support hotwords")
        if hints.context and not capabilities.context_prompt:
            raise ValueError(f"ASR backend {name!r} does not support context prompts")
        if hints.boost is not None and not capabilities.hotword_boost:
            raise ValueError(f"ASR backend {name!r} does not support numeric hotword boost")


def supports_session_hints(name: str) -> bool:
    """Return whether a backend can change hints without reloading its model."""
    try:
        implementation = load("asr", name)
    except LookupError:
        return False
    return bool(getattr(implementation, "session_hints", False))


def validate_asr_hints(name: str, hints: AsrHints, implementation=None) -> None:
    _validate_hints(name, implementation or load("asr", name), hints)


def create_asr(name: str, config: AsrConfig):
    implementation = load("asr", name)
    _validate_hints(name, implementation, config.hints)
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
