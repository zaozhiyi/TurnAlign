import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from turnalign.backends.faster_whisper import FasterWhisperBackend
from turnalign.backends.funasr import FunAsrBackend
from turnalign.backends.transformers import TransformersWhisperBackend
from turnalign.hints import AsrHints, glm_transcription_prompt, whisper_initial_prompt
from turnalign.models import AudioChunk
from turnalign.plugins import AsrConfig, BackendCapabilities
from turnalign.registry import create_asr


def chunk(duration: float = 0.2) -> AudioChunk:
    return AudioChunk(array("h", [1000] * round(16_000 * duration)).tobytes(), 0.0)


class CaptureModel:
    def __init__(self, result):
        self.result = result
        self.options = None

    def transcribe(self, audio, **options):
        self.options = options
        return self.result

    def generate(self, **options):
        self.options = options
        return self.result


class CaptureTokenizer:
    def __init__(self):
        self.prompt = None

    def get_prompt_ids(self, prompt, return_tensors):
        self.prompt = prompt
        return CapturePromptIds(return_tensors)


class CapturePromptIds:
    def __init__(self, tensor_type):
        self.tensor_type = tensor_type
        self.device = None

    def to(self, device):
        self.device = device
        return self


class CapturePipeline:
    def __init__(self):
        self.tokenizer = CaptureTokenizer()
        self.generate_kwargs = None
        self.model = SimpleNamespace(device="test-device")

    def __call__(self, audio, *, return_timestamps, generate_kwargs):
        self.generate_kwargs = generate_kwargs
        return {"text": "ok", "chunks": []}


class HintContractTests(unittest.TestCase):
    def test_hints_are_trimmed_deduplicated_and_reported_without_values(self):
        hints = AsrHints((" TERM_A ", "term_a", "# comment", "TERM_B"), context=" topic ")
        self.assertEqual(hints.hotwords, ("TERM_A", "TERM_B"))
        self.assertEqual(hints.context, "topic")
        metadata = hints.private_metadata("test")
        self.assertEqual(metadata["hotword_count"], 2)
        self.assertNotIn("TERM_A", str(metadata))

    def test_invalid_hint_limits_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "one phrase per item"):
            AsrHints(("TERM_A\nTERM_B",))
        with self.assertRaisesRegex(ValueError, "boost must be positive"):
            AsrHints(("TERM_A",), boost=0)

    def test_prompt_compilers_use_backend_specific_semantics(self):
        hints = AsrHints(("TERM_A",), context="topic")
        glm_prompt = glm_transcription_prompt(hints)
        whisper_prompt = whisper_initial_prompt(hints)
        self.assertIn("acoustically supported", glm_prompt)
        self.assertEqual(whisper_prompt, "topic\nTERM_A")

    def test_registry_rejects_unsupported_context_before_constructor(self):
        class Unsupported:
            capabilities = BackendCapabilities()

            def __init__(self, config):
                raise AssertionError("constructor must not run")

        with patch("turnalign.registry.load", return_value=Unsupported):
            with self.assertRaisesRegex(ValueError, "does not support context"):
                create_asr("unsupported", AsrConfig(hints=AsrHints(context="topic")))


class BackendHintMappingTests(unittest.TestCase):
    def test_faster_whisper_receives_native_hotwords_and_initial_prompt(self):
        segment = SimpleNamespace(text="ok", start=0.0, end=0.2, words=[])
        info = SimpleNamespace(language="en")
        backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
        backend.language = "en"
        backend.hints = AsrHints(("TERM_A", "TERM_B"), context="topic")
        backend.model = CaptureModel(([segment], info))
        result = list(backend.transcribe([chunk()]))
        self.assertEqual(backend.model.options["hotwords"], "TERM_A TERM_B")
        self.assertEqual(backend.model.options["initial_prompt"], "topic")
        self.assertNotIn("TERM_A", str(result[0].metadata))

    def test_funasr_receives_native_hotword_string(self):
        backend = FunAsrBackend.__new__(FunAsrBackend)
        backend.language = "en"
        backend.hints = AsrHints(("TERM_A", "TERM_B"))
        backend.model = CaptureModel([{"text": "ok"}])
        result = list(backend.transcribe([chunk()]))
        self.assertEqual(backend.model.options["hotword"], "TERM_A TERM_B")
        self.assertNotIn("TERM_A", str(result[0].metadata))

    def test_transformers_whisper_receives_prompt_ids(self):
        backend = TransformersWhisperBackend.__new__(TransformersWhisperBackend)
        backend.language = "en"
        backend.hints = AsrHints(("TERM_A",), context="topic")
        backend.pipe = CapturePipeline()
        result = list(backend.transcribe([chunk()]))
        prompt_ids = backend.pipe.generate_kwargs["prompt_ids"]
        self.assertEqual(prompt_ids.tensor_type, "pt")
        self.assertEqual(prompt_ids.device, "test-device")
        self.assertEqual(backend.pipe.tokenizer.prompt, "topic\nTERM_A")
        self.assertNotIn("TERM_A", str(result[0].metadata))


if __name__ == "__main__":
    unittest.main()
