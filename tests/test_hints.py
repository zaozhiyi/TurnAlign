import sys
import unittest
from array import array
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from turnalign.backends.faster_whisper import FasterWhisperBackend
from turnalign.backends.funasr import FunAsrBackend
from turnalign.backends.transformers import GlmAsrBackend, TransformersWhisperBackend
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
        with self.assertRaisesRegex(ValueError, "boost must be finite"):
            AsrHints(("TERM_A",), boost=0)
        for boost in (True, "2", float("nan"), float("inf"), 1_001):
            with self.subTest(boost=boost), self.assertRaises((TypeError, ValueError)):
                AsrHints(boost=boost)
        for context in (1, {}, [], b"topic"):
            with self.subTest(context=context), self.assertRaises(TypeError):
                AsrHints(context=context)

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

        with (
            patch("turnalign.registry.load", return_value=Unsupported),
            self.assertRaisesRegex(ValueError, "does not support context"),
        ):
            create_asr("unsupported", AsrConfig(hints=AsrHints(context="topic")))


class BackendHintMappingTests(unittest.TestCase):
    def test_default_batch_funasr_model_uses_immutable_revision(self):
        captured = {}

        class AutoModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = ModuleType("funasr")
        module.AutoModel = AutoModel
        with patch.dict(sys.modules, {"funasr": module}):
            backend = FunAsrBackend(AsrConfig(device="cpu"))
        self.assertRegex(backend.model_revision, r"^[0-9a-f]{40}$")
        self.assertEqual(captured["model_revision"], backend.model_revision)
        self.assertTrue(captured["disable_update"])

    def test_default_transformers_models_use_immutable_revisions(self):
        whisper_call = {}
        whisper_module = ModuleType("transformers")

        def fake_pipeline(*_args, **kwargs):
            whisper_call.update(kwargs)
            return CapturePipeline()

        whisper_module.pipeline = fake_pipeline
        with patch.dict(sys.modules, {"transformers": whisper_module}):
            whisper = TransformersWhisperBackend(AsrConfig(device="cpu"))
        self.assertRegex(whisper.model_revision, r"^[0-9a-f]{40}$")
        self.assertEqual(whisper_call["revision"], whisper.model_revision)

        processor_call = {}

        class ProcessorLoader:
            @classmethod
            def from_pretrained(cls, _model_id, **kwargs):
                processor_call.update(kwargs)
                return object()

        class LoadedModel:
            def to(self, _device):
                return self

            def eval(self):
                return None

        class ModelLoader:
            @classmethod
            def from_pretrained(cls, _model_id, **_kwargs):
                return LoadedModel()

        glm_module = ModuleType("transformers")
        glm_module.AutoProcessor = ProcessorLoader
        glm_module.AutoModelForSeq2SeqLM = ModelLoader
        with patch.dict(sys.modules, {"transformers": glm_module}):
            glm = GlmAsrBackend(AsrConfig(device="cpu"))
        self.assertRegex(glm.model_revision, r"^[0-9a-f]{40}$")
        self.assertEqual(processor_call["revision"], glm.model_revision)

    def test_transformers_backends_forward_one_explicit_model_revision(self):
        revision = "a" * 40
        whisper_call = {}

        def fake_pipeline(*args, **kwargs):
            whisper_call.update({"args": args, "kwargs": kwargs})
            return CapturePipeline()

        whisper_module = ModuleType("transformers")
        whisper_module.pipeline = fake_pipeline
        with patch.dict(sys.modules, {"transformers": whisper_module}):
            whisper = TransformersWhisperBackend(AsrConfig(
                device="cpu",
                extra={"revision": revision, "local_files_only": True},
            ))
        self.assertEqual(whisper.model_revision, revision)
        self.assertEqual(whisper_call["kwargs"]["revision"], revision)
        self.assertTrue(whisper_call["kwargs"]["local_files_only"])

        processor_call = {}
        model_call = {}

        class ProcessorLoader:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                processor_call.update({"model_id": model_id, **kwargs})
                return object()

        class LoadedModel:
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                return None

        class ModelLoader:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                model_call.update({"model_id": model_id, **kwargs})
                return LoadedModel()

        glm_module = ModuleType("transformers")
        glm_module.AutoProcessor = ProcessorLoader
        glm_module.AutoModelForSeq2SeqLM = ModelLoader
        with patch.dict(sys.modules, {"transformers": glm_module}):
            glm = GlmAsrBackend(AsrConfig(
                device="cpu",
                extra={"revision": revision, "local_files_only": True},
            ))
        self.assertEqual(glm.model_revision, revision)
        self.assertEqual(processor_call["revision"], revision)
        self.assertTrue(processor_call["local_files_only"])
        self.assertEqual(model_call["revision"], revision)
        self.assertTrue(model_call["local_files_only"])

    def test_transformers_revision_must_be_a_bounded_nonempty_string(self):
        module = ModuleType("transformers")
        module.pipeline = lambda *_args, **_kwargs: CapturePipeline()
        for revision in ("", " ", 1, "x" * 201):
            with self.subTest(revision=revision), patch.dict(
                sys.modules,
                {"transformers": module},
            ), self.assertRaisesRegex(ValueError, "revision"):
                TransformersWhisperBackend(AsrConfig(extra={"revision": revision}))

    def test_faster_whisper_receives_native_hotwords_and_initial_prompt(self):
        segment = SimpleNamespace(text="ok", start=0.0, end=0.2, words=[])
        info = SimpleNamespace(language="en")
        backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
        backend.language = "en"
        backend.hints = AsrHints(("TERM_A", "TERM_B"), context="topic")
        backend.model = CaptureModel(([segment], info))
        with patch("turnalign.backends.faster_whisper.pcm_to_float32", return_value=[0.0]):
            result = list(backend.transcribe([chunk()]))
        self.assertEqual(backend.model.options["hotwords"], "TERM_A TERM_B")
        self.assertEqual(backend.model.options["initial_prompt"], "topic")
        self.assertNotIn("TERM_A", str(result[0].metadata))

    def test_funasr_receives_native_hotword_string(self):
        backend = FunAsrBackend.__new__(FunAsrBackend)
        backend.language = "en"
        backend.hints = AsrHints(("TERM_A", "TERM_B"))
        backend.model = CaptureModel([{"text": "ok"}])
        with patch("turnalign.backends.funasr.pcm_to_float32", return_value=[0.0]):
            result = list(backend.transcribe([chunk()]))
        self.assertEqual(backend.model.options["hotword"], "TERM_A TERM_B")
        self.assertNotIn("TERM_A", str(result[0].metadata))

    def test_transformers_whisper_receives_prompt_ids(self):
        backend = TransformersWhisperBackend.__new__(TransformersWhisperBackend)
        backend.language = "en"
        backend.hints = AsrHints(("TERM_A",), context="topic")
        backend.pipe = CapturePipeline()
        with patch("turnalign.backends.transformers.pcm_to_float32", return_value=[0.0]):
            result = list(backend.transcribe([chunk()]))
        prompt_ids = backend.pipe.generate_kwargs["prompt_ids"]
        self.assertEqual(prompt_ids.tensor_type, "pt")
        self.assertEqual(prompt_ids.device, "test-device")
        self.assertEqual(backend.pipe.tokenizer.prompt, "topic\nTERM_A")
        self.assertNotIn("TERM_A", str(result[0].metadata))


if __name__ == "__main__":
    unittest.main()
