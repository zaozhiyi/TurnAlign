import sys
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from turnalign.backends.funasr_streaming import FunAsrStreamingBackend
from turnalign.models import AudioChunk
from turnalign.plugins import AsrConfig


class FakeFunAsrModel:
    def __init__(self):
        self.calls = []

    def generate(self, **options):
        self.calls.append(options)
        return [{"text": f"result-{len(self.calls)}"}]


class EmptyFinalFunAsrModel(FakeFunAsrModel):
    def generate(self, **options):
        self.calls.append(options)
        if options["is_final"]:
            return [{"text": ""}]
        return [{"text": "partial"}]


class FunAsrStreamingTests(unittest.TestCase):
    def test_default_model_is_pinned_and_update_check_is_disabled(self):
        captured = {}

        def auto_model(**options):
            captured.update(options)
            return FakeFunAsrModel()

        with patch.dict(sys.modules, {"funasr": SimpleNamespace(AutoModel=auto_model)}):
            FunAsrStreamingBackend(AsrConfig(device="cpu"))

        self.assertEqual(
            captured["model_revision"],
            "562b758fecc801f13079d846d06b0b024fd670c4",
        )
        self.assertTrue(captured["disable_update"])

    def test_each_generate_call_receives_only_the_new_audio_chunk(self):
        model = FakeFunAsrModel()
        fake_module = SimpleNamespace(AutoModel=lambda **_options: model)
        with (
            patch.dict(sys.modules, {"funasr": fake_module}),
            patch(
                "turnalign.backends.funasr_streaming.pcm_to_float32",
                side_effect=lambda data, _channels: len(data),
            ),
        ):
            backend = FunAsrStreamingBackend(AsrConfig(
                device="cpu",
                extra={"chunk_ms": 100},
            ))
            chunks = [
                AudioChunk(array("h", [100] * 1_600).tobytes(), index * 0.1)
                for index in range(3)
            ]
            events = list(backend.transcribe(chunks))

        self.assertEqual([call["input"] for call in model.calls], [3_200, 3_200, 3_200, 0])
        self.assertFalse(any(call["is_final"] for call in model.calls[:-1]))
        self.assertTrue(model.calls[-1]["is_final"])
        self.assertTrue(all(call["cache"] is model.calls[0]["cache"] for call in model.calls))
        self.assertEqual(len(events), 4)
        self.assertEqual(events[1].text, "result-1 result-2")
        self.assertTrue(events[-1].final)

    def test_short_pending_tail_is_flushed_with_real_audio(self):
        model = FakeFunAsrModel()
        fake_module = SimpleNamespace(AutoModel=lambda **_options: model)
        with (
            patch.dict(sys.modules, {"funasr": fake_module}),
            patch(
                "turnalign.backends.funasr_streaming.pcm_to_float32",
                side_effect=lambda data, _channels: len(data),
            ),
        ):
            backend = FunAsrStreamingBackend(AsrConfig(
                device="cpu",
                extra={"chunk_ms": 100},
            ))
            short = AudioChunk(array("h", [100] * 800).tobytes(), 0)
            events = list(backend.transcribe([short]))
        self.assertEqual(model.calls[0]["input"], 1_600)
        self.assertTrue(model.calls[0]["is_final"])
        self.assertTrue(events[0].final)

    def test_empty_final_flush_promotes_last_partial_to_commit(self):
        model = EmptyFinalFunAsrModel()
        fake_module = SimpleNamespace(AutoModel=lambda **_options: model)
        with (
            patch.dict(sys.modules, {"funasr": fake_module}),
            patch(
                "turnalign.backends.funasr_streaming.pcm_to_float32",
                side_effect=lambda data, _channels: len(data),
            ),
        ):
            backend = FunAsrStreamingBackend(AsrConfig(
                device="cpu",
                extra={"chunk_ms": 100},
            ))
            aligned = AudioChunk(array("h", [100] * 1_600).tobytes(), 0)
            events = list(backend.transcribe([aligned]))

        self.assertEqual([event.final for event in events], [False, True])
        self.assertEqual(events[-1].text, "partial")
        self.assertTrue(events[-1].metadata["synthetic_final_flush"])

    def test_empty_short_final_chunk_promotes_last_partial_to_commit(self):
        model = EmptyFinalFunAsrModel()
        fake_module = SimpleNamespace(AutoModel=lambda **_options: model)
        with (
            patch.dict(sys.modules, {"funasr": fake_module}),
            patch(
                "turnalign.backends.funasr_streaming.pcm_to_float32",
                side_effect=lambda data, _channels: len(data),
            ),
        ):
            backend = FunAsrStreamingBackend(AsrConfig(
                device="cpu",
                extra={"chunk_ms": 100},
            ))
            aligned = AudioChunk(array("h", [100] * 1_600).tobytes(), 0)
            tail = AudioChunk(
                array("h", [100] * 800).tobytes(),
                0.1,
                is_final=True,
            )
            events = list(backend.transcribe([aligned, tail]))

        self.assertEqual([event.final for event in events], [False, True])
        self.assertEqual(events[-1].text, "partial")
        self.assertTrue(events[-1].metadata["synthetic_final_flush"])


if __name__ == "__main__":
    unittest.main()
