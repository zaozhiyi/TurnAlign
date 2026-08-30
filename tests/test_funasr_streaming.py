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


class FunAsrStreamingTests(unittest.TestCase):
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

        self.assertEqual([call["input"] for call in model.calls], [3_200, 3_200, 3_200])
        self.assertFalse(any(call["is_final"] for call in model.calls))
        self.assertTrue(all(call["cache"] is model.calls[0]["cache"] for call in model.calls))
        self.assertEqual(len(events), 4)
        self.assertEqual(events[1].text, "result-1 result-2")
        self.assertTrue(events[-1].final)
        self.assertTrue(events[-1].metadata["synthetic_final_flush"])

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


if __name__ == "__main__":
    unittest.main()
