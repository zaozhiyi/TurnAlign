import unittest
from array import array
from itertools import pairwise
from unittest.mock import patch

from turnalign.components.energy_vad import EnergyVadBackend
from turnalign.components.funasr import (
    CamppDiarizationBackend,
    FsmnVadBackend,
    ParaformerAlignmentBackend,
    _aligned_words,
    _materialization_limit,
)
from turnalign.models import AudioChunk, Hypothesis, SpeechSegment
from turnalign.plugins import Accelerator, BackendCapabilities
from turnalign.registry import available
from turnalign.session import transcribe_events


def chunk(amplitude: int, start: float, duration: float = 0.1) -> AudioChunk:
    samples = array("h", [amplitude] * round(16_000 * duration))
    return AudioChunk(samples.tobytes(), start)


class FakeBatchBackend:
    name = "fake-batch"
    capabilities = BackendCapabilities(accelerators=(Accelerator.CPU,))

    def transcribe(self, chunks):
        items = list(chunks)
        if items:
            yield Hypothesis("你好", items[0].start, items[-1].start + items[-1].duration)

    def close(self):
        return None


class FakeVad:
    name = "fake-vad"

    def segment(self, chunks):
        items = list(chunks)
        yield SpeechSegment(items[1:3], 1.0, 3.0, confidence=0.9, forced_split=True)

    def close(self):
        return None


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.options = None

    def generate(self, **kwargs):
        self.options = kwargs
        return self.result


class EnergyVadTests(unittest.TestCase):
    def test_soft_speech_survives_adaptive_threshold_and_keeps_preroll(self):
        source = [chunk(0, 0.0), chunk(0, 0.1)]
        source.extend(chunk(300, 0.2 + index * 0.1) for index in range(5))
        source.extend(chunk(0, 0.7 + index * 0.1) for index in range(7))
        segments = list(EnergyVadBackend(min_silence_seconds=0.6).segment(source))
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertGreaterEqual(segments[0].end, 1.2)

    def test_continuous_speech_records_forced_split(self):
        source = [chunk(2000, index * 0.1) for index in range(12)]
        segments = list(EnergyVadBackend(max_segment_seconds=0.5).segment(source))
        self.assertEqual(len(segments), 3)
        self.assertTrue(all(segment.forced_split for segment in segments[:2]))
        self.assertFalse(segments[-1].forced_split)


class VadAuditTests(unittest.TestCase):
    def test_audit_covers_speech_and_skipped_intervals(self):
        audit = []
        source = [chunk(1000, index, 1.0) for index in range(4)]
        events = list(transcribe_events(
            source,
            FakeBatchBackend(),
            vad_backend=FakeVad(),
            vad_audit=audit.append,
        ))
        self.assertEqual([item["decision"] for item in audit], ["silence", "speech", "silence"])
        self.assertEqual(events[-1].metadata["vad_speech_seconds"], 2.0)
        self.assertEqual(events[-1].metadata["vad_skipped_seconds"], 2.0)
        self.assertEqual(events[-1].metadata["vad_forced_splits"], 1)


class FunAsrComponentTests(unittest.TestCase):
    def test_materialization_limit_must_be_finite(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _materialization_limit(value)

    def test_fsmn_regions_are_split_and_keep_original_timestamps(self):
        backend = FsmnVadBackend.__new__(FsmnVadBackend)
        backend.model = FakeModel([{"value": [[1000, 4500]]}])
        backend.model_id = "fake-fsmn"
        backend.batch_size_s = 300
        backend.max_segment_seconds = 2.0
        backend.max_materialized_seconds = 7_200.0
        source = [chunk(1000, 10.0 + index, 1.0) for index in range(5)]
        with patch("turnalign.components.funasr.pcm_to_float32", return_value=[0.0]):
            segments = list(backend.segment(source))
        self.assertEqual([(item.start, item.end) for item in segments], [(11.0, 13.0), (13.0, 14.5)])
        self.assertTrue(segments[0].forced_split)
        self.assertFalse(segments[1].forced_split)

    def test_paraformer_alignment_is_monotonic_and_preserves_target_text(self):
        words = _aligned_words(
            "你好，世界",
            "你 好 世 界",
            [[0, 100], [100, 200], [300, 400], [400, 500]],
            offset=7.0,
            duration=0.5,
        )
        self.assertEqual("".join(word.text for word in words), "你好，世界")
        self.assertTrue(all(left.end <= right.start for left, right in pairwise(words)))
        self.assertGreaterEqual(words[0].start, 7.0)
        self.assertLessEqual(words[-1].end, 7.5)

    def test_paraformer_batches_multiple_alignment_requests(self):
        backend = ParaformerAlignmentBackend.__new__(ParaformerAlignmentBackend)
        backend.model = FakeModel([
            {"text": "A", "timestamp": [[0, 100]]},
            {"text": "B", "timestamp": [[0, 100]]},
        ])
        backend.batch_size = 8
        backend.batch_size_s = 60
        items = [(chunk(1000, 0.0), "A"), (chunk(1000, 1.0), "B")]
        with patch("turnalign.components.funasr.pcm_to_float32", return_value=[0.0]):
            result = backend.align_many(items)
        self.assertEqual([[word.text for word in words] for words in result], [["A"], ["B"]])
        self.assertEqual(backend.model.options["batch_size"], 8)

    def test_campp_turns_are_merged_and_offset(self):
        backend = CamppDiarizationBackend.__new__(CamppDiarizationBackend)
        backend.model = FakeModel([{"sentence_info": [
            {"start": 0, "end": 1000, "spk": 0},
            {"start": 1100, "end": 2000, "spk": 0},
            {"start": 2100, "end": 3000, "spk": 1},
        ]}])
        backend.batch_size_s = 300
        backend.merge_gap_seconds = 0.2
        backend.max_materialized_seconds = 7_200.0
        with patch("turnalign.components.funasr.pcm_to_float32", return_value=[0.0]):
            turns = list(backend.diarize([chunk(1000, 5.0, 3.0)]))
        self.assertEqual([(turn.start, turn.end, turn.speaker) for turn in turns], [
            (5.0, 7.0, "speaker-1"),
            (7.1, 8.0, "speaker-2"),
        ])

    def test_full_input_components_fail_before_oversized_float_materialization(self):
        fsmn = FsmnVadBackend.__new__(FsmnVadBackend)
        fsmn.model = FakeModel([])
        fsmn.model_id = "fake-fsmn"
        fsmn.batch_size_s = 300
        fsmn.max_segment_seconds = 20.0
        fsmn.max_materialized_seconds = 1.0
        source = [chunk(1000, 0.0, 1.0), chunk(1000, 1.0, 0.1)]
        with self.assertRaisesRegex(ValueError, "max_materialized_seconds"):
            list(fsmn.segment(source))
        self.assertIsNone(fsmn.model.options)

        campp = CamppDiarizationBackend.__new__(CamppDiarizationBackend)
        campp.model = FakeModel([])
        campp.batch_size_s = 300
        campp.merge_gap_seconds = 0.2
        campp.max_materialized_seconds = 1.0
        with self.assertRaisesRegex(ValueError, "max_materialized_seconds"):
            list(campp.diarize(source))
        self.assertIsNone(campp.model.options)

    def test_first_party_components_are_discoverable(self):
        self.assertIn("fsmn-vad", available("vad"))
        self.assertIn("paraformer", available("alignment"))
        self.assertIn("campp", available("diarization"))


if __name__ == "__main__":
    unittest.main()
