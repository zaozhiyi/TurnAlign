import unittest

from turnalign.fusion import assign_speakers
from turnalign.backends.jsonl import JsonlBackend
from turnalign.models import AudioChunk, SpeakerTurn, TranscriptEvent, Word
from turnalign.plugins import AsrBackend
from turnalign.stabilizer import LocalAgreement, common_prefix
from turnalign.registry import discover
from turnalign.validation import EventStreamValidator


class StabilizerTests(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_prefix("你好世界", "你好呀"), "你好")

    def test_local_agreement_commits_stable_prefix(self):
        state = LocalAgreement(min_confirmations=2)
        first = state.update("今天我们讨论产品")
        self.assertEqual((first.committed_delta, first.partial), ("", "今天我们讨论产品"))
        result = state.update("今天我们讨论产品经理")
        self.assertEqual(result.committed_delta, "今天我们讨论产品")
        self.assertEqual(result.partial, "经理")

    def test_three_confirmations_work_with_growing_text(self):
        state = LocalAgreement(min_confirmations=3)
        state.update("产品")
        state.update("产品经理")
        result = state.update("产品经理讨论")
        self.assertEqual(result.committed_delta, "产品")

    def test_final_flushes_tail(self):
        state = LocalAgreement()
        state.update("测试文本")
        result = state.update("测试文本完成", final=True)
        self.assertEqual(result.committed_delta, "测试文本完成")
        self.assertEqual(result.partial, "")

    def test_final_correction_requests_replace(self):
        state = LocalAgreement()
        state.update("豆包产品")
        state.update("豆包产品")
        result = state.update("豆包的产品", final=True)
        self.assertEqual(result.replace, "豆包的产品")

    def test_invalid_confirmation_count(self):
        with self.assertRaises(ValueError):
            LocalAgreement(min_confirmations=1)


class FusionTests(unittest.TestCase):
    def test_word_uses_largest_overlap(self):
        words = [Word("你好", 1.0, 1.8)]
        turns = [SpeakerTurn(0.0, 1.2, "A"), SpeakerTurn(1.2, 2.0, "B")]
        self.assertEqual(assign_speakers(words, turns)[0].speaker, "B")

    def test_confidence_breaks_overlap_tie(self):
        words = [Word("好", 1.0, 2.0)]
        turns = [SpeakerTurn(1.0, 2.0, "A", 0.4), SpeakerTurn(1.0, 2.0, "B", 0.9)]
        self.assertEqual(assign_speakers(words, turns)[0].speaker, "B")


class ModelTests(unittest.TestCase):
    def test_pcm_must_have_complete_frames(self):
        with self.assertRaises(ValueError):
            AudioChunk(b"\x00", start=0)

    def test_invalid_time_range_fails(self):
        with self.assertRaises(ValueError):
            Word("x", 2.0, 1.0)

    def test_non_finite_time_fails(self):
        with self.assertRaises(ValueError):
            Word("x", float("nan"), 1.0)


class ValidationTests(unittest.TestCase):
    def test_commit_replace_end_sequence(self):
        validator = EventStreamValidator()
        validator.accept(TranscriptEvent("commit", "s1", 1, 0, 1, "a"))
        validator.accept(TranscriptEvent("replace", "s1", 2, 0, 1, "b"))
        validator.accept(TranscriptEvent("end", "session", 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "after session end"):
            validator.accept(TranscriptEvent("commit", "s2", 1, 1, 2, "c"))

    def test_replace_unknown_segment_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown segment"):
            EventStreamValidator().accept(TranscriptEvent("replace", "missing", 1, 0, 1))

    def test_out_of_order_commits_fail(self):
        validator = EventStreamValidator()
        validator.accept(TranscriptEvent("commit", "later", 1, 2, 3))
        with self.assertRaisesRegex(ValueError, "not chronological"):
            validator.accept(TranscriptEvent("commit", "earlier", 1, 1, 2))

    def test_early_end_fails(self):
        validator = EventStreamValidator()
        validator.accept(TranscriptEvent("commit", "s", 1, 0, 3))
        with self.assertRaisesRegex(ValueError, "precedes"):
            validator.accept(TranscriptEvent("end", "session", 1, 2, 2))


class RegistryTests(unittest.TestCase):
    def test_unknown_kind_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "unknown plugin kind"):
            discover("unknown")

    def test_jsonl_backend_satisfies_asr_protocol(self):
        self.assertIsInstance(JsonlBackend("unused.jsonl"), AsrBackend)


if __name__ == "__main__":
    unittest.main()
