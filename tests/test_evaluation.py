import unittest

from turnalign.evaluation import (
    character_error_rate,
    diarization_error_rate,
    evaluate_events,
    word_error_rate,
)
from turnalign.models import TranscriptEvent


class EvaluationTests(unittest.TestCase):
    def test_character_and_word_error_rates(self):
        self.assertEqual(character_error_rate("你好 世界", "你好世界"), 0)
        self.assertAlmostEqual(word_error_rate("one two", "one three"), 0.5)

    def test_diarization_labels_are_permutation_invariant(self):
        reference = [
            TranscriptEvent("commit", "r1", 1, 0, 1, "a", speaker="Alice"),
            TranscriptEvent("commit", "r2", 1, 1, 2, "b", speaker="Bob"),
        ]
        hypothesis = [
            TranscriptEvent("commit", "h1", 1, 0, 1, "a", speaker="speaker-2"),
            TranscriptEvent("commit", "h2", 1, 1, 2, "b", speaker="speaker-1"),
        ]
        self.assertEqual(diarization_error_rate(reference, hypothesis), 0)

    def test_report_counts_revisions_without_treating_them_as_segments(self):
        reference = [TranscriptEvent("commit", "r", 1, 0, 1, "hello world")]
        hypothesis = [
            TranscriptEvent("partial", "h", 1, 0, 0.5, "hello"),
            TranscriptEvent("commit", "h", 2, 0, 1, "hello word"),
            TranscriptEvent("replace", "h", 3, 0, 1, "hello world"),
        ]
        report = evaluate_events(reference, hypothesis)
        self.assertEqual(report.character_error_rate, 0)
        self.assertEqual(report.word_error_rate, 0)
        self.assertEqual(report.hypothesis_segments, 1)
        self.assertEqual(report.revision_updates_per_segment, 2)


if __name__ == "__main__":
    unittest.main()
