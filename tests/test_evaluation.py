import unittest

from turnalign.evaluation import (
    TextNormalization,
    character_error_rate,
    diarization_error_rate,
    evaluate_events,
    evaluate_quality_gate,
    word_error_rate,
)
from turnalign.models import TranscriptEvent


class EvaluationTests(unittest.TestCase):
    def test_character_and_word_error_rates(self):
        self.assertEqual(character_error_rate("你好 世界", "你好世界"), 0)
        self.assertAlmostEqual(word_error_rate("one two", "one three"), 0.5)

    def test_text_normalization_is_explicit_and_reported(self):
        reference = [TranscriptEvent("commit", "r", 1, 0, 1, "Ａ，Test")]
        hypothesis = [TranscriptEvent("commit", "h", 1, 0, 1, "a test")]
        self.assertGreater(
            evaluate_events(reference, hypothesis).character_error_rate,
            0,
        )
        policy = TextNormalization(
            unicode_form="NFKC",
            case_sensitive=False,
            punctuation_sensitive=False,
        )
        report = evaluate_events(
            reference,
            hypothesis,
            text_normalization=policy,
        )
        self.assertEqual(report.character_error_rate, 0)
        self.assertEqual(report.word_error_rate, 0)
        self.assertEqual(report.text_normalization, policy)
        self.assertEqual(report.to_dict()["text_normalization"]["unicode_form"], "NFKC")

    def test_text_normalization_rejects_ambiguous_configuration(self):
        with self.assertRaisesRegex(ValueError, "unicode_form"):
            TextNormalization(unicode_form="NFKD")
        with self.assertRaisesRegex(TypeError, "case_sensitive"):
            TextNormalization(case_sensitive="false")

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

    def test_diarization_counts_false_alarm_during_reference_silence(self):
        reference = [
            TranscriptEvent("commit", "r", 1, 0, 5, "a", speaker="Alice"),
        ]
        hypothesis = [
            TranscriptEvent("commit", "h1", 1, 0, 2, "a", speaker="speaker-1"),
            TranscriptEvent("commit", "h2", 1, 5, 10, "b", speaker="speaker-2"),
        ]
        # Three seconds missed plus five seconds of false alarm, normalized by
        # five seconds of reference speech.
        self.assertAlmostEqual(diarization_error_rate(reference, hypothesis), 1.6)

    def test_diarization_maps_fewer_hypothesis_speakers_without_overcounting(self):
        reference = [
            TranscriptEvent("commit", "r1", 1, 0, 2, "a", speaker="Alice"),
            TranscriptEvent("commit", "r2", 1, 2, 4, "b", speaker="Bob"),
            TranscriptEvent("commit", "r3", 1, 4, 6, "c", speaker="Carol"),
        ]
        hypothesis = [
            TranscriptEvent("commit", "h1", 1, 0, 2, "a", speaker="speaker-1"),
            TranscriptEvent("commit", "h2", 1, 2, 4, "b", speaker="speaker-2"),
        ]
        self.assertAlmostEqual(diarization_error_rate(reference, hypothesis), 1 / 3)

    def test_false_alarm_duration_does_not_distort_speaker_mapping(self):
        reference = [
            TranscriptEvent("commit", "r", 1, 0, 10, "a", speaker="Alice"),
        ]
        hypothesis = [
            TranscriptEvent("commit", "h1", 1, 0, 10, "a", speaker="speaker-1"),
            TranscriptEvent("commit", "h2", 1, 10, 11, "b", speaker="speaker-2"),
            TranscriptEvent("commit", "h3", 1, 11, 111, "c", speaker="speaker-1"),
        ]
        # Correct speech is mapped first; 101 seconds of false alarm are then
        # counted as error instead of rewarding a speaker-to-silence mapping.
        self.assertAlmostEqual(diarization_error_rate(reference, hypothesis), 10.1)

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
        self.assertEqual(report.reference_characters, 10)
        self.assertEqual(report.reference_words, 2)
        self.assertEqual(report.reference_speech_seconds, 1)

    def test_quality_gate_enforces_accuracy_stability_and_sample_scale(self):
        reference = [
            TranscriptEvent("commit", "r1", 1, 0, 1, "one two", speaker="Alice"),
            TranscriptEvent("commit", "r2", 1, 1, 2, "three", speaker="Bob"),
        ]
        hypothesis = [
            TranscriptEvent("commit", "h1", 1, 0, 1, "one two", speaker="speaker-1"),
            TranscriptEvent("commit", "h2", 1, 1, 2, "three", speaker="speaker-2"),
        ]
        passed = evaluate_quality_gate(
            reference,
            hypothesis,
            max_character_error_rate=0,
            max_word_error_rate=0,
            max_diarization_error_rate=0,
            max_revision_updates_per_segment=0,
            min_reference_segments=2,
            min_reference_characters=11,
            min_reference_speech_seconds=2,
        )
        self.assertTrue(passed.passed)
        self.assertEqual(passed.status, "passed")
        self.assertEqual(passed.evaluation.reference_speakers, 2)

        failed = evaluate_quality_gate(
            reference,
            [TranscriptEvent("commit", "h", 1, 0, 1, "wrong")],
            max_character_error_rate=0.1,
            max_word_error_rate=0.1,
            max_diarization_error_rate=0.1,
            max_revision_updates_per_segment=0,
            min_reference_segments=3,
            min_reference_speech_seconds=3,
        )
        self.assertFalse(failed.passed)
        self.assertGreaterEqual(len(failed.failures), 4)
        self.assertTrue(any(
            "diarization error rate" in item and "exceeds" in item
            for item in failed.failures
        ))

    def test_quality_gate_reports_one_immutable_hypothesis_model_revision(self):
        revision = "a" * 40
        reference = [TranscriptEvent("commit", "r", 1, 0, 1, "text")]
        hypothesis = [
            TranscriptEvent(
                "commit",
                "h",
                1,
                0,
                1,
                "text",
                metadata={"model_revision": revision},
            )
        ]

        report = evaluate_quality_gate(
            reference,
            hypothesis,
            max_character_error_rate=0,
        )

        self.assertEqual(report.model_revision, revision)

    def test_quality_gate_rejects_meaningless_or_invalid_thresholds(self):
        event = [TranscriptEvent("commit", "s", 1, 0, 1, "text")]
        with self.assertRaisesRegex(ValueError, "at least one maximum metric"):
            evaluate_quality_gate(event, event)
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
            evaluate_quality_gate(
                event,
                event,
                max_character_error_rate=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            evaluate_quality_gate(
                event,
                event,
                max_character_error_rate=0,
                min_reference_segments=True,
            )
        with self.assertRaisesRegex(ValueError, "non-negative and finite"):
            evaluate_quality_gate(
                event,
                event,
                max_character_error_rate=True,
            )

    def test_quality_gate_requires_speaker_labels_for_diarization_ceiling(self):
        event = [TranscriptEvent("commit", "s", 1, 0, 1, "text")]
        report = evaluate_quality_gate(
            event,
            event,
            max_diarization_error_rate=0,
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("unavailable" in item for item in report.failures))


if __name__ == "__main__":
    unittest.main()
