import unittest
from array import array
from unittest.mock import patch

from turnalign.models import AudioChunk, Hypothesis
from turnalign.plugins import Accelerator, BackendCapabilities
from turnalign.release_gate import run_release_gate


class StreamingGateBackend:
    name = "streaming-gate"
    capabilities = BackendCapabilities(
        streaming=True,
        accelerators=(Accelerator.CPU,),
    )

    def __init__(self):
        self.closed = False

    def transcribe(self, chunks):
        items = list(chunks)
        if not items:
            return
        start = items[0].start
        end = items[-1].start + items[-1].duration
        yield Hypothesis("draft", start, end, final=False)
        yield Hypothesis("final", start, end, final=True)

    def close(self):
        self.closed = True


class BatchGateBackend(StreamingGateBackend):
    name = "batch-gate"
    capabilities = BackendCapabilities(accelerators=(Accelerator.CPU,))

    def transcribe(self, chunks):
        items = list(chunks)
        if items:
            yield Hypothesis(
                "final",
                items[0].start,
                items[-1].start + items[-1].duration,
                final=True,
            )


class InvalidProtocolBackend(StreamingGateBackend):
    def transcribe(self, chunks):
        items = list(chunks)
        if items:
            yield Hypothesis("draft", 0, items[-1].duration, final=False)


class FailingGateBackend(StreamingGateBackend):
    def transcribe(self, chunks):
        list(chunks)
        raise RuntimeError("inference failed")
        yield  # pragma: no cover - keeps this method a generator


class PinnedStreamingGateBackend(StreamingGateBackend):
    model_revision = "a" * 40


def audio(duration: float = 0.2) -> list[AudioChunk]:
    pcm = array("h", [1_500] * round(16_000 * duration)).tobytes()
    return [AudioChunk(pcm, 0.0)]


class ReleaseGateTests(unittest.TestCase):
    def test_native_streaming_backend_passes_protocol_latency_and_rtf_gate(self):
        captured = []
        backend = StreamingGateBackend()
        report = run_release_gate(
            audio(),
            backend,
            max_realtime_factor=10,
            max_first_commit_seconds=1,
            min_audio_seconds=0.1,
            event_sink=captured.append,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.partials, 1)
        self.assertEqual(report.commits, 1)
        self.assertIsNotNone(report.first_commit_seconds)
        self.assertEqual(report.max_realtime_factor, 10)
        self.assertEqual(report.max_first_commit_seconds, 1)
        self.assertEqual(report.min_audio_seconds, 0.1)
        self.assertTrue(report.require_partial)
        self.assertTrue(report.require_native_streaming)
        self.assertEqual(captured[-1].kind, "end")
        self.assertTrue(backend.closed)

    def test_first_commit_latency_threshold_is_enforced(self):
        with patch(
            "turnalign.release_gate.perf_counter",
            side_effect=[0.0, 0.1, 0.8, 0.9],
        ):
            report = run_release_gate(
                audio(),
                StreamingGateBackend(),
                max_realtime_factor=10,
                max_first_commit_seconds=0.5,
                min_audio_seconds=0.1,
            )
        self.assertFalse(report.passed)
        self.assertEqual(report.first_commit_seconds, 0.8)
        self.assertTrue(any("first commit latency" in item for item in report.failures))

    def test_immutable_model_revision_can_be_required(self):
        missing = run_release_gate(
            audio(),
            StreamingGateBackend(),
            max_realtime_factor=10,
            min_audio_seconds=0.1,
            require_immutable_model_revision=True,
        )
        self.assertFalse(missing.passed)
        self.assertTrue(any("not pinned" in item for item in missing.failures))

        captured = []
        pinned = run_release_gate(
            audio(),
            PinnedStreamingGateBackend(),
            max_realtime_factor=10,
            min_audio_seconds=0.1,
            require_immutable_model_revision=True,
            event_sink=captured.append,
        )
        self.assertTrue(pinned.passed)
        self.assertEqual(pinned.model_revision, "a" * 40)
        self.assertTrue(pinned.require_immutable_model_revision)
        self.assertEqual(captured[-1].metadata["model_revision"], "a" * 40)

    def test_batch_backend_fails_native_streaming_and_partial_requirements(self):
        report = run_release_gate(
            audio(),
            BatchGateBackend(),
            max_realtime_factor=10,
            min_audio_seconds=0.1,
        )
        self.assertFalse(report.passed)
        self.assertIn("backend does not declare native streaming", report.failures)
        self.assertIn("no partial event was emitted", report.failures)

    def test_thresholds_must_be_positive(self):
        for options in (
            {"max_realtime_factor": 0},
            {"max_first_partial_seconds": 0},
            {"max_first_commit_seconds": 0},
            {"max_initialization_seconds": 0},
            {"initialization_seconds": -1},
            {"min_audio_seconds": 0},
            {"min_commits": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_release_gate(audio(), StreamingGateBackend(), **options)

        for name in (
            "max_realtime_factor",
            "max_first_partial_seconds",
            "max_first_commit_seconds",
            "max_initialization_seconds",
            "initialization_seconds",
            "min_audio_seconds",
        ):
            for value in (float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    run_release_gate(
                        audio(), StreamingGateBackend(), **{name: value}
                    )

    def test_protocol_violation_is_reported_as_gate_failure(self):
        report = run_release_gate(
            audio(),
            InvalidProtocolBackend(),
            max_realtime_factor=10,
            min_audio_seconds=0.1,
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("invalid event stream" in item for item in report.failures))

    def test_backend_exception_is_reported_as_gate_failure(self):
        report = run_release_gate(
            audio(),
            FailingGateBackend(),
            max_realtime_factor=10,
            min_audio_seconds=0.1,
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("backend execution failed" in item for item in report.failures))


if __name__ == "__main__":
    unittest.main()
