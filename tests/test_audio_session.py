import io
import subprocess
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from threading import Event
from unittest.mock import patch

from turnalign.audio import AudioTimeline, file_chunks, wave_chunks, write_wave
from turnalign.models import AudioChunk, Hypothesis, SpeakerTurn, TranscriptEvent, Word
from turnalign.offline import OfflineRefinementPipeline
from turnalign.pipelines import TwoPassPipeline
from turnalign.plugins import Accelerator, BackendCapabilities
from turnalign.realtime import RealtimePipeline
from turnalign.registry import available
from turnalign.session import live_windows, pcm_rms, transcribe_events, utterances
from turnalign.validation import EventStreamValidator


def chunk(amplitude: int, start: float, duration: float = 0.1) -> AudioChunk:
    samples = array("h", [amplitude] * round(16_000 * duration))
    return AudioChunk(samples.tobytes(), start)


class FakeBatchBackend:
    name = "fake-batch"
    capabilities = BackendCapabilities(accelerators=(Accelerator.CPU,))

    def __init__(self):
        self.closed = False
        self.calls = 0

    def transcribe(self, chunks):
        items = list(chunks)
        self.calls += 1
        if items:
            yield Hypothesis("hello", items[0].start, items[-1].start + items[-1].duration)

    def close(self):
        self.closed = True


class FakeStreamingBackend(FakeBatchBackend):
    name = "fake-streaming"
    capabilities = BackendCapabilities(streaming=True, accelerators=(Accelerator.CPU,))

    def transcribe(self, chunks):
        items = list(chunks)
        yield Hypothesis("hel", items[0].start, items[-1].start + items[-1].duration)
        yield Hypothesis("hello", items[0].start, items[-1].start + items[-1].duration, final=True)


class FakeStreamingSession:
    def __init__(self):
        self.accepted = []
        self.closed = False

    def accept_audio(self, item):
        self.accepted.append(item)
        yield Hypothesis(
            f"chunk-{len(self.accepted)}",
            self.accepted[0].start,
            item.start + item.duration,
        )

    def finish(self):
        last = self.accepted[-1]
        yield Hypothesis(
            "complete",
            self.accepted[0].start,
            last.start + last.duration,
            final=True,
        )

    def cancel(self):
        return None

    def close(self):
        self.closed = True


class FakeStatefulStreamingBackend(FakeBatchBackend):
    name = "fake-stateful-streaming"
    capabilities = BackendCapabilities(
        streaming=True,
        min_chunk_ms=20,
        accelerators=(Accelerator.CPU,),
    )

    def __init__(self):
        super().__init__()
        self.session = FakeStreamingSession()

    def start_session(self):
        return self.session

    def transcribe(self, chunks):
        raise AssertionError("stateful backend must consume incremental chunks")


class RevisingBatchBackend(FakeBatchBackend):
    def __init__(self):
        super().__init__()
        self.outputs = iter([
            "turn align",
            "turn alignment",
            "unexpected regression",
            "different final",
        ])

    def transcribe(self, chunks):
        items = list(chunks)
        self.calls += 1
        if items:
            yield Hypothesis(
                next(self.outputs),
                items[0].start,
                items[-1].start + items[-1].duration,
            )


class RefinementBackend(FakeBatchBackend):
    name = "fake-refinement"

    def transcribe(self, chunks):
        items = list(chunks)
        self.calls += 1
        if items:
            yield Hypothesis(
                "refined transcript",
                items[0].start,
                items[-1].start + items[-1].duration,
                final=True,
            )


class FailingRefinementBackend(FakeBatchBackend):
    name = "failing-refinement"

    def transcribe(self, chunks):
        list(chunks)
        raise RuntimeError("offline model failed")


class FailingRealtimeBackend(FakeBatchBackend):
    name = "failing-realtime"

    def transcribe(self, chunks):
        list(chunks)
        raise RuntimeError("realtime model failed")
        yield  # pragma: no cover - keeps this method a generator


class FakeOnlineDiarizationSession:
    def __init__(self):
        self.closed = False

    def accept_audio(self, item):
        yield SpeakerTurn(item.start, item.start + item.duration, "speaker-live")

    def finish(self):
        return ()

    def close(self):
        self.closed = True


class FakeOnlineDiarizer:
    name = "fake-online-diarizer"

    def __init__(self):
        self.session = FakeOnlineDiarizationSession()
        self.closed = False

    def start_session(self):
        return self.session

    def close(self):
        self.closed = True


class FailingOnlineDiarizer(FakeOnlineDiarizer):
    def start_session(self):
        raise RuntimeError("online session failed")


class FakeAligner:
    name = "fake-aligner"

    def align(self, audio, text):
        return [Word(text, audio.start, audio.start + audio.duration)]


class FakeBatchAligner(FakeAligner):
    name = "fake-batch-aligner"
    batch_size = 2

    def __init__(self):
        self.batch_calls = 0

    def align_many(self, items):
        self.batch_calls += 1
        return [self.align(audio, text) for audio, text in items]


class FakeDiarizer:
    name = "fake-diarizer"

    def diarize(self, chunks):
        items = list(chunks)
        yield SpeakerTurn(items[0].start, items[-1].start + items[-1].duration, "speaker-1")


class CoordinatedBackend(FakeBatchBackend):
    def __init__(self, diarizer_started):
        super().__init__()
        self.diarizer_started = diarizer_started

    def transcribe(self, chunks):
        self.assert_parallel_start()
        yield from super().transcribe(chunks)

    def assert_parallel_start(self):
        if not self.diarizer_started.wait(timeout=1):
            raise AssertionError("diarizer did not start before ASR")


class CoordinatedDiarizer(FakeDiarizer):
    def __init__(self, started):
        self.started = started

    def diarize(self, chunks):
        self.started.set()
        yield from super().diarize(chunks)


class AudioTests(unittest.TestCase):
    def test_pcm_rms_distinguishes_voice_and_silence(self):
        self.assertEqual(pcm_rms(chunk(0, 0)), 0)
        self.assertGreater(pcm_rms(chunk(2000, 0)), 0.05)

    def test_wave_round_trip_preserves_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            write_wave(path, iter([chunk(1000, 0), chunk(2000, 0.1)]))
            decoded = list(wave_chunks(path, chunk_ms=100))
            self.assertEqual(len(decoded), 2)
            self.assertAlmostEqual(decoded[-1].start, 0.1)
            with wave.open(str(path), "rb") as source:
                self.assertEqual((source.getframerate(), source.getnchannels()), (16_000, 1))

    def test_file_decoder_reaps_ffmpeg_when_consumer_stops_early(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO(bytes(64_000))
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                del timeout
                if self.returncode is None:
                    self.returncode = 0
                return self.returncode

        process = Process()
        with (
            patch("turnalign.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("turnalign.audio.subprocess.Popen", return_value=process),
        ):
            decoded = file_chunks(Path("recording.mp3"), chunk_ms=100)
            next(decoded)
            decoded.close()
        self.assertTrue(process.terminated)
        self.assertTrue(process.stdout.closed)

    def test_file_decoder_bounds_ffmpeg_diagnostics(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO()
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                del timeout
                if self.returncode is None:
                    self.returncode = 1
                return self.returncode

        process = Process()

        def popen(*_args, **kwargs):
            kwargs["stderr"].write(bytes(70_000) + b"diagnostic-tail")
            return process

        with (
            patch("turnalign.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("turnalign.audio.subprocess.Popen", side_effect=popen),
            self.assertRaisesRegex(RuntimeError, "earlier output truncated.*diagnostic-tail") as error,
        ):
            list(file_chunks(Path("broken.mp3")))
        self.assertLess(len(str(error.exception)), 66_000)

    def test_file_decoder_kills_ffmpeg_after_termination_timeout(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO(bytes(64_000))
                self.returncode = None
                self.killed = False

            def poll(self):
                return self.returncode

            def terminate(self):
                return None

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout=None):
                if timeout is not None and self.returncode is None:
                    raise subprocess.TimeoutExpired("ffmpeg", timeout)
                return self.returncode

        process = Process()
        with (
            patch("turnalign.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("turnalign.audio.subprocess.Popen", return_value=process),
        ):
            decoded = file_chunks(Path("recording.mp3"), chunk_ms=100)
            next(decoded)
            decoded.close()
        self.assertTrue(process.killed)
        self.assertEqual(process.returncode, -9)

    def test_file_decoder_bounds_wait_after_output_closes(self):
        class Process:
            def __init__(self):
                self.stdout = io.BytesIO()
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                if timeout is not None and not self.terminated:
                    raise subprocess.TimeoutExpired("ffmpeg", timeout)
                return self.returncode

        process = Process()
        with (
            patch("turnalign.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("turnalign.audio.subprocess.Popen", return_value=process),
            self.assertRaisesRegex(RuntimeError, "did not exit after closing its output"),
        ):
            list(file_chunks(Path("stalled.mp3")))
        self.assertTrue(process.terminated)

    def test_endpointing_keeps_preroll_and_flushes_speech(self):
        source = [chunk(0, 0), chunk(1200, 0.1), chunk(1200, 0.2)]
        source.extend(chunk(0, 0.3 + index * 0.1) for index in range(7))
        groups = list(utterances(source, silence_seconds=0.6))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0].start, 0)

    def test_segmentation_settings_must_be_finite(self):
        source = [chunk(1000, 0)]
        for value in (float("nan"), float("inf"), float("-inf")):
            for setting in ("threshold", "silence_seconds", "max_seconds"):
                with (
                    self.subTest(api="utterances", setting=setting, value=value),
                    self.assertRaises(ValueError),
                ):
                    list(utterances(source, **{setting: value}))
            for setting in (
                "threshold",
                "silence_seconds",
                "max_seconds",
                "partial_seconds",
            ):
                with (
                    self.subTest(api="live_windows", setting=setting, value=value),
                    self.assertRaises(ValueError),
                ):
                    list(live_windows(source, **{setting: value}))

    def test_disk_timeline_slices_by_timestamp_without_chunk_scan(self):
        with AudioTimeline() as timeline:
            for index in range(1_000):
                timeline.append(chunk(index % 100, index * 0.1))
            excerpt = timeline.slice(99.8, 100.0)
            self.assertAlmostEqual(excerpt.start, 99.8)
            self.assertAlmostEqual(excerpt.duration, 0.2)
            self.assertEqual(timeline.chunk_count, 1_000)

    def test_disk_timeline_round_trip_preserves_gaps(self):
        with AudioTimeline() as timeline:
            timeline.append(chunk(1000, 1.0))
            timeline.append(chunk(2000, 1.2))
            decoded = list(timeline.iter_chunks(chunk_ms=100))
            self.assertEqual([round(item.start, 1) for item in decoded], [1.0, 1.1, 1.2])
            self.assertEqual(pcm_rms(decoded[1]), 0)

    def test_eight_hour_sparse_timeline_keeps_constant_python_state(self):
        with AudioTimeline() as timeline:
            for hour in range(9):
                timeline.append(AudioChunk(
                    array("h", [1000]).tobytes(),
                    hour * 3_600,
                    sample_rate=1,
                ))
            self.assertGreaterEqual(timeline.duration, 8 * 3_600)
            excerpt = timeline.slice(8 * 3_600, 8 * 3_600 + 1)
            self.assertAlmostEqual(excerpt.duration, 1)
            self.assertGreater(pcm_rms(excerpt), 0)
            self.assertNotIn("chunks", vars(timeline))


class SessionTests(unittest.TestCase):
    def test_batch_backend_runs_per_live_utterance(self):
        backend = FakeBatchBackend()
        source = [chunk(1500, 0), chunk(1500, 0.1), chunk(0, 0.2), chunk(0, 0.3)]
        events = list(transcribe_events(source, backend, live=True, silence_seconds=0.2))
        self.assertEqual([event.kind for event in events], ["commit", "end"])
        self.assertEqual(events[-1].metadata["backend"], "fake-batch")
        self.assertTrue(backend.closed)

    def test_native_streaming_preserves_partial_and_commit(self):
        backend = FakeStreamingBackend()
        events = list(transcribe_events([chunk(1500, 0)], backend, live=True))
        self.assertEqual([event.kind for event in events], ["partial", "commit", "end"])
        self.assertEqual(events[0].segment_id, events[1].segment_id)
        self.assertEqual([events[0].revision, events[1].revision], [1, 2])

    def test_stateful_streaming_backend_receives_only_new_chunks(self):
        backend = FakeStatefulStreamingBackend()
        source = [chunk(1500, index * 0.1) for index in range(3)]
        events = list(transcribe_events(source, backend, live=True))
        self.assertEqual(len(backend.session.accepted), 3)
        self.assertTrue(backend.session.closed)
        self.assertEqual(
            [event.kind for event in events],
            ["partial", "partial", "partial", "commit", "end"],
        )

    def test_batch_backend_emits_growing_partial_before_final(self):
        backend = FakeBatchBackend()
        source = [chunk(1500, index * 0.1) for index in range(12)]
        events = list(transcribe_events(
            source, backend, live=True, partial_seconds=0.5, max_utterance_seconds=5,
        ))
        self.assertEqual([event.kind for event in events], ["partial", "partial", "commit", "end"])
        self.assertEqual([event.revision for event in events[:3]], [1, 2, 3])

    def test_batch_live_path_stabilizes_partial_prefix_and_allows_final_correction(self):
        backend = RevisingBatchBackend()
        source = [chunk(1500, index * 0.1) for index in range(17)]
        events = list(transcribe_events(
            source, backend, live=True, partial_seconds=0.5, max_utterance_seconds=5,
        ))
        self.assertEqual([event.text for event in events[:4]], [
            "turn align", "turn alignment", "turn align", "different final",
        ])
        self.assertTrue(all(event.metadata["stabilized"] for event in events[:4]))

    def test_builtin_backends_are_discoverable_without_importing_models(self):
        names = available("asr")
        self.assertTrue({"glm-asr", "faster-whisper", "funasr", "transformers-whisper", "whisper-cpp"} <= set(names))

    def test_alignment_and_diarization_emit_replace_before_end(self):
        backend = FakeBatchBackend()
        events = list(transcribe_events(
            [chunk(1500, 0), chunk(1500, 0.1)], backend,
            live=True, max_utterance_seconds=1,
            aligner=FakeAligner(), diarizer=FakeDiarizer(),
        ))
        self.assertEqual([event.kind for event in events], ["commit", "replace", "end"])
        self.assertEqual(events[1].speaker, "speaker-1")
        self.assertEqual(events[1].words[0].speaker, "speaker-1")
        self.assertEqual(events[-1].metadata["audio_seconds"], 0.2)

    def test_batch_aligner_runs_once_for_all_commits(self):
        aligner = FakeBatchAligner()
        backend = FakeBatchBackend()
        source = [chunk(1500, 0), chunk(0, 0.1), chunk(1500, 0.2), chunk(0, 0.3)]
        events = list(transcribe_events(
            source, backend, live=True, silence_seconds=0.1,
            max_utterance_seconds=1, aligner=aligner,
        ))
        self.assertEqual(aligner.batch_calls, 1)
        self.assertEqual([event.kind for event in events], [
            "commit", "commit", "replace", "replace", "end",
        ])

    def test_batch_aligner_never_materializes_more_than_its_batch_size(self):
        aligner = FakeBatchAligner()
        source = [
            chunk(1500, 0),
            chunk(0, 0.1),
            chunk(1500, 0.2),
            chunk(0, 0.3),
            chunk(1500, 0.4),
            chunk(0, 0.5),
        ]
        list(transcribe_events(
            source,
            FakeBatchBackend(),
            live=True,
            silence_seconds=0.1,
            aligner=aligner,
        ))
        self.assertEqual(aligner.batch_calls, 2)

    def test_preloaded_diarization_starts_in_parallel_with_asr(self):
        started = Event()
        source = [chunk(1500, 0), chunk(0, 0.1)]
        events = list(transcribe_events(
            iter(source), CoordinatedBackend(started), live=True,
            silence_seconds=0.1, diarizer=CoordinatedDiarizer(started),
            recorded_audio=source, parallel_diarization=True,
        ))
        self.assertTrue(events[-1].metadata["parallel_diarization"])
        self.assertGreaterEqual(events[-1].metadata["diarization_seconds"], 0)

    def test_two_pass_pipeline_refines_the_same_segment_id(self):
        realtime_backend = FakeBatchBackend()
        refinement_backend = RefinementBackend()
        pipeline = TwoPassPipeline(
            RealtimePipeline(realtime_backend, silence_seconds=0.1),
            OfflineRefinementPipeline(refinement_backend),
        )
        source = [chunk(1500, 0), chunk(0, 0.1)]
        events = list(pipeline.events(source))
        self.assertEqual([event.kind for event in events], ["commit", "replace", "end"])
        self.assertEqual(events[0].segment_id, events[1].segment_id)
        self.assertEqual(events[1].text, "refined transcript")
        validator = EventStreamValidator()
        for event in events:
            validator.accept(event)
        self.assertTrue(validator.ended)

    def test_online_diarizer_labels_commits_before_session_end(self):
        online_diarizer = FakeOnlineDiarizer()
        events = list(RealtimePipeline(
            FakeBatchBackend(),
            silence_seconds=0.1,
            online_diarizer=online_diarizer,
        ).events([chunk(1500, 0), chunk(0, 0.1)]))
        self.assertEqual(events[0].kind, "commit")
        self.assertEqual(events[0].speaker, "speaker-live")
        self.assertTrue(events[0].metadata["speaker_provisional"])
        self.assertTrue(online_diarizer.session.closed)
        self.assertTrue(online_diarizer.closed)

    def test_component_setup_failures_still_close_owned_resources(self):
        backend = FakeBatchBackend()
        online_diarizer = FailingOnlineDiarizer()
        with self.assertRaisesRegex(RuntimeError, "online session failed"):
            list(transcribe_events(
                [chunk(1500, 0)],
                backend,
                online_diarizer=online_diarizer,
            ))
        self.assertTrue(backend.closed)
        self.assertTrue(online_diarizer.closed)

        backend = FakeBatchBackend()
        with self.assertRaisesRegex(ValueError, "parallel diarization"):
            list(transcribe_events(
                [chunk(1500, 0)],
                backend,
                parallel_diarization=True,
            ))
        self.assertTrue(backend.closed)

    def test_recording_setup_failure_closes_owned_resources(self):
        backend = FakeBatchBackend()

        def failing_recording():
            yield chunk(1500, 0)
            raise RuntimeError("recording failed")

        with self.assertRaisesRegex(RuntimeError, "recording failed"):
            list(transcribe_events(
                [],
                backend,
                aligner=FakeAligner(),
                recorded_audio=failing_recording(),
            ))
        self.assertTrue(backend.closed)

    def test_two_pass_realtime_failure_closes_unused_refinement_backend(self):
        realtime_backend = FailingRealtimeBackend()
        refinement_backend = RefinementBackend()
        pipeline = TwoPassPipeline(
            RealtimePipeline(realtime_backend, silence_seconds=0.1),
            OfflineRefinementPipeline(refinement_backend),
        )
        with self.assertRaisesRegex(RuntimeError, "realtime model failed"):
            list(pipeline.events([chunk(1500, 0), chunk(0, 0.1)]))
        self.assertTrue(realtime_backend.closed)
        self.assertTrue(refinement_backend.closed)

    def test_offline_refinement_emits_consistent_speaker_merge(self):
        with AudioTimeline() as timeline:
            timeline.append(chunk(1500, 0))
            commit = TranscriptEvent(
                "commit", "seg-000000", 1, 0, 0.1,
                "draft", speaker="speaker-live",
            )
            events = list(OfflineRefinementPipeline(
                RefinementBackend(),
                diarizer=FakeDiarizer(),
            ).refine(timeline, [commit]))
        self.assertEqual([event.kind for event in events], ["speaker_merge", "replace"])
        self.assertEqual(events[0].metadata["from_speaker"], "speaker-live")
        self.assertEqual(events[0].metadata["to_speaker"], "speaker-1")
        self.assertEqual(events[1].segment_id, commit.segment_id)

    def test_offline_refinement_falls_back_to_single_item_aligner(self):
        with AudioTimeline() as timeline:
            timeline.append(chunk(1500, 0))
            commit = TranscriptEvent(
                "commit", "seg-000000", 1, 0, 0.1, "draft"
            )
            events = list(OfflineRefinementPipeline(
                RefinementBackend(),
                aligner=FakeAligner(),
            ).refine(timeline, [commit]))
        self.assertEqual([event.kind for event in events], ["replace"])
        self.assertEqual(events[0].words[0].text, "refined transcript")

    def test_two_pass_pipeline_preserves_draft_when_refinement_fails(self):
        pipeline = TwoPassPipeline(
            RealtimePipeline(FakeBatchBackend(), silence_seconds=0.1),
            OfflineRefinementPipeline(FailingRefinementBackend()),
        )
        events = list(pipeline.events([chunk(1500, 0), chunk(0, 0.1)]))
        self.assertEqual([event.kind for event in events], ["commit", "end"])
        self.assertEqual(events[-1].metadata["refinement_status"], "failed")
        self.assertEqual(events[0].text, "hello")


if __name__ == "__main__":
    unittest.main()
