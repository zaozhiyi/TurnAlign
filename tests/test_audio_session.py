import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from turnalign.audio import wave_chunks, write_wave
from turnalign.models import AudioChunk, Hypothesis, SpeakerTurn, Word
from turnalign.plugins import Accelerator, BackendCapabilities
from turnalign.registry import available
from turnalign.session import pcm_rms, transcribe_events, utterances


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


class FakeAligner:
    name = "fake-aligner"

    def align(self, audio, text):
        return [Word(text, audio.start, audio.start + audio.duration)]


class FakeDiarizer:
    name = "fake-diarizer"

    def diarize(self, chunks):
        items = list(chunks)
        yield SpeakerTurn(items[0].start, items[-1].start + items[-1].duration, "speaker-1")


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

    def test_endpointing_keeps_preroll_and_flushes_speech(self):
        source = [chunk(0, 0), chunk(1200, 0.1), chunk(1200, 0.2)]
        source.extend(chunk(0, 0.3 + index * 0.1) for index in range(7))
        groups = list(utterances(source, silence_seconds=0.6))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0].start, 0)


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

    def test_batch_backend_emits_growing_partial_before_final(self):
        backend = FakeBatchBackend()
        source = [chunk(1500, index * 0.1) for index in range(12)]
        events = list(transcribe_events(
            source, backend, live=True, partial_seconds=0.5, max_utterance_seconds=5,
        ))
        self.assertEqual([event.kind for event in events], ["partial", "partial", "commit", "end"])
        self.assertEqual([event.revision for event in events[:3]], [1, 2, 3])

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


if __name__ == "__main__":
    unittest.main()
