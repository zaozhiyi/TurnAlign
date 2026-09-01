import hashlib
import json
import os
import platform
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turnalign import cli
from turnalign.cli import replay, validate_events
from turnalign.devices import Device
from turnalign.models import TranscriptEvent
from turnalign.plugins import Accelerator
from turnalign.profiles import select_execution_profile


class ClosableResource:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class CliIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_event_stream(path, events):
        path.write_text(
            "\n".join(
                json.dumps(event.to_dict(), ensure_ascii=False)
                for event in events
            ) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _transcribe_args(**overrides):
        values = {
            "execution_profile": "cpu-low-memory",
            "backend": "fake",
            "device": "cpu",
            "model": None,
            "language": None,
            "compute_type": None,
            "executable": None,
            "model_path": None,
            "backend_option": [],
            "hotword": [],
            "hotwords_file": [],
            "context": None,
            "context_file": None,
            "hotword_boost": None,
            "vad_backend": "none",
            "vad_option": [],
            "aligner": None,
            "aligner_option": [],
            "diarizer": None,
            "diarizer_option": [],
            "parallel_postprocess": None,
            "source": Path("sample.wav"),
            "chunk_ms": 100,
            "ffmpeg": "ffmpeg",
            "vad_output": None,
            "output": None,
            "output_format": "jsonl",
            "vad_threshold": 0.012,
            "silence_seconds": 0.7,
            "max_utterance_seconds": 20.0,
            "partial_seconds": 2.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _execution_profile():
        return SimpleNamespace(
            name="cpu-low-memory",
            asr_device="cpu",
            vad_device="cpu",
            alignment_device="cpu",
            alignment_batch_size=1,
            diarization_device="cpu",
            parallel_diarization=False,
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_whisper_cpp_vulkan_uses_cpu_profile_without_hiding_asr_device(self):
        devices = [Device(
            accelerator=Accelerator.CPU,
            available=True,
            runtime="native",
            device="cpu",
            vendor="test",
            name="test CPU",
            dtype="float32",
        )]
        requested = cli._profile_requested_device("whisper-cpp", "vulkan:1")
        self.assertEqual(requested, "cpu")
        self.assertEqual(cli._profile_requested_device("whisper-cpp", "vulkan:0"), "cpu")
        self.assertEqual(
            cli._profile_requested_device("whisper-cpp", " Vulkan:1 "), "cpu"
        )
        self.assertEqual(cli._profile_requested_device("whisper-cpp", "vulkan"), "cpu")
        self.assertEqual(
            cli._profile_name("whisper-cpp", "auto", "vulkan:1"),
            "cpu-low-memory",
        )
        self.assertEqual(cli._resolved_device("vulkan:1"), "vulkan:1")
        self.assertEqual(
            select_execution_profile(
                "auto", requested_device=requested, devices=devices, system="Windows"
            ).name,
            "cpu-low-memory",
        )
        with self.assertRaisesRegex(RuntimeError, "does not support requested accelerator"):
            select_execution_profile(
                "auto",
                requested_device=cli._profile_requested_device(
                    "transformers-whisper", "vulkan:1"
                ),
                devices=devices,
                system="Windows",
            )

    def test_whisper_cpp_vulkan_environment_reaches_backend_without_generic_probe(self):
        captured = {}

        def fake_profile(name, *, requested_device):
            captured["profile_name"] = name
            captured["profile_device"] = requested_device
            return SimpleNamespace(asr_device="cpu")

        def fake_create(name, config):
            captured["backend"] = name
            captured["asr_device"] = config.device
            raise RuntimeError("backend-created")

        args = SimpleNamespace(
            execution_profile="auto",
            backend="whisper-cpp",
            device="auto",
            model=None,
            language="zh",
            compute_type=None,
            executable="whisper-cli",
            model_path="model.bin",
            backend_option=[],
        )
        with patch.dict(os.environ, {"TURNALIGN_DEVICE": "vulkan:1"}), patch.object(
            cli, "select_execution_profile", side_effect=fake_profile
        ), patch.object(cli, "create_asr", side_effect=fake_create), self.assertRaisesRegex(
            RuntimeError, "backend-created"
        ):
            cli.transcribe_file(args)

        self.assertEqual(captured["profile_name"], "cpu-low-memory")
        self.assertEqual(captured["profile_device"], "cpu")
        self.assertEqual(captured["backend"], "whisper-cpp")
        self.assertEqual(captured["asr_device"], "vulkan:1")

    def test_private_hint_files_are_loaded_without_becoming_output_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hotwords = root / "hotwords.txt"
            context = root / "context.txt"
            hotwords.write_text("TERM_A\n# ignored\nTERM_B\n", encoding="utf-8")
            context.write_text("topic", encoding="utf-8")
            hints = cli._hints(SimpleNamespace(
                hotword=[], hotwords_file=[hotwords], context=None,
                context_file=context, hotword_boost=None,
            ))
            self.assertEqual(hints.hotwords, ("TERM_A", "TERM_B"))
            self.assertEqual(hints.context, "topic")

    def test_transcribe_setup_failure_closes_created_resources(self):
        backend = ClosableResource()
        vad = ClosableResource()
        args = self._transcribe_args(vad_backend="energy", aligner="broken")
        with (
            patch.object(cli, "select_execution_profile", return_value=self._execution_profile()),
            patch.object(cli, "create_asr", return_value=backend),
            patch.object(
                cli,
                "create_component",
                side_effect=[vad, RuntimeError("aligner failed")],
            ),
            self.assertRaisesRegex(RuntimeError, "aligner failed"),
        ):
            cli.transcribe_file(args)
        self.assertTrue(backend.closed)
        self.assertTrue(vad.closed)

    def test_output_open_failure_closes_resources_before_pipeline_starts(self):
        backend = ClosableResource()
        with tempfile.TemporaryDirectory() as directory:
            args = self._transcribe_args(output=Path(directory))
            with (
                patch.object(
                    cli,
                    "select_execution_profile",
                    return_value=self._execution_profile(),
                ),
                patch.object(cli, "create_asr", return_value=backend),
                patch.object(cli, "file_chunks", return_value=iter(())),
                self.assertRaises(OSError),
            ):
                cli.transcribe_file(args)
        self.assertTrue(backend.closed)

    def test_file_transcription_defaults_to_energy_vad(self):
        captured = {}

        def fake_transcribe(args):
            captured["vad_backend"] = args.vad_backend
            return 0

        with patch.object(cli, "transcribe_file", fake_transcribe), patch(
            "sys.argv", ["turnalign", "transcribe", "sample.wav"]
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured["vad_backend"], "energy")

    def test_no_vad_overrides_safe_default(self):
        captured = {}

        def fake_transcribe(args):
            captured["vad_backend"] = args.vad_backend
            return 0

        with patch.object(cli, "transcribe_file", fake_transcribe), patch(
            "sys.argv", ["turnalign", "transcribe", "sample.wav", "--no-vad"]
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured["vad_backend"], "none")

    def test_listen_accepts_explicit_second_pass_backend(self):
        captured = {}

        def fake_listen(args):
            captured["backend"] = args.refinement_backend
            captured["model"] = args.refinement_model
            captured["online_diarizer"] = args.online_diarizer
            return 0

        with patch.object(cli, "listen", fake_listen), patch(
            "sys.argv",
            [
                "turnalign",
                "listen",
                "--refinement-backend",
                "funasr",
                "--refinement-model",
                "paraformer-zh",
                "--online-diarizer",
                "custom-online-speaker",
            ],
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured, {
            "backend": "funasr",
            "model": "paraformer-zh",
            "online_diarizer": "custom-online-speaker",
        })

    def test_release_gate_exposes_real_model_thresholds(self):
        captured = {}

        def fake_gate(args):
            captured.update({
                "backend": args.backend,
                "max_rtf": args.max_realtime_factor,
                "max_commit": args.max_first_commit_seconds,
                "min_audio": args.min_audio_seconds,
                "native": args.require_native_streaming,
                "immutable_revision": args.require_immutable_model_revision,
                "source_commit": args.source_commit,
            })
            return 0

        with patch.object(cli, "release_gate", fake_gate), patch(
            "sys.argv",
            [
                "turnalign",
                "release-gate",
                "sample.wav",
                "--backend",
                "funasr-streaming",
                "--max-realtime-factor",
                "0.5",
                "--max-first-commit-seconds",
                "8",
                "--min-audio-seconds",
                "30",
                "--require-immutable-model-revision",
                "--source-commit",
                "a" * 40,
            ],
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured, {
            "backend": "funasr-streaming",
            "max_rtf": 0.5,
            "max_commit": 8.0,
            "min_audio": 30.0,
            "native": True,
            "immutable_revision": True,
            "source_commit": "a" * 40,
        })

    def test_release_gate_validates_audio_before_loading_backend(self):
        args = SimpleNamespace(
            source=Path("not-normalized.wav"),
            chunk_ms=500,
            ffmpeg="missing-ffmpeg",
            output=None,
        )
        with patch.object(
            cli,
            "file_chunks",
            side_effect=RuntimeError("ffmpeg is required"),
        ), patch.object(cli, "create_asr") as create_asr, self.assertRaisesRegex(
            RuntimeError, "ffmpeg is required"
        ):
            cli.release_gate(args)
        create_asr.assert_not_called()

    def test_serve_exposes_output_backpressure_timeout(self):
        captured = {}

        async def fake_serve(*_args, **options):
            captured.update(options)

        with patch.object(cli, "serve", fake_serve), patch.object(
            cli.logging, "basicConfig"
        ) as configure_logging, patch(
            "sys.argv",
            [
                "turnalign",
                "serve",
                "--backend",
                "glm-asr",
                "--log-level",
                "WARNING",
                "--output-backpressure-timeout",
                "2.5",
            ],
        ):
            self.assertEqual(cli.main(), 0)
        configure_logging.assert_called_once_with(
            level=cli.logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        self.assertEqual(captured["output_backpressure_timeout"], 2.5)

    def test_authentication_token_file_is_restricted_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "auth-token"
            token_path.write_text("private-token\n", encoding="utf-8")
            token_path.chmod(0o600)
            args = SimpleNamespace(auth_token_env=None, auth_token_file=token_path)

            self.assertEqual(cli._authentication_token(args), "private-token")

            token_path.write_text("私密令牌\n", encoding="utf-8")
            self.assertEqual(cli._authentication_token(args), "私密令牌")

            token_path.write_text("first\nsecond\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one non-empty token"):
                cli._authentication_token(args)

            token_path.write_text("private-token\n\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one non-empty token"):
                cli._authentication_token(args)

            token_path.write_bytes(b"x" * (8 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "exceeds 8192 bytes"):
                cli._authentication_token(args)

            if os.name == "posix":
                token_path.write_text("private-token\n", encoding="utf-8")
                token_path.chmod(0o640)
                with self.assertRaisesRegex(ValueError, "group or others"):
                    cli._authentication_token(args)

    def test_authentication_token_environment_uses_the_same_bounds(self):
        args = SimpleNamespace(
            auth_token_env="TURNALIGN_TEST_AUTH_TOKEN",
            auth_token_file=None,
        )
        with patch.dict(os.environ, {"TURNALIGN_TEST_AUTH_TOKEN": "私密令牌"}):
            self.assertEqual(cli._authentication_token(args), "私密令牌")
        for token in ("first\nsecond", "x" * (8 * 1024 + 1)):
            with self.subTest(length=len(token)), patch.dict(
                os.environ,
                {"TURNALIGN_TEST_AUTH_TOKEN": token},
            ), self.assertRaises(ValueError):
                cli._authentication_token(args)

    @unittest.skipUnless(os.name == "posix", "secure symlink rejection is POSIX-only")
    def test_authentication_token_file_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_path = root / "token"
            token_path.write_text("private-token\n", encoding="utf-8")
            token_path.chmod(0o600)
            link_path = root / "link"
            link_path.symlink_to(token_path)
            args = SimpleNamespace(auth_token_env=None, auth_token_file=link_path)

            with self.assertRaisesRegex(ValueError, "securely open"):
                cli._authentication_token(args)

    def test_serve_exposes_lifecycle_timeouts(self):
        captured = {}

        async def fake_serve(*_args, **options):
            captured.update(options)

        with patch.object(cli, "serve", fake_serve), patch(
            "sys.argv",
            [
                "turnalign",
                "serve",
                "--backend",
                "glm-asr",
                "--language",
                "zh",
                "--compute-type",
                "float16",
                "--executable",
                "/opt/whisper-cli",
                "--model-path",
                "/models/asr.bin",
                "--backend-option",
                "threads=4",
                "--allow-language",
                "en",
                "--allow-compute-type",
                "int8",
                "--initialization-timeout",
                "30",
                "--worker-shutdown-timeout",
                "1.5",
                "--finalization-timeout",
                "45",
                "--max-concurrent-sessions",
                "12",
                "--max-recovery-sessions",
                "10",
                "--max-recovery-event-kib",
                "256",
                "--max-recovery-events-mib",
                "4",
                "--max-recovery-audio-mib",
                "128",
                "--max-recovery-total-mib",
                "512",
                "--recovery-ttl-seconds",
                "90",
                "--max-control-message-bytes",
                "4096",
                "--start-timeout",
                "4",
                "--client-idle-timeout",
                "15",
                "--shutdown-grace-timeout",
                "20",
                "--backend-replicas",
                "2",
                "--allow-origin",
                "https://app.example",
                "--preload",
                "--require-immutable-model-revision",
            ],
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured["initialization_timeout"], 30.0)
        self.assertEqual(captured["worker_shutdown_timeout"], 1.5)
        self.assertEqual(captured["finalization_timeout"], 45.0)
        self.assertEqual(captured["max_concurrent_sessions"], 12)
        self.assertEqual(captured["max_recovery_sessions"], 10)
        self.assertEqual(captured["max_recovery_event_bytes"], 256 * 1024)
        self.assertEqual(
            captured["max_recovery_event_bytes_per_session"],
            4 * 1024 * 1024,
        )
        self.assertEqual(captured["max_recovery_audio_bytes"], 128 * 1024 * 1024)
        self.assertEqual(captured["max_recovery_total_bytes"], 512 * 1024 * 1024)
        self.assertEqual(captured["recovery_ttl_seconds"], 90.0)
        self.assertEqual(captured["max_control_message_bytes"], 4096)
        self.assertEqual(captured["start_timeout"], 4.0)
        self.assertEqual(captured["client_idle_timeout"], 15.0)
        self.assertEqual(captured["shutdown_grace_timeout"], 20.0)
        self.assertEqual(captured["backend_replicas"], 2)
        self.assertEqual(captured["allowed_origins"], (None, "https://app.example"))
        self.assertTrue(captured["preload"])
        self.assertTrue(captured["require_immutable_revision"])
        self.assertEqual(captured["default_language"], "zh")
        self.assertEqual(captured["default_compute_type"], "float16")
        self.assertEqual(captured["default_executable"], "/opt/whisper-cli")
        self.assertEqual(captured["default_model_path"], "/models/asr.bin")
        self.assertEqual(captured["default_backend_options"], {"threads": 4})
        self.assertEqual(captured["policy"].allowed_languages, frozenset({"zh", "en"}))
        self.assertEqual(
            captured["policy"].allowed_compute_types,
            frozenset({"float16", "int8"}),
        )

    def test_component_options_reject_ambiguous_or_nonstandard_values(self):
        self.assertEqual(
            cli._extra_options([
                "threads=4",
                "flash_attention=false",
                "revision=COMMIT_SHA",
                'settings={"batch":2}',
            ]),
            {
                "threads": 4,
                "flash_attention": False,
                "revision": "COMMIT_SHA",
                "settings": {"batch": 2},
            },
        )
        for options, expected in (
            (["temperature=NaN"], "non-standard JSON number"),
            (["temperature=Infinity"], "non-standard JSON number"),
            (["settings={\"batch\":1,\"batch\":2}"], "duplicate JSON key"),
            (["threads=1", "threads=2"], "duplicate backend option"),
            (["=1"], "invalid key"),
            (["bad key=1"], "invalid key"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError, expected
            ):
                cli._extra_options(options)

    def test_serve_handles_operator_interrupt_without_traceback(self):
        async def interrupted(*_args, **_options):
            raise KeyboardInterrupt

        with patch.object(cli, "serve", interrupted), patch(
            "sys.argv",
            ["turnalign", "serve", "--backend", "glm-asr"],
        ):
            self.assertEqual(cli.main(), 130)

    def test_websocket_gate_maps_cli_options_without_literal_auth(self):
        captured = {}

        class Report:
            passed = True

            @staticmethod
            def to_dict():
                return {"status": "passed"}

        async def fake_gate(uri, **options):
            captured["uri"] = uri
            captured.update(options)
            return Report()

        with patch.object(cli, "run_websocket_gate", fake_gate), patch.dict(
            os.environ,
            {"TURNALIGN_GATE_TOKEN": "private-token"},
        ), patch(
            "sys.argv",
            [
                "turnalign",
                "websocket-gate",
                "wss://asr.example/ws",
                "--sessions",
                "8",
                "--audio-seconds",
                "60",
                "--realtime",
                "--compute-type",
                "int8",
                "--min-audio-acks",
                "2",
                "--max-dropped-partials",
                "1",
                "--max-backpressure-pauses",
                "3",
                "--verify-recovery",
                "--recovery-resume-timeout",
                "7",
                "--auth-token-env",
                "TURNALIGN_GATE_TOKEN",
            ],
        ):
            self.assertEqual(cli.main(), 0)
        self.assertEqual(captured["uri"], "wss://asr.example/ws")
        self.assertEqual(captured["sessions"], 8)
        self.assertEqual(captured["audio_seconds"], 60.0)
        self.assertTrue(captured["realtime"])
        self.assertEqual(captured["compute_type"], "int8")
        self.assertEqual(captured["min_audio_acks"], 2)
        self.assertEqual(captured["max_dropped_partials"], 1)
        self.assertEqual(captured["max_backpressure_pauses"], 3)
        self.assertTrue(captured["verify_recovery"])
        self.assertEqual(captured["recovery_resume_timeout"], 7.0)
        self.assertEqual(captured["auth_token"], "private-token")

    def test_websocket_gate_reads_restricted_auth_token_file(self):
        captured = {}

        class Report:
            passed = True

            @staticmethod
            def to_dict():
                return {"status": "passed"}

        async def fake_gate(_uri, **options):
            captured.update(options)
            return Report()

        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "auth-token"
            token_path.write_text("file-token\n", encoding="utf-8")
            token_path.chmod(0o600)
            with patch.object(cli, "run_websocket_gate", fake_gate), patch(
                "sys.argv",
                [
                    "turnalign",
                    "websocket-gate",
                    "wss://asr.example/ws",
                    "--auth-token-file",
                    str(token_path),
                ],
            ):
                self.assertEqual(cli.main(), 0)

        self.assertEqual(captured["auth_token"], "file-token")

    def test_production_gate_maps_evidence_and_persists_verdict(self):
        captured = {}

        class Report:
            passed = True

            @staticmethod
            def to_dict():
                return {"schema_version": 1, "status": "passed", "failures": []}

        def fake_gate(release, quality, websocket, **options):
            captured.update({
                "reports": (release, quality, websocket),
                **options,
            })
            return Report()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "nested" / "production.json"
            arguments = [
                "turnalign",
                "production-gate",
                "release.json",
                "quality.json",
                "websocket.json",
                "--source-commit",
                "a" * 40,
            ]
            for kind in sorted(cli.REQUIRED_ARTIFACT_KINDS):
                arguments.extend(("--artifact", f"{kind}={kind}.evidence"))
            arguments.extend(("--report", str(output)))
            with patch.object(cli, "run_production_gate", fake_gate), patch(
                "sys.argv", arguments
            ), patch("builtins.print"):
                self.assertEqual(cli.main(), 0)

            self.assertEqual(captured["source_commit"], "a" * 40)
            self.assertEqual(
                {kind for kind, _path in captured["artifacts"]},
                cli.REQUIRED_ARTIFACT_KINDS,
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "passed",
            )

    def test_deployment_rehearsal_maps_safe_production_options(self):
        captured = {}

        class Report:
            passed = True

            @staticmethod
            def to_dict():
                return {"schema_version": 1, "status": "passed", "failures": []}

        async def fake_rehearsal(previous, candidate, uri, **options):
            captured.update({
                "previous": previous,
                "candidate": candidate,
                "uri": uri,
                **options,
            })
            return Report()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rehearsal.json"
            with patch.object(
                cli,
                "run_deployment_rehearsal",
                side_effect=fake_rehearsal,
            ), patch(
                "sys.argv",
                [
                    "turnalign",
                    "deployment-rehearsal",
                    "wss://asr.example.com/ws",
                    "--previous-commit",
                    "a" * 40,
                    "--candidate-commit",
                    "b" * 40,
                    "--backend",
                    "funasr-streaming",
                    "--model",
                    "paraformer-zh-streaming",
                    "--report",
                    str(output),
                ],
            ), patch("builtins.print"):
                self.assertEqual(cli.main(), 0)

            self.assertEqual(captured["previous"], "a" * 40)
            self.assertEqual(captured["candidate"], "b" * 40)
            self.assertEqual(captured["uri"], "wss://asr.example.com/ws")
            probe = captured["probe"]
            self.assertEqual(probe.backend, "funasr-streaming")
            self.assertEqual(probe.model, "paraformer-zh-streaming")
            self.assertEqual(probe.sessions, 2)
            self.assertEqual(json.loads(output.read_text())["status"], "passed")

    def test_model_manifest_hashes_files_and_persists_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "model.safetensors"
            second = root / "tokenizer.json"
            output = root / "evidence" / "model-manifest.json"
            first.write_bytes(b"weights")
            second.write_bytes(b"tokenizer")

            with patch("sys.argv", [
                "turnalign",
                "model-manifest",
                "--model-id",
                "huggingface://organization/model",
                "--model-revision",
                "a" * 40,
                "--file",
                str(first),
                "--file",
                str(second),
                "--output",
                str(output),
            ]), patch("builtins.print"):
                self.assertEqual(cli.main(), 0)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_revision"], "a" * 40)
            self.assertEqual(
                [item["name"] for item in payload["files"]],
                ["model.safetensors", "tokenizer.json"],
            )
            self.assertEqual(
                payload["files"][0]["sha256"],
                cli._sha256_file(first),
            )

    def test_host_profile_binds_source_host_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "host-profile.json"
            arguments = [
                "turnalign",
                "host-profile",
            ]
            for kind in sorted(cli.REQUIRED_ARTIFACT_KINDS - {"host-profile"}):
                path = root / f"{kind}.evidence"
                path.write_text(f"immutable {kind}\n", encoding="utf-8")
                arguments.extend(("--artifact", f"{kind}={path}"))
            arguments.extend(("--output", str(output)))
            runtime_prefix = f"/opt/turnalign/releases/{'a' * 40}/venv"
            runtime = {
                "python_executable": f"{runtime_prefix}/bin/python",
                "python_prefix": runtime_prefix,
                "turnalign_source_commit": "a" * 40,
                "turnalign_version": "0.1.0",
            }
            python_version = ".".join(platform.python_version().split(".")[:2])
            installed_distribution = {
                "name": "turnalign",
                "version": "0.1.0",
                "root": f"{runtime_prefix}/lib/python{python_version}/site-packages",
                "files": [{
                    "name": "turnalign/_source_commit.txt",
                    "sha256": hashlib.sha256(f"{'a' * 40}\n".encode()).hexdigest(),
                    "bytes": 41,
                }],
            }

            with patch(
                "turnalign.production_gate._installed_runtime_identity",
                return_value=runtime,
            ), patch(
                "turnalign.production_gate._installed_distribution_identity",
                return_value=installed_distribution,
            ), patch(
                "turnalign.production_gate.platform.system",
                return_value="Linux",
            ), patch(
                "turnalign.production_gate._read_linux_boot_id",
                return_value="12345678-1234-4234-8234-123456789abc",
            ), patch("sys.argv", arguments), patch("builtins.print"):
                self.assertEqual(cli.main(), 0)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 4)
            self.assertEqual(payload["source_commit"], "a" * 40)
            self.assertEqual(payload["runtime"], runtime)
            self.assertEqual(
                payload["installed_distribution"], installed_distribution
            )
            self.assertEqual(
                payload["platform"]["boot_id"],
                "12345678-1234-4234-8234-123456789abc",
            )
            self.assertGreater(payload["platform"]["logical_cpu_count"], 0)
            self.assertEqual(
                {item["kind"] for item in payload["artifacts"]},
                cli.REQUIRED_ARTIFACT_KINDS - {"host-profile"},
            )

    def test_replay_creates_parent_and_valid_end_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "nested" / "events.jsonl"
            source.write_text(
                json.dumps({"start": 0, "end": 1.25, "text": "你好"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(replay(source, output), 0)
            events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["kind"] for event in events], ["commit", "end"])
            self.assertEqual(events[-1]["metadata"]["segments"], 1)
            self.assertEqual(validate_events(output), 0)

    def test_validate_requires_end(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                json.dumps({
                    "kind": "commit", "segment_id": "s", "revision": 1,
                    "start": 0, "end": 1, "text": "x"
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing end"):
                validate_events(source)

    def test_validate_rejects_ambiguous_or_nonstandard_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            for content, message in (
                (
                    (
                        '{"kind":"end","segment_id":"session","revision":1,'
                        '"revision":2,"start":0,"end":0}\n'
                    ),
                    "duplicate JSON key",
                ),
                (
                    (
                        '{"kind":"end","segment_id":"session","revision":1,'
                        '"start":0,"end":0,"metadata":{"value":NaN}}\n'
                    ),
                    "non-standard JSON number",
                ),
            ):
                source.write_text(content, encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    validate_events(source)

    def test_quality_gate_cli_passes_and_fails_with_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            matching = root / "matching.jsonl"
            mismatching = root / "mismatching.jsonl"
            report_path = root / "reports" / "quality.json"
            end = TranscriptEvent("end", "session", 1, 1, 1)
            self._write_event_stream(reference, [
                TranscriptEvent("commit", "r", 1, 0, 1, "hello world"),
                end,
            ])
            self._write_event_stream(matching, [
                TranscriptEvent("commit", "h", 1, 0, 1, "hello world"),
                end,
            ])
            self._write_event_stream(mismatching, [
                TranscriptEvent("commit", "h", 1, 0, 1, "wrong"),
                end,
            ])

            with patch("sys.argv", [
                "turnalign", "quality-gate", str(reference), str(matching),
                "--max-cer", "0", "--max-wer", "0",
                "--min-reference-speech-seconds", "1",
                "--source-commit", "a" * 40,
                "--report", str(report_path),
            ]), patch("builtins.print"):
                self.assertEqual(cli.main(), 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["source_commit"], "a" * 40)
            self.assertEqual(report["reference_sha256"], cli._sha256_file(reference))
            self.assertEqual(report["hypothesis_sha256"], cli._sha256_file(matching))
            with patch("sys.argv", [
                "turnalign", "quality-gate", str(reference), str(mismatching),
                "--max-cer", "0.1",
            ]), patch("builtins.print"):
                self.assertEqual(cli.main(), 1)

    def test_quality_gate_cli_requires_complete_event_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete.jsonl"
            complete = root / "complete.jsonl"
            commit = TranscriptEvent("commit", "s", 1, 0, 1, "text")
            self._write_event_stream(incomplete, [commit])
            self._write_event_stream(complete, [
                commit,
                TranscriptEvent("end", "session", 1, 1, 1),
            ])
            with patch("sys.argv", [
                "turnalign", "quality-gate", str(incomplete), str(complete),
                "--max-cer", "0",
            ]), self.assertRaisesRegex(ValueError, "missing end"):
                cli.main()

    def test_quality_gate_cli_applies_and_reports_text_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.jsonl"
            hypothesis = root / "hypothesis.jsonl"
            end = TranscriptEvent("end", "session", 1, 1, 1)
            self._write_event_stream(reference, [
                TranscriptEvent("commit", "r", 1, 0, 1, "Ａ，Test"),
                end,
            ])
            self._write_event_stream(hypothesis, [
                TranscriptEvent("commit", "h", 1, 0, 1, "a test"),
                end,
            ])
            with patch("sys.argv", [
                "turnalign", "quality-gate", str(reference), str(hypothesis),
                "--max-cer", "0", "--max-wer", "0",
                "--unicode-normalization", "NFKC",
                "--ignore-case", "--ignore-punctuation",
            ]), patch("builtins.print") as output:
                self.assertEqual(cli.main(), 0)
            report = json.loads(output.call_args.args[0])
            self.assertEqual(report["evaluation"]["character_error_rate"], 0)
            self.assertEqual(
                report["evaluation"]["text_normalization"],
                {
                    "unicode_form": "NFKC",
                    "case_sensitive": False,
                    "punctuation_sensitive": False,
                },
            )


if __name__ == "__main__":
    unittest.main()
