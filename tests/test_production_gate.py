import base64
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from os import fstat as real_fstat
from pathlib import Path
from unittest.mock import patch

from turnalign import production_gate as production_gate_module
from turnalign.production_gate import (
    REQUIRED_ARTIFACT_KINDS,
    create_host_profile,
    run_production_gate,
    write_json_report,
)


class ProductionGateTests(unittest.TestCase):
    @staticmethod
    def _write_test_wheel(
        path: Path,
        *,
        corrupt_record: bool = False,
        invalid_entry_point: bool = False,
    ) -> None:
        dist_info = "turnalign-0.1.0.dist-info"
        files = {
            "turnalign/__init__.py": b'__version__ = "0.1.0"\n',
            f"{dist_info}/METADATA": (
                b"Metadata-Version: 2.4\nName: turnalign\nVersion: 0.1.0\n\n"
            ),
            f"{dist_info}/WHEEL": (
                b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
            ),
            f"{dist_info}/entry_points.txt": (
                b"[console_scripts]\n"
                + (
                    b"# turnalign = turnalign.cli:main\n"
                    if invalid_entry_point
                    else b"turnalign = turnalign.cli:main\n"
                )
            ),
        }
        rows = []
        for name, content in sorted(files.items()):
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(
                b"="
            ).decode("ascii")
            if corrupt_record and name == "turnalign/__init__.py":
                digest = "a" * 43
            rows.append((name, f"sha256={digest}", str(len(content))))
        record_name = f"{dist_info}/RECORD"
        rows.append((record_name, "", ""))
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(rows)
        files[record_name] = record.getvalue().encode("utf-8")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)

    @staticmethod
    def _reports(root: Path) -> tuple[Path, Path, Path]:
        release = root / "release.json"
        quality = root / "quality.json"
        websocket = root / "websocket.json"
        write_json_report(release, {
            "status": "passed",
            "source_commit": "b" * 40,
            "input_audio_sha256": hashlib.sha256(
                b"immutable release-audio\n"
            ).hexdigest(),
            "failures": [],
            "backend": "native-streaming-test",
            "require_native_streaming": True,
            "native_streaming": True,
            "require_partial": True,
            "require_immutable_model_revision": True,
            "model_revision": "a" * 40,
            "min_audio_seconds": 30.0,
            "min_commits": 1,
            "max_realtime_factor": 1.0,
            "max_first_partial_seconds": 3.0,
            "max_first_commit_seconds": 8.0,
            "max_initialization_seconds": 120.0,
            "realtime_factor": 0.5,
            "first_partial_seconds": 0.5,
            "first_commit_seconds": 2.0,
            "initialization_seconds": 10.0,
            "processing_seconds": 15.0,
            "audio_seconds": 30.0,
            "events": 4,
            "partials": 1,
            "commits": 2,
            "replacements": 0,
        })
        write_json_report(quality, {
            "status": "passed",
            "source_commit": "b" * 40,
            "reference_sha256": hashlib.sha256(
                b"immutable quality-reference\n"
            ).hexdigest(),
            "hypothesis_sha256": hashlib.sha256(
                b"immutable quality-hypothesis\n"
            ).hexdigest(),
            "model_revision": "a" * 40,
            "failures": [],
            "max_character_error_rate": 0.1,
            "max_word_error_rate": None,
            "max_diarization_error_rate": None,
            "max_revision_updates_per_segment": None,
            "min_reference_segments": 10,
            "min_reference_characters": 100,
            "min_reference_speech_seconds": 60.0,
            "evaluation": {
                "text_normalization": {
                    "unicode_form": "none",
                    "case_sensitive": True,
                    "punctuation_sensitive": True,
                },
                "character_error_rate": 0.05,
                "word_error_rate": 0.1,
                "diarization_error_rate": None,
                "revision_updates_per_segment": 0.1,
                "reference_segments": 10,
                "hypothesis_segments": 10,
                "reference_characters": 100,
                "reference_words": 20,
                "reference_speech_seconds": 60.0,
                "reference_speakers": 2,
            },
        })
        write_json_report(websocket, {
            "status": "passed",
            "source_commit": "b" * 40,
            "uri": "wss://asr.example.com/ws",
            "sessions": 8,
            "passed_sessions": 8,
            "failed_sessions": 0,
            "realtime_pacing": True,
            "recovery_probe_required": True,
            "recovery_probe": {
                "passed": True,
                "disconnected_audio_seconds": 30.0,
                "first_last_acknowledged_sequence": 299,
                "resumed_next_audio_sequence": 300,
                "final_acknowledged_sequence": 599,
                "final_buffered_bytes": 0,
                "events": 2,
                "commits": 1,
                "audio_acks": 600,
                "failure": None,
            },
            "max_ready_seconds": 10.0,
            "max_total_seconds": 75.0,
            "min_commits_per_session": 1,
            "min_audio_acks_per_session": 600,
            "max_dropped_partials_per_session": 0,
            "max_backpressure_pauses_per_session": 0,
            "audio_seconds_per_session": 60.0,
            "ready_seconds_p95": 2.0,
            "total_seconds_p95": 62.0,
            "events": 16,
            "commits": 8,
            "audio_acks": 4_800,
            "backpressure_pauses": 0,
            "dropped_partials": 0,
            "results": [
                {
                    "session": session,
                    "passed": True,
                    "ready_seconds": 2.0,
                    "total_seconds": 62.0,
                    "events": 2,
                    "partials": 0,
                    "commits": 1,
                    "audio_acks": 600,
                    "last_acknowledged_sequence": 599,
                    "final_buffered_bytes": 0,
                    "backpressure_pauses": 0,
                    "dropped_partials": 0,
                    "failure": None,
                }
                for session in range(1, 9)
            ],
        })
        return release, quality, websocket

    @staticmethod
    def _artifacts(root: Path) -> list[tuple[str, Path]]:
        artifacts = []
        for kind in REQUIRED_ARTIFACT_KINDS:
            path = root / f"{kind}.evidence"
            if kind == "dependency-lock":
                path.write_text(
                    "websockets==17.1 \\\n"
                    f"    --hash=sha256:{'c' * 64}\n",
                    encoding="utf-8",
                )
            elif kind == "sbom":
                write_json_report(path, {
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.6",
                    "metadata": {"component": {
                        "bom-ref": "root-component",
                        "name": "turnalign",
                        "version": "0.1.0",
                        "purl": "pkg:pypi/turnalign@0.1.0",
                    }},
                    "components": [{
                        "bom-ref": "websockets==17.1",
                        "name": "websockets",
                        "version": "17.1",
                        "purl": "pkg:pypi/websockets@17.1",
                    }],
                    "dependencies": [{
                        "ref": "root-component",
                        "dependsOn": ["websockets==17.1"],
                    }],
                })
            elif kind == "model-manifest":
                path.write_text("pending\n", encoding="utf-8")
            elif kind == "wheel":
                ProductionGateTests._write_test_wheel(path)
            else:
                path.write_bytes(f"immutable {kind}\n".encode())
            artifacts.append((kind, path))
        model = next(path for kind, path in artifacts if kind == "model")
        manifest = next(
            path for kind, path in artifacts if kind == "model-manifest"
        )
        write_json_report(manifest, {
            "schema_version": 1,
            "model_id": "modelscope://damo/paraformer-zh-streaming",
            "model_revision": "a" * 40,
            "files": [{
                "name": model.name,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "bytes": model.stat().st_size,
            }],
        })
        host_profile = next(
            path for kind, path in artifacts if kind == "host-profile"
        )
        profile_artifacts = [
            (kind, path) for kind, path in artifacts if kind != "host-profile"
        ]
        write_json_report(
            host_profile,
            create_host_profile("b" * 40, profile_artifacts),
        )
        return artifacts

    def test_passes_and_hash_binds_all_required_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=self._artifacts(root),
            )

            self.assertTrue(report.passed)
            self.assertEqual(report.schema_version, 1)
            self.assertEqual(len(report.artifacts), len(REQUIRED_ARTIFACT_KINDS))
            self.assertEqual(len(report.release_report.sha256), 64)
            self.assertEqual(
                {artifact.kind for artifact in report.artifacts},
                REQUIRED_ARTIFACT_KINDS,
            )

    def test_reports_every_weakened_production_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            release_payload = json.loads(release.read_text(encoding="utf-8"))
            release_payload["require_immutable_model_revision"] = False
            write_json_report(release, release_payload)
            quality_payload = json.loads(quality.read_text(encoding="utf-8"))
            quality_payload["min_reference_speech_seconds"] = 0
            write_json_report(quality, quality_payload)
            websocket_payload = json.loads(websocket.read_text(encoding="utf-8"))
            websocket_payload["uri"] = "wss://127.0.0.1:8765/ws"
            websocket_payload["recovery_probe_required"] = False
            websocket_payload["sessions"] = 1
            websocket_payload["passed_sessions"] = 1
            websocket_payload["results"] = [{"passed": True}]
            websocket_payload["max_backpressure_pauses_per_session"] = None
            write_json_report(websocket, websocket_payload)

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=[],
            )

            self.assertFalse(report.passed)
            failures = "\n".join(report.failures)
            self.assertIn("immutable model revision", failures)
            self.assertIn("labelled-speech", failures)
            self.assertIn("public wss://", failures)
            self.assertIn("recovery verification", failures)
            self.assertIn("concurrent sessions", failures)
            self.assertIn("backpressure-pause ceiling", failures)
            for kind in REQUIRED_ARTIFACT_KINDS:
                self.assertIn(f"missing required artifact kind: {kind}", failures)

    def test_independently_rejects_unsafe_websocket_report_uris(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            for uri, expected in (
                (
                    "wss://user:secret@asr.example.com/ws",
                    "must not contain credentials",
                ),
                ("wss://asr.example.com/ws?token=secret", "must not contain"),
                ("wss://asr.example.com/ws#secret", "must not contain"),
                ("wss://asr.example.com:99999/ws", "invalid port"),
            ):
                payload = json.loads(websocket.read_text(encoding="utf-8"))
                payload["uri"] = uri
                write_json_report(websocket, payload)
                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )
                with self.subTest(uri=uri):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_rejects_forged_or_type_ambiguous_websocket_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            for mutate, expected in (
                (
                    lambda payload: payload.update(recovery_probe={"passed": True}),
                    "recovery probe lacks complete typed evidence",
                ),
                (
                    lambda payload: payload.update(failed_sessions=False),
                    "contains failed sessions",
                ),
                (
                    lambda payload: payload.update(
                        max_dropped_partials_per_session=False
                    ),
                    "did not forbid dropped partials",
                ),
                (
                    lambda payload: payload["recovery_probe"].update(
                        resumed_next_audio_sequence=301
                    ),
                    "inconsistent sequence or buffer evidence",
                ),
                (
                    lambda payload: payload["results"][0].update(audio_acks=True),
                    "lacks complete typed per-session evidence",
                ),
                (
                    lambda payload: payload.update(ready_seconds_p95=1.0),
                    "latency p95 does not match per-session evidence",
                ),
            ):
                payload = json.loads(websocket.read_text(encoding="utf-8"))
                mutate(payload)
                write_json_report(websocket, payload)
                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )
                with self.subTest(expected=expected):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))
                _, _, websocket = self._reports(root)

    def test_rejects_incomplete_or_inconsistent_release_and_quality_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._artifacts(root)
            for report_name, mutate, expected in (
                (
                    "release",
                    lambda payload: payload.update(events=True),
                    "complete typed event counts",
                ),
                (
                    "release",
                    lambda payload: payload.update(events=2),
                    "inconsistent with its typed events",
                ),
                (
                    "release",
                    lambda payload: payload.update(processing_seconds=10.0),
                    "real-time factor is inconsistent",
                ),
                (
                    "quality",
                    lambda payload: payload["evaluation"].update(
                        reference_segments=10.5
                    ),
                    "complete typed evaluation counts",
                ),
                (
                    "quality",
                    lambda payload: payload["evaluation"].pop("text_normalization"),
                    "text-normalization policy",
                ),
                (
                    "quality",
                    lambda payload: payload["evaluation"].update(word_error_rate=False),
                    "WER is missing or invalid",
                ),
            ):
                release, quality, websocket = self._reports(root)
                target = release if report_name == "release" else quality
                payload = json.loads(target.read_text(encoding="utf-8"))
                mutate(payload)
                write_json_report(target, payload)

                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )

                with self.subTest(report=report_name, expected=expected):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_binds_model_manifest_revision_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for mutate, expected in (
                (
                    lambda payload: payload.update(model_revision="c" * 40),
                    "revision does not match",
                ),
                (
                    lambda payload: payload["files"][0].update(sha256="d" * 64),
                    "does not match the retained model artifacts",
                ),
                (
                    lambda payload: payload["files"][0].update(bytes=True),
                    "invalid file identity",
                ),
            ):
                artifacts = self._artifacts(root)
                manifest = next(
                    path for kind, path in artifacts if kind == "model-manifest"
                )
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                mutate(payload)
                write_json_report(manifest, payload)

                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )

                with self.subTest(expected=expected):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_rejects_invalid_or_record_tampered_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for mutate, expected in (
                (
                    lambda path: path.write_bytes(b"not a wheel\n"),
                    "not a valid readable ZIP archive",
                ),
                (
                    lambda path: self._write_test_wheel(path, corrupt_record=True),
                    "RECORD hash or size does not match",
                ),
                (
                    lambda path: self._write_test_wheel(
                        path, invalid_entry_point=True
                    ),
                    "does not expose the TurnAlign console entry point",
                ),
            ):
                artifacts = self._artifacts(root)
                wheel = next(path for kind, path in artifacts if kind == "wheel")
                mutate(wheel)
                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )

                with self.subTest(expected=expected):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_binds_host_profile_to_source_and_every_other_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for mutate, expected in (
                (
                    lambda payload: payload.update(source_commit="c" * 40),
                    "not bound to the production source commit",
                ),
                (
                    lambda payload: payload["platform"].update(
                        logical_cpu_count=True
                    ),
                    "complete typed platform evidence",
                ),
                (
                    lambda payload: payload["artifacts"][0].update(bytes=1),
                    "does not match the retained deployment artifacts",
                ),
            ):
                artifacts = self._artifacts(root)
                profile = next(
                    path for kind, path in artifacts if kind == "host-profile"
                )
                payload = json.loads(profile.read_text(encoding="utf-8"))
                mutate(payload)
                write_json_report(profile, payload)
                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )

                with self.subTest(expected=expected):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_rejects_ambiguous_or_mutable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            release.write_text('{"status":"passed","status":"failed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=self._artifacts(root),
                )

            release.unlink()
            release.symlink_to(quality)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=self._artifacts(root),
                )

    def test_report_writer_rejects_nonstandard_numbers_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with self.assertRaises(ValueError):
                write_json_report(report, {"metric": float("nan")})
            self.assertFalse(report.exists())

    def test_rejects_noncanonical_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            with self.assertRaisesRegex(ValueError, "lowercase 40-character"):
                run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="B" * 40,
                    artifacts=self._artifacts(root),
                )

    def test_rejects_stale_reports_and_mismatched_input_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            release_payload = json.loads(release.read_text(encoding="utf-8"))
            release_payload["source_commit"] = "c" * 40
            release_payload["input_audio_sha256"] = "d" * 64
            write_json_report(release, release_payload)

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=self._artifacts(root),
            )

            self.assertFalse(report.passed)
            failures = "\n".join(report.failures)
            self.assertIn("release report is not bound to source commit", failures)
            self.assertIn("release audio report digest does not match", failures)

    def test_rejects_quality_evidence_from_a_different_model_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            quality_payload = json.loads(quality.read_text(encoding="utf-8"))
            quality_payload["model_revision"] = "c" * 40
            write_json_report(quality, quality_payload)

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=self._artifacts(root),
            )

            self.assertFalse(report.passed)
            self.assertIn(
                "quality and release reports identify different model revisions",
                report.failures,
            )

    def test_rejects_malformed_or_incomplete_sbom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            sbom = next(path for kind, path in artifacts if kind == "sbom")
            write_json_report(sbom, {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "metadata": {"component": {"name": "another-package"}},
                "components": [],
            })
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            failures = "\n".join(report.failures)
            self.assertIn("TurnAlign root component", failures)
            self.assertIn("no runtime components", failures)

    def test_rejects_unhashed_or_sbom_mismatched_dependency_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            lock = next(
                path for kind, path in artifacts if kind == "dependency-lock"
            )
            lock.write_text("websockets==16.0\n", encoding="utf-8")
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            failures = "\n".join(report.failures)
            self.assertIn("lacks a SHA-256 hash", failures)
            self.assertIn("SBOM does not match locked runtime requirement", failures)

    def test_rejects_sbom_with_inconsistent_package_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            sbom = next(path for kind, path in artifacts if kind == "sbom")
            payload = json.loads(sbom.read_text(encoding="utf-8"))
            payload["components"][0]["purl"] = "pkg:pypi/another-package@17.1"
            write_json_report(sbom, payload)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            self.assertIn(
                "unversioned or unidentifiable component",
                "\n".join(report.failures),
            )

    def test_rejects_requirement_smuggled_through_hash_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            lock = next(
                path for kind, path in artifacts if kind == "dependency-lock"
            )
            lock.write_text(
                "websockets==17.1 \\\n"
                f"torch==2.0.0 --hash=sha256:{'d' * 64}\n",
                encoding="utf-8",
            )
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            self.assertIn(
                "continuation contains a second requirement",
                "\n".join(report.failures),
            )

    def test_rejects_oversized_dependency_lock_without_parsing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            lock = next(
                path for kind, path in artifacts if kind == "dependency-lock"
            )
            lock.write_bytes(b"x" * (4 * 1024 * 1024 + 1))

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            self.assertIn(
                "dependency lock exceeds 4194304 bytes",
                report.failures,
            )

    def test_rejects_oversized_gate_report_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            release.write_bytes(b"x" * (2 * 1024 * 1024 + 1))

            with self.assertRaisesRegex(ValueError, "gate report exceeds 2097152 bytes"):
                run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=self._artifacts(root),
                )

    def test_snapshot_rejects_evidence_that_changes_during_read(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.bin"
            evidence.write_bytes(b"immutable")
            calls = 0

            def changing_fstat(descriptor):
                nonlocal calls
                calls += 1
                metadata = real_fstat(descriptor)
                if calls == 2:
                    values = list(metadata)
                    values[6] += 1
                    return type(metadata)(values)
                return metadata

            with patch.object(
                production_gate_module.os,
                "fstat",
                side_effect=changing_fstat,
            ), self.assertRaisesRegex(ValueError, "changed while it was being read"):
                production_gate_module._snapshot_evidence(evidence)

    def test_atomic_writer_creates_parent_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.json"
            write_json_report(output, {"status": "passed"})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "passed"},
            )
            self.assertEqual(list(output.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
