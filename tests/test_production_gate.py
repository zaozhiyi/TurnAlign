import base64
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from os import fstat as real_fstat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from turnalign import production_gate as production_gate_module
from turnalign.production_gate import (
    REQUIRED_ARTIFACT_KINDS,
    create_deployment_state,
    create_host_profile,
    run_production_gate,
    write_json_report,
)

ROOT = Path(__file__).resolve().parents[1]


class ProductionGateTests(unittest.TestCase):
    BOOT_ID = "12345678-1234-4234-8234-123456789abc"
    NOW = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    @staticmethod
    def _effective_configuration(
        service_sha256: str,
        service_bytes: int,
        nginx_sha256: str,
        nginx_bytes: int,
    ) -> dict[str, object]:
        return {
            "systemd": {
                "fragment_path": "/etc/systemd/system/turnalign.service",
                "drop_in_paths": [],
                "need_daemon_reload": False,
                "active_state": "active",
                "sub_state": "running",
                "sha256": service_sha256,
                "bytes": service_bytes,
            },
            "nginx": {
                "configuration_path": "/etc/nginx/conf.d/turnalign.conf",
                "loaded_occurrences": 1,
                "warning_free": True,
                "sha256": nginx_sha256,
                "bytes": nginx_bytes,
            },
        }

    @staticmethod
    def _write_test_wheel(
        path: Path,
        *,
        corrupt_record: bool = False,
        invalid_entry_point: bool = False,
        source_commit: str = "b" * 40,
    ) -> None:
        dist_info = "turnalign-0.1.0.dist-info"
        files = {
            "turnalign/__init__.py": b'__version__ = "0.1.0"\n',
            "turnalign/_source_commit.txt": f"{source_commit}\n".encode("ascii"),
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
    def _installed_distribution(wheel: Path, source_commit: str = "b" * 40):
        with zipfile.ZipFile(wheel) as archive:
            files = [
                {
                    "name": name,
                    "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
                    "bytes": archive.getinfo(name).file_size,
                }
                for name in sorted(archive.namelist())
                if name.startswith("turnalign/")
            ]
        version = ".".join(
            production_gate_module.platform.python_version().split(".")[:2]
        )
        return {
            "name": "turnalign",
            "version": "0.1.0",
            "root": (
                f"/opt/turnalign/releases/{source_commit}/venv/lib/"
                f"python{version}/site-packages"
            ),
            "files": files,
        }

    @staticmethod
    def _reports(root: Path) -> tuple[Path, Path, Path]:
        release = root / "release.json"
        quality = root / "quality.json"
        websocket = root / "websocket.json"
        write_json_report(release, {
            "status": "passed",
            "source_commit": "b" * 40,
            "created_at": ProductionGateTests.NOW,
            "validity_seconds": 86400.0,
            "input_audio_sha256": hashlib.sha256(
                b"immutable release-audio\n"
            ).hexdigest(),
            "failures": [],
            "backend": "native-streaming-test",
            "model": "paraformer-zh-streaming",
            "loaded_models": [{
                "path": "/var/lib/turnalign/models/paraformer-zh-streaming/model.evidence",
                "sha256": hashlib.sha256(
                    b"immutable model\n"
                ).hexdigest(),
                "bytes": len(b"immutable model\n"),
            }],
            "require_native_streaming": True,
            "native_streaming": True,
            "require_partial": True,
            "require_immutable_model_revision": True,
            "require_local_model": True,
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
            "created_at": ProductionGateTests.NOW,
            "validity_seconds": 86400.0,
            "reference_sha256": hashlib.sha256(
                b"immutable quality-reference\n"
            ).hexdigest(),
            "hypothesis_sha256": hashlib.sha256(
                b"immutable quality-hypothesis\n"
            ).hexdigest(),
            "model": "paraformer-zh-streaming",
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
            "created_at": ProductionGateTests.NOW,
            "validity_seconds": 86400.0,
            "uri": "wss://asr.example.com/ws",
            "identity_consistent": True,
            "backend": "native-streaming-test",
            "backend_implementation": "native-streaming-test",
            "model": "paraformer-zh-streaming",
            "model_revision": "a" * 40,
            "device": "cpu",
            "language": "zh",
            "compute_type": None,
            "loaded_models": [{
                "path": "/var/lib/turnalign/models/paraformer-zh-streaming/model.evidence",
                "sha256": hashlib.sha256(
                    b"immutable model\n"
                ).hexdigest(),
                "bytes": len(b"immutable model\n"),
            }],
            "probe_audio_sha256": hashlib.sha256(
                b"immutable websocket-probe-audio\n"
            ).hexdigest(),
            "probe_audio_bytes": len(b"immutable websocket-probe-audio\n"),
            "probe_audio_rms": 1234.5,
            "sessions": 8,
            "passed_sessions": 8,
            "failed_sessions": 0,
            "realtime_pacing": True,
            "recovery_probe_required": True,
            "recovery_probe": {
                "passed": True,
                "backend": "native-streaming-test",
                "backend_implementation": "native-streaming-test",
                "model": "paraformer-zh-streaming",
                "model_revision": "a" * 40,
                "device": "cpu",
                "language": "zh",
                "compute_type": None,
                "loaded_models": [{
                    "path": "/var/lib/turnalign/models/paraformer-zh-streaming/model.evidence",
                    "sha256": hashlib.sha256(
                        b"immutable model\n"
                    ).hexdigest(),
                    "bytes": len(b"immutable model\n"),
                }],
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
                    "backend": "native-streaming-test",
                    "backend_implementation": "native-streaming-test",
                    "model": "paraformer-zh-streaming",
                    "model_revision": "a" * 40,
                    "device": "cpu",
                    "language": "zh",
                    "compute_type": None,
                    "loaded_models": [{
                        "path": "/var/lib/turnalign/models/paraformer-zh-streaming/model.evidence",
                        "sha256": hashlib.sha256(
                            b"immutable model\n"
                        ).hexdigest(),
                        "bytes": len(b"immutable model\n"),
                    }],
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
    def _rollback_rehearsal(websocket: Path) -> dict[str, object]:
        candidate = json.loads(websocket.read_text(encoding="utf-8"))
        previous = json.loads(websocket.read_text(encoding="utf-8"))
        previous["source_commit"] = "c" * 40

        def phase(
            name: str,
            from_commit: str,
            target_commit: str,
            started_at: str,
            activated_at: str,
            completed_at: str,
            probe: dict[str, object],
        ) -> dict[str, object]:
            return {
                "name": name,
                "status": "passed",
                "from_commit": from_commit,
                "target_commit": target_commit,
                "target_path": f"/opt/turnalign/releases/{target_commit}",
                "started_at": started_at,
                "activated_at": activated_at,
                "completed_at": completed_at,
                "activation_seconds": 0.01,
                "restart": {
                    "restart_exit_code": 0,
                    "active_exit_code": 0,
                    "seconds": 1.0,
                    "failure": None,
                },
                "readiness": {
                    "uri": "http://127.0.0.1:8765/readyz",
                    "status_code": 200,
                    "ready": True,
                    "preloaded": True,
                    "attempts": 2,
                    "seconds": 1.0,
                    "failure": None,
                    "loaded_models": [{
                        "path": "/var/lib/turnalign/models/paraformer-zh-streaming/model.evidence",
                        "sha256": hashlib.sha256(
                            b"immutable model\n"
                        ).hexdigest(),
                        "bytes": len(b"immutable model\n"),
                    }],
                },
                "websocket_report": probe,
                "failures": [],
            }

        return {
            "schema_version": 2,
            "status": "passed",
            "candidate_commit": "b" * 40,
            "previous_commit": "c" * 40,
            "boot_id": ProductionGateTests.BOOT_ID,
            "release_root": "/opt/turnalign/releases",
            "current_link": "/opt/turnalign/current",
            "lock_path": "/run/lock/turnalign-deployment.lock",
            "service": "turnalign.service",
            "systemctl": "/usr/bin/systemctl",
            "ready_uri": "http://127.0.0.1:8765/readyz",
            "websocket_uri": "wss://asr.example.com/ws",
            "started_at": "2026-09-01T08:00:00.000Z",
            "completed_at": "2026-09-01T08:00:08.000Z",
            "initial_active_commit": "b" * 40,
            "final_active_commit": "b" * 40,
            "transaction_id": "d" * 64,
            "transaction_path": (
                "/var/lib/turnalign-deployment/pending-activation.json"
            ),
            "rollback": phase(
                "rollback",
                "b" * 40,
                "c" * 40,
                "2026-09-01T08:00:00.000Z",
                "2026-09-01T08:00:01.000Z",
                "2026-09-01T08:00:03.000Z",
                previous,
            ),
            "restore": phase(
                "restore",
                "c" * 40,
                "b" * 40,
                "2026-09-01T08:00:04.000Z",
                "2026-09-01T08:00:05.000Z",
                "2026-09-01T08:00:07.000Z",
                candidate,
            ),
            "failures": [],
        }

    @staticmethod
    def _deployment_activation(websocket: Path) -> dict[str, object]:
        rehearsal = ProductionGateTests._rollback_rehearsal(websocket)
        activation = json.loads(json.dumps(rehearsal["restore"]))
        activation.update({
            "name": "activate",
            "from_commit": "c" * 40,
            "target_commit": "b" * 40,
            "target_path": f"/opt/turnalign/releases/{'b' * 40}",
            "started_at": "2026-09-01T07:59:00.000Z",
            "activated_at": "2026-09-01T07:59:01.000Z",
            "completed_at": "2026-09-01T07:59:03.000Z",
        })
        return {
            "schema_version": 2,
            "status": "passed",
            "candidate_commit": "b" * 40,
            "previous_commit": "c" * 40,
            "boot_id": ProductionGateTests.BOOT_ID,
            "release_root": "/opt/turnalign/releases",
            "current_link": "/opt/turnalign/current",
            "lock_path": "/run/lock/turnalign-deployment.lock",
            "service": "turnalign.service",
            "systemctl": "/usr/bin/systemctl",
            "ready_uri": "http://127.0.0.1:8765/readyz",
            "websocket_uri": "wss://asr.example.com/ws",
            "started_at": "2026-09-01T07:59:00.000Z",
            "completed_at": "2026-09-01T07:59:04.000Z",
            "initial_active_commit": "c" * 40,
            "final_active_commit": "b" * 40,
            "transaction_id": "e" * 64,
            "transaction_path": (
                "/var/lib/turnalign-deployment/pending-activation.json"
            ),
            "activation": activation,
            "rollback": None,
            "failures": [],
        }

    @staticmethod
    def _artifacts(root: Path) -> list[tuple[str, Path]]:
        artifacts = []
        for kind in REQUIRED_ARTIFACT_KINDS:
            path = root / f"{kind}.evidence"
            if kind == "service-unit":
                path = root / "turnalign.service"
            elif kind == "nginx-config":
                path = root / "turnalign.conf"
            if kind == "deployment-activation":
                websocket_path = root / "websocket.json"
                if not websocket_path.exists():
                    _release, _quality, websocket_path = ProductionGateTests._reports(
                        root
                    )
                write_json_report(
                    path,
                    ProductionGateTests._deployment_activation(websocket_path),
                )
            elif kind == "deployment-state":
                path.write_bytes(b"pending deployment state\n")
            elif kind == "dependency-lock":
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
            elif kind == "rollback-rehearsal":
                websocket_path = root / "websocket.json"
                if not websocket_path.exists():
                    _release, _quality, websocket_path = ProductionGateTests._reports(
                        root
                    )
                write_json_report(
                    path,
                    ProductionGateTests._rollback_rehearsal(websocket_path),
                )
            elif kind == "wheel":
                ProductionGateTests._write_test_wheel(path)
            elif kind == "service-unit":
                path.write_bytes(
                    (ROOT / "deploy" / "systemd" / "turnalign.service").read_bytes()
                )
            elif kind == "nginx-config":
                path.write_bytes(
                    (ROOT / "deploy" / "nginx" / "turnalign.conf.example").read_bytes()
                )
            elif kind == "websocket-probe-audio":
                path.write_bytes(b"immutable websocket-probe-audio\n")
            else:
                path.write_bytes(f"immutable {kind}\n".encode())
            artifacts.append((kind, path))
        model = next(path for kind, path in artifacts if kind == "model")
        manifest = next(
            path for kind, path in artifacts if kind == "model-manifest"
        )
        write_json_report(manifest, {
            "schema_version": 1,
            "model_id": "paraformer-zh-streaming",
            "model_revision": "a" * 40,
            "files": [{
                "name": model.name,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "bytes": model.stat().st_size,
            }],
        })
        service_path = next(
            path for kind, path in artifacts if kind == "service-unit"
        )
        nginx_path = next(
            path for kind, path in artifacts if kind == "nginx-config"
        )
        deployment_state = next(
            path for kind, path in artifacts if kind == "deployment-state"
        )
        write_json_report(deployment_state, {
            "schema_version": 2,
            "active_commit": "b" * 40,
            "pending_transaction_id": None,
            "boot_id": ProductionGateTests.BOOT_ID,
            "effective_configuration": ProductionGateTests._effective_configuration(
                hashlib.sha256(service_path.read_bytes()).hexdigest(),
                service_path.stat().st_size,
                hashlib.sha256(nginx_path.read_bytes()).hexdigest(),
                nginx_path.stat().st_size,
            ),
            "created_at": ProductionGateTests.NOW,
            "validity_seconds": 300.0,
        })
        host_profile = next(
            path for kind, path in artifacts if kind == "host-profile"
        )
        profile_artifacts = [
            (kind, path) for kind, path in artifacts if kind != "host-profile"
        ]
        runtime_prefix = f"/opt/turnalign/releases/{'b' * 40}/venv"
        python_version = ".".join(
            production_gate_module.platform.python_version().split(".")[:2]
        )
        wheel = next(path for kind, path in artifacts if kind == "wheel")
        with patch.object(
            production_gate_module,
            "_installed_runtime_identity",
            return_value={
                "python_executable": f"{runtime_prefix}/bin/python",
                "python_prefix": runtime_prefix,
                "turnalign_source_commit": "b" * 40,
                "turnalign_version": "0.1.0",
            },
        ), patch.object(
            production_gate_module,
            "_installed_distribution_identity",
            return_value=ProductionGateTests._installed_distribution(wheel),
        ), patch.object(
            production_gate_module,
            "_installed_dependency_identity",
            return_value={
                "websockets": {
                    "name": "websockets",
                    "version": "17.1",
                    "root": (
                        f"/opt/turnalign/releases/{'b' * 40}/venv/lib/"
                        f"python{python_version}/site-packages"
                    ),
                    "file_count": 1,
                    "sha256": hashlib.sha256(b"dependency\n").hexdigest(),
                    "bytes": len(b"dependency\n"),
                },
            },
        ), patch.object(
            production_gate_module,
            "_SERVICE_UNIT_PATH",
            service_path,
        ), patch.object(
            production_gate_module,
            "_NGINX_CONFIG_PATH",
            nginx_path,
        ), patch.object(
            production_gate_module,
            "_active_release_commit",
            return_value="b" * 40,
        ), patch.object(
            production_gate_module.platform,
            "system",
            return_value="Linux",
        ), patch.object(
            production_gate_module,
            "_read_linux_boot_id",
            return_value=ProductionGateTests.BOOT_ID,
        ), patch.object(
            production_gate_module,
            "_acquire_deployment_lock",
            side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
        ), patch.object(
            production_gate_module,
            "_require_root_owned_production_config",
        ), patch.object(
            production_gate_module,
            "_capture_effective_configuration",
            side_effect=lambda **kwargs: ProductionGateTests._effective_configuration(
                kwargs["service_snapshot"].sha256,
                kwargs["service_snapshot"].size,
                kwargs["nginx_snapshot"].sha256,
                kwargs["nginx_snapshot"].size,
            ),
        ):
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

    def test_zero_commit_websocket_evidence_never_passes_production_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            payload = json.loads(websocket.read_text(encoding="utf-8"))
            payload["min_commits_per_session"] = 0
            for result in payload["results"]:
                result["commits"] = 0
            payload["commits"] = 0
            write_json_report(websocket, payload)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=self._artifacts(root),
            )
            self.assertFalse(report.passed)
            self.assertTrue(
                any("commit per session" in failure for failure in report.failures)
            )

    def test_stale_gate_report_cannot_be_replayed_as_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            payload = json.loads(release.read_text(encoding="utf-8"))
            payload["created_at"] = "2020-01-01T00:00:00.000Z"
            payload["validity_seconds"] = 60.0
            write_json_report(release, payload)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=self._artifacts(root),
            )
            self.assertFalse(report.passed)
            self.assertTrue(
                any("stale" in failure for failure in report.failures)
            )

    def test_future_or_excessively_long_gate_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value, expected in (
                ("created_at", "2099-01-01T00:00:00.000Z", "in the future"),
                ("validity_seconds", 86_401.0, "no greater than 86400"),
            ):
                release, quality, websocket = self._reports(root)
                payload = json.loads(release.read_text(encoding="utf-8"))
                payload[field] = value
                write_json_report(release, payload)
                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=self._artifacts(root),
                )
                with self.subTest(field=field):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_model_id_is_bound_across_manifest_and_all_gate_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            manifest = next(
                path for kind, path in artifacts if kind == "model-manifest"
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["model_id"] = "wrong-model"
            write_json_report(manifest, payload)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )
            self.assertFalse(report.passed)
            self.assertTrue(
                any("model_id" in failure for failure in report.failures)
            )

    def test_installed_dependency_version_must_match_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            host_profile = next(
                path for kind, path in artifacts if kind == "host-profile"
            )
            payload = json.loads(host_profile.read_text(encoding="utf-8"))
            payload["installed_dependencies"]["websockets"]["version"] = "99.0"
            write_json_report(host_profile, payload)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )
            self.assertFalse(report.passed)
            self.assertTrue(
                any("does not match the lock" in failure for failure in report.failures)
            )

    def test_effective_configuration_evidence_cannot_be_weakened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for kind, expected in (
                ("host-profile", "canonical active systemd unit without drop-ins"),
                ("deployment-state", "canonical active systemd unit without drop-ins"),
            ):
                artifacts = self._artifacts(root)
                evidence = next(path for item, path in artifacts if item == kind)
                payload = json.loads(evidence.read_text(encoding="utf-8"))
                payload["effective_configuration"]["systemd"]["drop_in_paths"] = [
                    "/etc/systemd/system/turnalign.service.d/override.conf"
                ]
                write_json_report(evidence, payload)

                report = run_production_gate(
                    release,
                    quality,
                    websocket,
                    source_commit="b" * 40,
                    artifacts=artifacts,
                )

                with self.subTest(kind=kind):
                    self.assertFalse(report.passed)
                    self.assertIn(expected, "\n".join(report.failures))

    def test_systemd_effective_capture_requires_loaded_active_unit(self):
        snapshot = production_gate_module._EvidenceSnapshot("a" * 64, 123, b"unit")
        valid = (
            b"FragmentPath=/etc/systemd/system/turnalign.service\n"
            b"DropInPaths=\n"
            b"NeedDaemonReload=no\n"
            b"ActiveState=active\n"
            b"SubState=running\n"
        )
        with patch.object(
            production_gate_module,
            "_run_effective_config_command",
            return_value=(valid, b""),
        ):
            identity = production_gate_module._capture_systemd_effective_configuration(
                snapshot
            )
        self.assertEqual(identity["sha256"], "a" * 64)
        self.assertEqual(identity["drop_in_paths"], [])

        unsafe = valid.replace(
            b"DropInPaths=\n",
            b"DropInPaths=/etc/systemd/system/turnalign.service.d/override.conf\n",
        )
        with patch.object(
            production_gate_module,
            "_run_effective_config_command",
            return_value=(unsafe, b""),
        ), self.assertRaisesRegex(RuntimeError, "without drop-ins"):
            production_gate_module._capture_systemd_effective_configuration(snapshot)

    def test_nginx_effective_capture_requires_one_exact_loaded_file(self):
        content = b"server { listen 443 ssl; }\n"
        snapshot = production_gate_module._EvidenceSnapshot(
            hashlib.sha256(content).hexdigest(),
            len(content),
            content,
        )
        marker = b"# configuration file /etc/nginx/conf.d/turnalign.conf:\n"
        dump = (
            b"# configuration file /etc/nginx/nginx.conf:\nhttp {}\n\n"
            + marker
            + content
            + b"\n# configuration file /etc/nginx/mime.types:\ntypes {}\n"
        )
        with patch.object(
            production_gate_module,
            "_run_effective_config_command",
            return_value=(dump, b"syntax is ok\n"),
        ):
            identity = production_gate_module._capture_nginx_effective_configuration(
                snapshot
            )
        self.assertEqual(identity["loaded_occurrences"], 1)
        self.assertEqual(identity["sha256"], hashlib.sha256(content).hexdigest())

        with patch.object(
            production_gate_module,
            "_run_effective_config_command",
            return_value=(dump + marker + content, b""),
        ), self.assertRaisesRegex(RuntimeError, "load the canonical.*once"):
            production_gate_module._capture_nginx_effective_configuration(snapshot)

        with patch.object(
            production_gate_module,
            "_run_effective_config_command",
            return_value=(dump, b"nginx: [warn] conflicting server name\n"),
        ), self.assertRaisesRegex(RuntimeError, "contains warnings"):
            production_gate_module._capture_nginx_effective_configuration(snapshot)

    def test_production_configuration_rejects_a_writable_ancestor(self):
        path = Path("/etc/nginx/conf.d/turnalign.conf")

        def metadata(item):
            if Path(item) == path:
                return SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0)
            mode = 0o775 if Path(item) == Path("/etc/nginx") else 0o755
            return SimpleNamespace(st_mode=stat.S_IFDIR | mode, st_uid=0)

        with patch.object(
            production_gate_module.os,
            "lstat",
            side_effect=metadata,
        ), self.assertRaisesRegex(ValueError, "all ancestors"):
            production_gate_module._require_root_owned_production_config(path)

    def test_http_dependency_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            lock = next(
                path for kind, path in artifacts if kind == "dependency-lock"
            )
            lock.write_text(
                "--index-url http://pypi.example/simple\n"
                "websockets==17.1 \\\n"
                f"    --hash=sha256:{'c' * 64}\n",
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
            self.assertTrue(
                any("must use HTTPS" in failure for failure in report.failures)
            )

    def test_https_dependency_index_is_parsed_as_a_directive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            lock = next(
                path for kind, path in artifacts if kind == "dependency-lock"
            )
            lock.write_text(
                "--index-url https://pypi.example/simple\n"
                "websockets==17.1 \\\n"
                f"    --hash=sha256:{'c' * 64}\n",
                encoding="utf-8",
            )
            host_profile = next(
                path for kind, path in artifacts if kind == "host-profile"
            )
            profile = json.loads(host_profile.read_text(encoding="utf-8"))
            lock_entry = next(
                item
                for item in profile["artifacts"]
                if item["kind"] == "dependency-lock"
            )
            lock_entry["sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
            lock_entry["bytes"] = lock.stat().st_size
            write_json_report(host_profile, profile)

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )
            self.assertTrue(report.passed, report.failures)

    def test_conditional_dependency_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            lock = next(
                path for kind, path in artifacts if kind == "dependency-lock"
            )
            lock.write_text(
                "websockets==17.1; python_version >= '3.10' \\\n"
                f"    --hash=sha256:{'c' * 64}\n",
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
                "contain no environment markers",
                "\n".join(report.failures),
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
                ("wss://bad host.example.com/ws", "public wss://"),
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
                    lambda payload: payload.update(identity_consistent=False),
                    "consistent deployment identity",
                ),
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
                    lambda payload: payload["results"][0].update(
                        model_revision="c" * 40
                    ),
                    "lacks complete typed per-session evidence",
                ),
                (
                    lambda payload: payload.update(model_revision="c" * 40),
                    "websocket and release reports identify different model revisions",
                ),
                (
                    lambda payload: payload.update(
                        backend_implementation="different-backend"
                    ),
                    "websocket and release reports identify different backends",
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

    def test_loaded_model_path_is_bound_to_model_identity_and_file_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            payload = json.loads(release.read_text(encoding="utf-8"))
            payload["loaded_models"][0]["path"] = (
                "/var/lib/turnalign/models/different-model/different-name.bin"
            )
            write_json_report(release, payload)
            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=self._artifacts(root),
            )
            self.assertFalse(report.passed)
            self.assertIn(
                "loaded runtime model evidence does not exactly match",
                "\n".join(report.failures),
            )

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
                (
                    lambda path: self._write_test_wheel(
                        path,
                        source_commit="c" * 40,
                    ),
                    "source commit does not match",
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
                    lambda payload: payload.update(active_commit="c" * 40),
                    "did not capture the active candidate release",
                ),
                (
                    lambda payload: payload["platform"].update(
                        logical_cpu_count=True
                    ),
                    "complete typed platform evidence",
                ),
                (
                    lambda payload: payload["runtime"].update(
                        turnalign_source_commit="c" * 40
                    ),
                    "installed versioned Wheel runtime",
                ),
                (
                    lambda payload: payload["runtime"].update(
                        python_prefix="/opt/turnalign/current/venv"
                    ),
                    "installed versioned Wheel runtime",
                ),
                (
                    lambda payload: payload["platform"].update(system="Darwin"),
                    "complete typed platform evidence",
                ),
                (
                    lambda payload: payload["platform"].update(
                        boot_id="invalid"
                    ),
                    "complete typed platform evidence",
                ),
                (
                    lambda payload: payload["artifacts"][0].update(bytes=1),
                    "does not match the retained deployment artifacts",
                ),
                (
                    lambda payload: payload["installed_distribution"]["files"][
                        0
                    ].update(sha256="0" * 64),
                    "installed package files do not exactly match",
                ),
                (
                    lambda payload: payload["installed_distribution"]["files"].append({
                        "name": "turnalign/injected.py",
                        "sha256": "0" * 64,
                        "bytes": 1,
                    }),
                    "installed package files do not exactly match",
                ),
                (
                    lambda payload: payload["installed_distribution"].update(
                        root=(
                            "/opt/turnalign/current/venv/lib/"
                            "python3.12/site-packages"
                        )
                    ),
                    "installed package files do not exactly match",
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

    def test_rejects_forged_or_incomplete_rollback_rehearsal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for mutate, expected in (
                (
                    lambda payload: payload.update(status="failed"),
                    "rollback rehearsal report did not pass",
                ),
                (
                    lambda payload: payload.update(previous_commit="b" * 40),
                    "not bound to one prior and candidate release",
                ),
                (
                    lambda payload: payload.update(transaction_id="d" * 63),
                    "invalid transaction identity",
                ),
                (
                    lambda payload: payload.update(transaction_path="/tmp/pending"),
                    "invalid transaction identity",
                ),
                (
                    lambda payload: payload.update(final_active_commit="c" * 40),
                    "not bound to one prior and candidate release",
                ),
                (
                    lambda payload: payload.update(
                        boot_id="87654321-4321-4321-8321-cba987654321"
                    ),
                    "identify different Linux boots",
                ),
                (
                    lambda payload: payload.update(current_link="/tmp/current"),
                    "production activation layout",
                ),
                (
                    lambda payload: payload["rollback"].update(
                        from_commit="c" * 40
                    ),
                    "rollback has an invalid transition",
                ),
                (
                    lambda payload: payload["rollback"]["restart"].update(
                        restart_exit_code=False
                    ),
                    "did not restart an active service",
                ),
                (
                    lambda payload: payload["restore"]["readiness"].update(
                        preloaded=False
                    ),
                    "did not prove preloaded readiness",
                ),
                (
                    lambda payload: payload["rollback"][
                        "websocket_report"
                    ].update(source_commit="b" * 40),
                    "not bound to source commit",
                ),
                (
                    lambda payload: payload["restore"].update(
                        started_at="2026-09-01T07:59:59.000Z"
                    ),
                    "phase timestamps are out of order",
                ),
            ):
                artifacts = self._artifacts(root)
                rehearsal_path = next(
                    path
                    for kind, path in artifacts
                    if kind == "rollback-rehearsal"
                )
                payload = json.loads(rehearsal_path.read_text(encoding="utf-8"))
                mutate(payload)
                write_json_report(rehearsal_path, payload)
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

    def test_rejects_forged_or_inconsistent_deployment_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for mutate, expected in (
                (
                    lambda payload: payload.update(status="failed"),
                    "deployment activation report did not pass",
                ),
                (
                    lambda payload: payload.update(transaction_id="e" * 63),
                    "invalid transaction identity",
                ),
                (
                    lambda payload: payload.update(transaction_path="/tmp/pending"),
                    "invalid transaction identity",
                ),
                (
                    lambda payload: payload.update(initial_active_commit="b" * 40),
                    "not bound to one prior and candidate release",
                ),
                (
                    lambda payload: payload.update(final_active_commit="c" * 40),
                    "not bound to one prior and candidate release",
                ),
                (
                    lambda payload: payload["activation"].update(
                        from_commit="b" * 40
                    ),
                    "activate has an invalid transition",
                ),
                (
                    lambda payload: payload.update(rollback={}),
                    "unexpectedly contains rollback evidence",
                ),
                (
                    lambda payload: payload["activation"][
                        "websocket_report"
                    ].update(model_revision="d" * 40),
                    "selected a different model identity",
                ),
                (
                    lambda payload: payload.update(
                        boot_id="87654321-4321-4321-8321-cba987654321"
                    ),
                    "host profile and deployment activation identify different",
                ),
            ):
                artifacts = self._artifacts(root)
                activation_path = next(
                    path
                    for kind, path in artifacts
                    if kind == "deployment-activation"
                )
                payload = json.loads(activation_path.read_text(encoding="utf-8"))
                mutate(payload)
                write_json_report(activation_path, payload)
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

    def test_rejects_activation_and_rehearsal_for_different_release_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            activation_path = next(
                path for kind, path in artifacts if kind == "deployment-activation"
            )
            payload = json.loads(activation_path.read_text(encoding="utf-8"))
            different_previous = "d" * 40
            payload["previous_commit"] = different_previous
            payload["initial_active_commit"] = different_previous
            payload["activation"]["from_commit"] = different_previous
            write_json_report(activation_path, payload)

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            self.assertIn(
                "deployment activation and rollback rehearsal identify different "
                "release pairs or Linux boots",
                report.failures,
            )

    def test_runtime_identity_requires_the_bound_versioned_wheel(self):
        source_commit = "b" * 40
        prefix = f"/opt/turnalign/releases/{source_commit}/venv"
        flags_patch = patch.object(
            production_gate_module.sys,
            "flags",
            SimpleNamespace(isolated=1),
        )
        bytecode_patch = patch.object(
            production_gate_module.sys,
            "dont_write_bytecode",
            True,
        )
        flags_patch.start()
        bytecode_patch.start()
        self.addCleanup(flags_patch.stop)
        self.addCleanup(bytecode_patch.stop)

        class SourceIdentity:
            def __init__(self, value: str):
                self.value = value

            def __str__(self) -> str:
                version = production_gate_module.sys.version_info
                return (
                    f"{prefix}/lib/python{version.major}.{version.minor}/"
                    "site-packages/turnalign"
                )

            def joinpath(self, _name: str):
                return self

            def read_text(self, *, encoding: str) -> str:
                self.assert_encoding = encoding
                return self.value

        with patch.object(
            production_gate_module.os.path,
            "abspath",
            side_effect=lambda value: value,
        ), patch.object(production_gate_module.sys, "prefix", prefix), patch.object(
            production_gate_module.sys,
            "executable",
            f"{prefix}/bin/python",
        ), patch.object(
            production_gate_module.importlib.resources,
            "files",
            return_value=SourceIdentity(f"{source_commit}\n"),
        ):
            identity = production_gate_module._installed_runtime_identity()
        self.assertEqual(identity["turnalign_source_commit"], source_commit)
        self.assertEqual(identity["python_prefix"], prefix)

        with patch.object(
            production_gate_module.sys, "prefix", "/opt/turnalign/current/venv"
        ), patch.object(
            production_gate_module.importlib.resources,
            "files",
            return_value=SourceIdentity(f"{source_commit}\n"),
        ), self.assertRaisesRegex(ValueError, "versioned production"):
            production_gate_module._installed_runtime_identity(source_commit)

        with patch.object(
            production_gate_module.os.path,
            "abspath",
            side_effect=lambda value: value,
        ), patch.object(production_gate_module.sys, "prefix", prefix), patch.object(
            production_gate_module.sys,
            "executable",
            f"{prefix}/bin/python",
        ), patch.object(
            production_gate_module.importlib.resources,
            "files",
            return_value=SourceIdentity("unbound\n"),
        ), self.assertRaisesRegex(ValueError, "source identity"):
            production_gate_module._installed_runtime_identity(source_commit)

        with patch.object(
            production_gate_module.importlib.resources,
            "files",
            return_value=SourceIdentity(f"{'c' * 40}\n"),
        ), self.assertRaisesRegex(ValueError, "does not match the candidate"):
            production_gate_module._installed_runtime_identity(source_commit)

    def test_runtime_identity_requires_isolated_no_bytecode_python(self):
        with patch.object(
            production_gate_module.sys,
            "flags",
            SimpleNamespace(isolated=0),
        ), patch.object(
            production_gate_module.sys,
            "dont_write_bytecode",
            False,
        ), self.assertRaisesRegex(ValueError, "Python -I -B"):
            production_gate_module._installed_runtime_identity("b" * 40)

    @unittest.skipUnless(
        production_gate_module.os.name == "posix",
        "installed production package validation is POSIX-only",
    )
    def test_installed_distribution_hashes_only_secure_active_package_files(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve() / "venv"
            version = production_gate_module.sys.version_info
            root = (
                prefix
                / "lib"
                / f"python{version.major}.{version.minor}"
                / "site-packages"
            )
            package = root / "turnalign"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                '__version__ = "0.1.0"\n', encoding="utf-8"
            )
            (package / "_source_commit.txt").write_text(
                f"{'b' * 40}\n", encoding="ascii"
            )

            class Distribution:
                version = "0.1.0"

                @staticmethod
                def locate_file(_path: str) -> Path:
                    return root

            runtime = {
                "python_executable": f"{prefix}/bin/python",
                "python_prefix": str(prefix),
                "turnalign_source_commit": "b" * 40,
                "turnalign_version": "0.1.0",
            }
            with patch.object(
                production_gate_module.importlib.metadata,
                "distribution",
                return_value=Distribution(),
            ), patch.object(
                production_gate_module,
                "_root_owned_immutable",
                return_value=True,
            ):
                identity = production_gate_module._installed_distribution_identity(
                    runtime
                )
                self.assertEqual(
                    [item["name"] for item in identity["files"]],
                    [
                        "turnalign/__init__.py",
                        "turnalign/_source_commit.txt",
                    ],
                )
                (package / "injected.py").symlink_to(package / "__init__.py")
                with self.assertRaisesRegex(ValueError, "unsafe or mutable"):
                    production_gate_module._installed_distribution_identity(runtime)

    def test_installed_dependency_uses_a_bounded_tree_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve() / "venv"
            version = production_gate_module.sys.version_info
            site_packages = (
                prefix
                / "lib"
                / f"python{version.major}.{version.minor}"
                / "site-packages"
            )
            package = site_packages / "websockets"
            package.mkdir(parents=True)
            (package / "__init__.py").write_bytes(b"first\n")
            (package / "client.py").write_bytes(b"second\n")
            lock = Path(directory) / "requirements.lock"
            lock.write_text(
                "websockets==17.1 \\\n"
                f"    --hash=sha256:{'c' * 64}\n",
                encoding="utf-8",
            )

            class Distribution:
                version = "17.1"
                files = (
                    Path("websockets/client.py"),
                    Path("websockets/__init__.py"),
                )

                @staticmethod
                def locate_file(_name):
                    return site_packages

            with patch.object(
                production_gate_module.importlib.metadata,
                "distribution",
                return_value=Distribution(),
            ), patch.object(
                production_gate_module,
                "_root_owned_immutable",
                return_value=True,
            ):
                identity = production_gate_module._installed_dependency_identity(
                    lock,
                    {"python_prefix": str(prefix)},
                )["websockets"]

            self.assertEqual(identity["file_count"], 2)
            self.assertEqual(identity["bytes"], len(b"first\nsecond\n"))
            self.assertRegex(identity["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("files", identity)

    def test_host_profile_generation_requires_linux(self):
        runtime_prefix = f"/opt/turnalign/releases/{'b' * 40}/venv"
        with patch.object(
            production_gate_module,
            "_installed_runtime_identity",
            return_value={
                "python_executable": f"{runtime_prefix}/bin/python",
                "python_prefix": runtime_prefix,
                "turnalign_source_commit": "b" * 40,
                "turnalign_version": "0.1.0",
            },
        ), patch.object(
            production_gate_module.platform,
            "system",
            return_value="Darwin",
        ), self.assertRaisesRegex(RuntimeError, "Linux production host"):
            create_host_profile(None, [])

    def test_deployment_state_recaptures_effective_configuration(self):
        effective = {
            "systemd": {"active_state": "active"},
            "nginx": {"loaded_occurrences": 1},
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            production_gate_module.platform,
            "system",
            return_value="Linux",
        ), patch.object(
            production_gate_module.os,
            "geteuid",
            return_value=0,
            create=True,
        ), patch.object(
            production_gate_module,
            "_active_release_commit",
            return_value="b" * 40,
        ), patch.object(
            production_gate_module,
            "_read_linux_boot_id",
            return_value=self.BOOT_ID,
        ), patch.object(
            production_gate_module,
            "_capture_effective_configuration",
            return_value=effective,
        ), patch.object(
            production_gate_module,
            "_DEPLOYMENT_TRANSACTION_PATH",
            Path(directory) / "missing.json",
        ), patch.object(
            production_gate_module,
            "_acquire_deployment_lock",
            side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
        ):
            payload = create_deployment_state(120.0)

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["effective_configuration"], effective)
        self.assertEqual(payload["active_commit"], "b" * 40)
        self.assertEqual(payload["validity_seconds"], 120.0)

    def test_host_profile_refuses_pending_deployment_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            pending = Path(directory) / "pending-activation.json"
            pending.write_text("{}\n", encoding="utf-8")
            with patch.object(
                production_gate_module.platform,
                "system",
                return_value="Linux",
            ), patch.object(
                production_gate_module,
                "_DEPLOYMENT_TRANSACTION_PATH",
                pending,
            ), patch.object(
                production_gate_module,
                "_acquire_deployment_lock",
                side_effect=lambda: os.open(os.devnull, os.O_RDONLY),
            ), self.assertRaisesRegex(RuntimeError, "pending deployment"):
                create_host_profile(None, [])

    def test_host_profile_holds_deployment_lock_through_capture(self):
        descriptor = os.open(os.devnull, os.O_RDONLY)
        events = []

        def reject_pending():
            events.append("checked")

        def capture(source_commit, artifacts, system):
            self.assertEqual(events, ["checked"])
            self.assertEqual(source_commit, "b" * 40)
            self.assertEqual(artifacts, [])
            self.assertEqual(system, "Linux")
            os.fstat(descriptor)
            events.append("captured")
            return {"schema_version": 7}

        with patch.object(
            production_gate_module.platform,
            "system",
            return_value="Linux",
        ), patch.object(
            production_gate_module,
            "_acquire_deployment_lock",
            return_value=descriptor,
        ), patch.object(
            production_gate_module,
            "_reject_pending_deployment_transaction",
            side_effect=reject_pending,
        ), patch.object(
            production_gate_module,
            "_create_host_profile_locked",
            side_effect=capture,
        ):
            self.assertEqual(
                create_host_profile("b" * 40, []),
                {"schema_version": 7},
            )
        self.assertEqual(events, ["checked", "captured"])
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    @unittest.skipUnless(
        os.name == "posix",
        "deployment lock validation is POSIX-only",
    )
    def test_host_profile_deployment_lock_is_root_only_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.lock"

            def root_metadata(descriptor):
                metadata = real_fstat(descriptor)
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_uid=0,
                    st_nlink=metadata.st_nlink,
                )

            with patch.object(
                production_gate_module,
                "_DEPLOYMENT_LOCK_PATH",
                path,
            ), patch.object(
                production_gate_module.os,
                "fstat",
                side_effect=root_metadata,
            ):
                first = production_gate_module._acquire_deployment_lock()
                try:
                    with self.assertRaisesRegex(RuntimeError, "operation is active"):
                        production_gate_module._acquire_deployment_lock()
                finally:
                    os.close(first)
                second = production_gate_module._acquire_deployment_lock()
                os.close(second)

    @unittest.skipUnless(
        os.name == "posix",
        "active release link validation is POSIX-only",
    )
    def test_host_profile_requires_a_canonical_active_candidate_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = root / "releases"
            current = root / "current"
            commit = "b" * 40
            (releases / commit).mkdir(parents=True)
            current.symlink_to(releases / commit)
            with patch.object(
                production_gate_module,
                "_RELEASE_ROOT",
                releases,
            ), patch.object(
                production_gate_module,
                "_CURRENT_RELEASE_LINK",
                current,
            ), patch.object(
                production_gate_module,
                "_required_deployment_owner",
                return_value=os.getuid(),
            ), patch.object(
                production_gate_module,
                "_root_owned_immutable",
                return_value=True,
            ):
                self.assertEqual(
                    production_gate_module._active_release_commit(),
                    commit,
                )
                current.unlink()
                current.symlink_to(Path("releases") / commit)
                with self.assertRaisesRegex(ValueError, "absolute target"):
                    production_gate_module._active_release_commit()

    def test_rejects_weakened_or_comment_only_systemd_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for replace, expected in (
                (
                    ("IPAddressDeny=any", "# IPAddressDeny=any"),
                    "deny non-allowlisted network traffic",
                ),
                (
                    ("--host 127.0.0.1", "--host 0.0.0.0"),
                    "bind TurnAlign only to loopback",
                ),
                (
                    (
                        "--auth-token-file ${CREDENTIALS_DIRECTORY}/auth-token",
                        "--auth-token-file /tmp/token",
                    ),
                    "credential directory",
                ),
                (
                    ("ProtectSystem=strict", "ProtectSystem=full"),
                    "ProtectSystem=strict",
                ),
                (
                    (
                        "/opt/turnalign/current/venv/bin/python -I -B -u",
                        "/opt/turnalign/venv/bin/python -I -B -u",
                    ),
                    "isolated versioned Python runtime",
                ),
                (
                    ("bin/python -I -B -u -m", "bin/python -B -u -m"),
                    "isolated versioned Python runtime",
                ),
                (
                    ("bin/python -I -B -u -m", "bin/python -I -u -m"),
                    "isolated versioned Python runtime",
                ),
                (
                    ("bin/python -I -B -u -m", "bin/python -I -B -m"),
                    "isolated versioned Python runtime",
                ),
                (
                    ("  --require-local-model \\\n", ""),
                    "require retained local model files",
                ),
                (
                    (
                        "/var/lib/turnalign/models/paraformer-zh-streaming",
                        "/tmp/paraformer-zh-streaming",
                    ),
                    "canonical child of /var/lib/turnalign/models",
                ),
                (
                    ("  --backend-cancel-timeout 5 \\\n", ""),
                    "--backend-cancel-timeout exactly once",
                ),
                (
                    ("StateDirectory=turnalign-state", "StateDirectory=turnalign"),
                    "must not place model files under its writable StateDirectory",
                ),
                (
                    (
                        (
                            "  --backend-option model_revision="
                            "562b758fecc801f13079d846d06b0b024fd670c4 \\\n"
                        ),
                        "",
                    ),
                    "must pin one immutable model_revision",
                ),
            ):
                artifacts = self._artifacts(root)
                service = next(
                    path for kind, path in artifacts if kind == "service-unit"
                )
                content = service.read_text(encoding="utf-8")
                service.write_text(content.replace(*replace), encoding="utf-8")
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

    def test_rejects_weakened_or_mismatched_nginx_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            for replace, expected in (
                (
                    ("proxy_next_upstream off;", "# proxy_next_upstream off;"),
                    "upstream retry disablement",
                ),
                (
                    ("server 127.0.0.1:8765;", "server 10.0.0.2:8765;"),
                    "only a loopback TCP server",
                ),
                (
                    (
                        "ssl_protocols TLSv1.2 TLSv1.3;",
                        "ssl_protocols TLSv1 TLSv1.2;",
                    ),
                    "only TLSv1.2 and TLSv1.3",
                ),
                (
                    ("return 404;", "proxy_pass http://turnalign_backend/metrics;"),
                    "keep metrics private",
                ),
                (
                    ("proxy_read_timeout 330s;", "proxy_read_timeout 30s;"),
                    "read timeout",
                ),
                (
                    ("server_name asr.example.com;", "server_name other.example.com;"),
                    "one server for the probed host",
                ),
                (
                    (
                        "proxy_buffering off;",
                        "proxy_buffering off;\n        include /tmp/unsafe.conf;",
                    ),
                    "self-contained without include directives",
                ),
                (
                    (
                        "proxy_set_header Upgrade $http_upgrade;",
                        (
                            "proxy_set_header Upgrade $http_upgrade;\n"
                            "        proxy_set_header Upgrade close;"
                        ),
                    ),
                    "Upgrade forwarding",
                ),
                (
                    (
                        (
                            "limit_req_zone $binary_remote_addr "
                            "zone=turnalign_handshake:10m rate=5r/s;"
                        ),
                        "# limit_req_zone removed",
                    ),
                    "handshake rate-limit zone",
                ),
            ):
                artifacts = self._artifacts(root)
                nginx = next(
                    path for kind, path in artifacts if kind == "nginx-config"
                )
                content = nginx.read_text(encoding="utf-8")
                self.assertIn(replace[0], content)
                nginx.write_text(content.replace(*replace), encoding="utf-8")
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

    def test_rejects_unlocked_or_build_only_sbom_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, quality, websocket = self._reports(root)
            artifacts = self._artifacts(root)
            sbom = next(path for kind, path in artifacts if kind == "sbom")
            payload = json.loads(sbom.read_text(encoding="utf-8"))
            payload["components"].append({
                "bom-ref": "pip==26.0.1",
                "name": "pip",
                "version": "26.0.1",
                "purl": "pkg:pypi/pip@26.0.1",
            })
            write_json_report(sbom, payload)

            report = run_production_gate(
                release,
                quality,
                websocket,
                source_commit="b" * 40,
                artifacts=artifacts,
            )

            self.assertFalse(report.passed)
            failures = "\n".join(report.failures)
            self.assertIn("build-only tooling", failures)
            self.assertIn("unlocked runtime component: pip==26.0.1", failures)

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
