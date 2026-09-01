import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "deploy" / "systemd" / "turnalign.service"
NGINX = ROOT / "deploy" / "nginx" / "turnalign.conf.example"
LEGACY_ENVIRONMENT = ROOT / "deploy" / "systemd" / "turnalign.env.example"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def option_number(content: str, option: str) -> float:
    match = re.search(rf"--{re.escape(option)}\s+(\d+(?:\.\d+)?)", content)
    if match is None:
        raise AssertionError(f"missing deployment option: --{option}")
    return float(match.group(1))


class DeploymentArtifactTests(unittest.TestCase):
    def test_service_is_local_authenticated_and_fail_closed(self):
        content = SERVICE.read_text(encoding="utf-8")
        for setting in (
            "--host 127.0.0.1",
            (
                "ExecStart=/opt/turnalign/current/venv/bin/python "
                "-I -B -u -m turnalign.cli serve"
            ),
            "--preload",
            "--model paraformer-zh-streaming",
            "--model-path /var/lib/turnalign/models/paraformer-zh-streaming",
            "--require-local-model",
            "--backend-option model_revision=562b758fecc801f13079d846d06b0b024fd670c4",
            "--warmup-file /var/lib/turnalign/warmup.wav",
            "--require-immutable-model-revision",
            "--backend-cancel-timeout 5",
            "WorkingDirectory=/var/lib/turnalign-state",
            "StateDirectory=turnalign-state",
            "Environment=HOME=/var/lib/turnalign-state",
            "LoadCredential=auth-token:/etc/turnalign/auth-token",
            "--auth-token-file ${CREDENTIALS_DIRECTORY}/auth-token",
            "ExecStartPre=/usr/bin/test -r /var/lib/turnalign/warmup.wav",
            "ExecStartPre=/usr/bin/test -s ${CREDENTIALS_DIRECTORY}/auth-token",
        ):
            self.assertIn(setting, content)
        self.assertNotIn("--allow-remote", content)
        self.assertNotIn("StateDirectory=turnalign\n", content)

    def test_service_shutdown_and_recovery_limits_are_consistent(self):
        content = SERVICE.read_text(encoding="utf-8")
        grace = option_number(content, "shutdown-grace-timeout")
        stop = float(re.search(r"TimeoutStopSec=(\d+(?:\.\d+)?)s", content).group(1))
        self.assertGreater(stop, grace)
        per_session = option_number(content, "max-recovery-audio-mib")
        total = option_number(content, "max-recovery-total-mib")
        self.assertLessEqual(per_session, total)
        raw_audio_bytes = option_number(content, "max-session-seconds") * 32_000
        self.assertLessEqual(raw_audio_bytes, per_session * 1024 * 1024)

    def test_service_uses_a_restricted_cpu_security_profile(self):
        content = SERVICE.read_text(encoding="utf-8")
        for setting in (
            "User=turnalign",
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "PrivateDevices=true",
            "ProtectSystem=strict",
            "ProtectHome=true",
            "CapabilityBoundingSet=",
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
            "IPAddressDeny=any",
            "IPAddressAllow=localhost",
        ):
            self.assertIn(setting, content)

    def test_nginx_preserves_websocket_upgrade_and_disables_retry(self):
        content = NGINX.read_text(encoding="utf-8")
        for setting in (
            "proxy_set_header Upgrade $http_upgrade;",
            "proxy_set_header Connection $turnalign_connection_upgrade;",
            "proxy_buffering off;",
            "proxy_request_buffering off;",
            "proxy_next_upstream off;",
            "limit_req zone=turnalign_handshake",
            "limit_conn turnalign_connections",
        ):
            self.assertIn(setting, content)

    def test_nginx_timeouts_cover_application_lifecycle_deadlines(self):
        service = SERVICE.read_text(encoding="utf-8")
        nginx = NGINX.read_text(encoding="utf-8")
        idle = option_number(service, "client-idle-timeout")
        initialization = option_number(service, "initialization-timeout")
        finalization = option_number(service, "finalization-timeout")
        read_timeout = float(re.search(r"proxy_read_timeout\s+(\d+(?:\.\d+)?)s", nginx).group(1))
        send_timeout = float(re.search(r"proxy_send_timeout\s+(\d+(?:\.\d+)?)s", nginx).group(1))
        self.assertGreater(read_timeout, max(idle, initialization, finalization))
        self.assertGreater(send_timeout, idle)
        self.assertIn("proxy_pass http://turnalign_backend/healthz;", nginx)
        self.assertIn("proxy_pass http://turnalign_backend/readyz;", nginx)
        self.assertRegex(nginx, r"location = /metrics\s*\{[^}]*return 404;")
        self.assertNotIn("proxy_pass http://turnalign_backend/metrics", nginx)

    def test_secret_is_not_stored_in_the_deployment_bundle(self):
        self.assertFalse(LEGACY_ENVIRONMENT.exists())
        service = SERVICE.read_text(encoding="utf-8")
        self.assertNotRegex(service, r"sk-[A-Za-z0-9_-]{16,}")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"/deploy"', pyproject)
        self.assertIn('"/hatch_build.py"', pyproject)
        self.assertIn("[tool.hatch.build.targets.wheel.hooks.custom]", pyproject)

    def test_release_workflow_is_upstream_tag_only_and_attested(self):
        content = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        for requirement in (
            'tags: ["v*.*.*"]',
            "pull_request:",
            (
                "if: github.event_name == 'push' && github.repository == "
                "'GuanZhengPM/TurnAlign'"
            ),
            "artifact-metadata: write",
            "attestations: write",
            "id-token: write",
            "persist-credentials: false",
            "pip==26.0.1",
            "setuptools==80.9.0",
            'git merge-base --is-ancestor "$GITHUB_SHA" origin/main',
            "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6",
            "sbom-path: dist-a/sbom.cdx.json",
            'gh attestation verify "$artifact"',
            "retention-days: 90",
        ):
            self.assertIn(requirement, content)

    def test_distribution_sbom_environment_is_reproducible_and_runtime_only(self):
        requirement = (
            "sbom-env/bin/python -m pip install --upgrade \\\n"
            "            pip==26.0.1 setuptools==80.9.0"
        )
        for workflow in (CI_WORKFLOW, RELEASE_WORKFLOW):
            content = workflow.read_text(encoding="utf-8")
            self.assertIn(requirement, content)
            self.assertIn(
                "sbom-env/bin/python -m pip uninstall --yes pip setuptools",
                content,
            )


if __name__ == "__main__":
    unittest.main()
