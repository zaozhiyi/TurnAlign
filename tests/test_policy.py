import unittest

from turnalign.model_pool import BackendPoolCapacityError
from turnalign.policy import ServerBusyError, ServerPolicy
from turnalign.recovery import RecoveryCapacityError, RecoveryConflictError


class ServerPolicyTests(unittest.TestCase):
    def test_default_policy_allows_only_server_selected_backend_and_model(self):
        policy = ServerPolicy.defaults("funasr", "approved-model")
        self.assertEqual(
            policy.validate_start(
                {"type": "start", "backend": "funasr", "model": "approved-model"},
                default_backend="funasr",
                default_model="approved-model",
            ),
            ("funasr", "approved-model", None, None),
        )
        with self.assertRaisesRegex(ValueError, "backend"):
            policy.validate_start(
                {"backend": "whisper-cpp"},
                default_backend="funasr",
                default_model="approved-model",
            )
        with self.assertRaisesRegex(ValueError, "model"):
            policy.validate_start(
                {"model": "unapproved-model"},
                default_backend="funasr",
                default_model="approved-model",
            )

    def test_default_policy_rejects_client_paths_and_components(self):
        policy = ServerPolicy.defaults("whisper-cpp")
        with self.assertRaisesRegex(ValueError, "paths"):
            policy.validate_start(
                {"executable": "/tmp/untrusted"},
                default_backend="whisper-cpp",
                default_model=None,
            )
        with self.assertRaisesRegex(ValueError, "component"):
            policy.validate_start(
                {"aligner": "paraformer"},
                default_backend="whisper-cpp",
                default_model=None,
            )
        component_policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            allowed_components=frozenset({"paraformer"}),
        )
        with self.assertRaisesRegex(ValueError, "component options"):
            component_policy.validate_start(
                {"aligner": "paraformer", "aligner_options": {"device": "cuda"}},
                default_backend="fake",
                default_model=None,
            )

    def test_language_and_compute_variations_require_allowlisting(self):
        policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            allowed_languages=frozenset({"en"}),
            allowed_compute_types=frozenset({"int8"}),
        )
        self.assertEqual(
            policy.validate_start(
                {"language": "zh", "compute_type": "float16"},
                default_backend="fake",
                default_model=None,
                default_language="zh",
                default_compute_type="float16",
            ),
            ("fake", None, "zh", "float16"),
        )
        self.assertEqual(
            policy.validate_start(
                {"language": "en", "compute_type": "int8"},
                default_backend="fake",
                default_model=None,
                default_language="zh",
                default_compute_type="float16",
            ),
            ("fake", None, "en", "int8"),
        )
        with self.assertRaisesRegex(ValueError, "language"):
            policy.validate_start(
                {"language": "fr"},
                default_backend="fake",
                default_model=None,
                default_language="zh",
            )
        with self.assertRaisesRegex(ValueError, "compute"):
            policy.validate_start(
                {"compute_type": "float32"},
                default_backend="fake",
                default_model=None,
                default_compute_type="float16",
            )

    def test_remote_bind_requires_explicit_opt_in(self):
        policy = ServerPolicy.defaults("fake")
        policy.validate_bind("127.0.0.1")
        policy.validate_bind("::1")
        with self.assertRaisesRegex(ValueError, "allow_remote"):
            policy.validate_bind("0.0.0.0")

    def test_session_limit_must_be_finite_and_positive(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ServerPolicy(max_session_seconds=value)

    def test_policy_and_request_identity_fields_require_strict_types(self):
        for options, expected in (
            ({"allowed_backends": "fake"}, "frozenset"),
            ({"allowed_models": frozenset({1})}, "must be a string"),
            ({"allowed_components": frozenset({" bad "})}, "trimmed string"),
            ({"allow_remote": 1}, "must be a boolean"),
            ({"allow_client_paths": None}, "must be a boolean"),
            ({"redact_errors": 0}, "must be a boolean"),
            ({"max_session_seconds": True}, "must be a number"),
            ({"max_session_seconds": "60"}, "must be a number"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(
                (TypeError, ValueError), expected
            ):
                ServerPolicy(**options)

        policy = ServerPolicy(
            allowed_backends=frozenset({"fake", "True"}),
            allowed_models=frozenset({"7"}),
            allowed_languages=frozenset({"7"}),
            allowed_compute_types=frozenset({"7"}),
            allowed_components=frozenset({"7"}),
        )
        for request, expected in (
            ({"backend": True}, "backend must be a string"),
            ({"backend": ""}, "non-empty trimmed string"),
            ({"model": 7}, "model must be a string"),
            ({"language": 7}, "language must be a string"),
            ({"compute_type": 7}, "compute_type must be a string"),
            ({"aligner": 7}, "aligner must be a string"),
        ):
            with self.subTest(request=request), self.assertRaisesRegex(
                (TypeError, ValueError), expected
            ):
                policy.validate_start(
                    request,
                    default_backend="fake",
                    default_model=None,
                )

    def test_authentication_uses_public_generic_error(self):
        policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            auth_token="secret",
        )
        with self.assertRaises(PermissionError) as raised:
            policy.validate_start({}, default_backend="fake", default_model=None)
        self.assertEqual(policy.public_error(raised.exception)["code"], "unauthorized")
        self.assertNotIn("secret", str(policy.public_error(raised.exception)))

    def test_authentication_supports_utf8_and_rejects_invalid_configuration(self):
        policy = ServerPolicy(
            allowed_backends=frozenset({"fake"}),
            auth_token="私密令牌",
        )
        self.assertEqual(
            policy.validate_start(
                {"auth": "私密令牌"},
                default_backend="fake",
                default_model=None,
            ),
            ("fake", None, None, None),
        )
        with self.assertRaises(PermissionError):
            policy.validate_start(
                {"auth": "私密令牌2"},
                default_backend="fake",
                default_model=None,
            )
        for token in ("", "first\nsecond", "nul\x00token", "密" * 3_000):
            with self.subTest(token=token[:20]), self.assertRaises(ValueError):
                ServerPolicy(auth_token=token)

    def test_internal_paths_are_redacted(self):
        policy = ServerPolicy.defaults("fake")
        payload = policy.public_error(RuntimeError("failed at /Users/private/model.bin"))
        self.assertEqual(payload["code"], "session_error")
        self.assertNotIn("/Users/private", payload["message"])

    def test_capacity_and_timeout_errors_have_actionable_public_codes(self):
        policy = ServerPolicy()
        self.assertEqual(policy.public_error(ServerBusyError())["code"], "server_busy")
        self.assertEqual(
            policy.public_error(BackendPoolCapacityError())["code"],
            "server_busy",
        )
        self.assertEqual(policy.public_error(RecoveryCapacityError())["code"], "server_busy")
        self.assertEqual(
            policy.public_error(RecoveryConflictError())["code"],
            "session_conflict",
        )
        self.assertEqual(policy.public_error(TimeoutError())["code"], "timeout")


if __name__ == "__main__":
    unittest.main()
