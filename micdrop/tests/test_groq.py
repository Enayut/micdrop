"""Tests for the Groq cloud provider.

No test here makes a real API call. The Groq client is injected as a fake, so
the suite is deterministic, offline and instant.

What is actually under test:

* the provider is a real, registered, mode-bound ``TextProcessor``
* a missing API key fails clearly and never falls back to another provider
* the API key never appears in a log line, an exception or a ``repr``
* API errors (auth, network, rate limit, unknown model) produce distinct,
  actionable ``ProcessingError`` messages
* a network failure is reported as a failure, never as a silent provider swap
"""

import logging
import os
import unittest
from unittest import mock

from ..app.config import GROQ_API_KEY_ENV_VAR, AppConfig
from ..app.errors import ModeMismatchError, ProcessingError
from ..app.modes import Mode
from ..refinement import create_processor, create_router
from ..refinement.groq import GroqProcessor, _redact


# --------------------------------------------------------------------------- #
# Fake Groq client
# --------------------------------------------------------------------------- #


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content="generated text", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeCompletion(self.content)


class FakeClient:
    """Stands in for groq.Groq. Records requests, returns canned text."""

    def __init__(self, content="generated text", error=None):
        self.completions = FakeCompletions(content, error)
        self.chat = mock.Mock()
        self.chat.completions = self.completions


def _groq_exc(name):
    """Build an exception type named like the real groq client raises."""
    return type(name, (Exception,), {})


def _config(**models):
    data = {"models": {"refine_provider": "groq", "generate_provider": "groq"}}
    if models:
        data["groq"] = models
    return AppConfig.from_dict(data)


class EnvMixin(unittest.TestCase):
    """Keep the developer's real API key out of the tests."""

    def setUp(self):
        for name in (GROQ_API_KEY_ENV_VAR, "GROQ_API_KEY"):
            self._saved = getattr(self, "_saved", {})
            self._saved[name] = os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# --------------------------------------------------------------------------- #
# Registration and mode binding
# --------------------------------------------------------------------------- #


class TestGroqIsARealProvider(EnvMixin):
    def test_built_from_config(self):
        processor = create_processor(_config(), Mode.GENERATE)
        self.assertIsInstance(processor, GroqProcessor)

    def test_can_back_either_mode(self):
        processor = create_processor(_config(), Mode.REFINE)
        self.assertEqual(processor.mode, Mode.REFINE)

    def test_is_mode_bound(self):
        processor = GroqProcessor(_config(), Mode.GENERATE, client=FakeClient())
        with self.assertRaises(ModeMismatchError):
            processor.process("hello", Mode.REFINE)

    def test_still_cannot_borrow_the_other_mode_via_the_router(self):
        """The Round 2 invariant must survive adding a cloud provider."""
        from ..app.errors import ModeProcessorUnavailableError
        from ..refinement.router import ModeProcessorRouter

        config = AppConfig.from_dict({"models": {"refine_provider": "groq"}})
        # build only the REFINE half, then ask for GENERATE.
        router = ModeProcessorRouter(
            {Mode.REFINE: create_processor(config, Mode.REFINE)}
        )
        with self.assertRaises(ModeProcessorUnavailableError):
            router.process("write an email", Mode.GENERATE)

    def test_uses_the_mode_own_system_prompt(self):
        from ..refinement.prompts import GENERATE_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT

        self.assertEqual(
            GroqProcessor(_config(), Mode.REFINE, client=FakeClient()).system_prompt,
            REFINE_SYSTEM_PROMPT,
        )
        self.assertEqual(
            GroqProcessor(_config(), Mode.GENERATE, client=FakeClient()).system_prompt,
            GENERATE_SYSTEM_PROMPT,
        )

    def test_model_is_per_mode_and_configurable(self):
        config = _config(refine_model="fast-model", generate_model="big-model")
        self.assertEqual(
            GroqProcessor(config, Mode.REFINE, client=FakeClient()).model_name,
            "fast-model",
        )
        self.assertEqual(
            GroqProcessor(config, Mode.GENERATE, client=FakeClient()).model_name,
            "big-model",
        )

    def test_token_budget_is_per_mode(self):
        refine = GroqProcessor(_config(), Mode.REFINE, client=FakeClient())
        generate = GroqProcessor(_config(), Mode.GENERATE, client=FakeClient())
        self.assertLess(refine.max_new_tokens, generate.max_new_tokens)


# --------------------------------------------------------------------------- #
# Successful processing
# --------------------------------------------------------------------------- #


class TestGroqProcessing(EnvMixin):
    def test_returns_the_model_text(self):
        client = FakeClient(content="Can you make this more professional?")
        processor = GroqProcessor(_config(), Mode.REFINE, client=client)
        self.assertEqual(
            processor.process("uh can you make this more professional", Mode.REFINE),
            "Can you make this more professional?",
        )

    def test_sends_the_system_prompt_and_the_text(self):
        client = FakeClient()
        processor = GroqProcessor(_config(), Mode.REFINE, client=client)
        processor.process("some words", Mode.REFINE)

        sent = client.completions.calls[0]["messages"]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[1]["content"], "some words")

    def test_sends_the_configured_model(self):
        client = FakeClient()
        config = _config(generate_model="my-model")
        GroqProcessor(config, Mode.GENERATE, client=client).process("hi", Mode.GENERATE)
        self.assertEqual(client.completions.calls[0]["model"], "my-model")

    def test_respects_the_token_budget(self):
        client = FakeClient()
        processor = GroqProcessor(_config(), Mode.GENERATE, client=client)
        processor.process("hi", Mode.GENERATE)
        sent = client.completions.calls[0]
        self.assertEqual(
            sent.get("max_completion_tokens"), processor.max_new_tokens
        )

    def test_empty_input_makes_no_request(self):
        client = FakeClient()
        processor = GroqProcessor(_config(), Mode.GENERATE, client=client)
        self.assertEqual(processor.process("   ", Mode.GENERATE), "")
        self.assertEqual(client.completions.calls, [])

    def test_reasoning_tags_are_stripped(self):
        client = FakeClient(content="<think>hmm</think>Real answer")
        processor = GroqProcessor(_config(), Mode.REFINE, client=client)
        self.assertEqual(processor.process("hi", Mode.REFINE), "Real answer")


# --------------------------------------------------------------------------- #
# Missing API key
# --------------------------------------------------------------------------- #


class TestMissingApiKey(EnvMixin):
    def test_missing_key_raises_a_clear_error(self):
        processor = GroqProcessor(_config(), Mode.GENERATE)
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("write an email", Mode.GENERATE)
        message = str(ctx.exception)
        self.assertIn(GROQ_API_KEY_ENV_VAR, message)
        self.assertIn("gsk_", message)  # tells the user the shape of the value

    def test_missing_key_names_the_mode(self):
        processor = GroqProcessor(_config(), Mode.REFINE)
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hello", Mode.REFINE)
        self.assertIn("REFINE", str(ctx.exception))

    def test_missing_key_offers_the_local_way_out(self):
        processor = GroqProcessor(_config(), Mode.GENERATE)
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertIn("local_qwen", str(ctx.exception))

    def test_missing_key_does_not_silently_use_another_provider(self):
        """A configured cloud mode must not quietly become a local one."""
        from ..refinement.local_qwen import LocalQwenProcessor

        config = _config()
        processor = create_processor(config, Mode.GENERATE)
        self.assertNotIsInstance(processor, LocalQwenProcessor)
        with self.assertRaises(ProcessingError):
            processor.process("hi", Mode.GENERATE)

    def test_warmup_also_fails_on_a_missing_key(self):
        processor = GroqProcessor(_config(), Mode.GENERATE)
        with self.assertRaises(ProcessingError):
            processor.warmup()

    def test_key_from_environment_is_used(self):
        os.environ[GROQ_API_KEY_ENV_VAR] = "gsk_from_env"
        processor = GroqProcessor(_config(), Mode.GENERATE)
        self.assertTrue(processor._has_key)

    def test_fallback_to_plain_groq_api_key(self):
        os.environ["GROQ_API_KEY"] = "gsk_fallback"
        processor = GroqProcessor(_config(), Mode.GENERATE)
        self.assertTrue(processor._has_key)


# --------------------------------------------------------------------------- #
# Error handling — no fallback, actionable messages
# --------------------------------------------------------------------------- #


class TestGroqErrors(EnvMixin):
    def _failing(self, exc_name, message="boom"):
        client = FakeClient(error=_groq_exc(exc_name)(message))
        return GroqProcessor(_config(), Mode.GENERATE, client=client), exc_name

    def test_authentication_error_is_actionable(self):
        processor, _ = self._failing("AuthenticationError")
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertIn("API key", str(ctx.exception))

    def test_connection_error_is_actionable(self):
        processor, _ = self._failing("APIConnectionError")
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        message = str(ctx.exception)
        self.assertIn("network", message.lower())

    def test_timeout_is_reported_as_network(self):
        processor, _ = self._failing("APITimeoutError")
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertIn("timed out", str(ctx.exception))

    def test_offline_failure_says_it_will_not_switch_providers(self):
        processor, _ = self._failing("APIConnectionError")
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertIn("not switching", str(ctx.exception))

    def test_rate_limit_is_reported(self):
        processor, _ = self._failing("RateLimitError")
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertIn("rate limit", str(ctx.exception))

    def test_unknown_model_is_reported_with_the_config_key(self):
        config = _config(generate_model="no-such-model")
        client = FakeClient(error=_groq_exc("NotFoundError")("nope"))
        processor = GroqProcessor(config, Mode.GENERATE, client=client)
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        message = str(ctx.exception)
        self.assertIn("no-such-model", message)
        self.assertIn("generate_model", message)

    def test_unexpected_error_still_becomes_processing_error(self):
        client = FakeClient(error=ValueError("something odd"))
        processor = GroqProcessor(_config(), Mode.GENERATE, client=client)
        with self.assertRaises(ProcessingError):
            processor.process("hi", Mode.GENERATE)

    def test_bad_response_shape_is_reported(self):
        client = FakeClient()
        client.completions.create = lambda **kw: object()  # no .choices
        processor = GroqProcessor(_config(), Mode.GENERATE, client=client)
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertIn("response shape", str(ctx.exception))

    def test_pipeline_contains_a_groq_failure(self):
        """A dead API must not kill the hotkey listener."""
        from .fakes import FakeRouter, RecordingProcessor, make_pipeline
        from ..refinement.router import ModeProcessorRouter

        client = FakeClient(error=_groq_exc("APIConnectionError")("offline"))
        groq = GroqProcessor(_config(), Mode.GENERATE, client=client)
        local = RecordingProcessor(Mode.REFINE, default="local refinement")

        pipeline, _, _, _, output = make_pipeline(
            recognizer="write an email to my professor",
            processor=ModeProcessorRouter({Mode.REFINE: local, Mode.GENERATE: groq}),
        )
        pipeline.handle_press(Mode.GENERATE)
        self.assertIsNone(pipeline.handle_release())
        self.assertEqual(output.delivered, [])

        # REFINE, which is local, keeps working immediately afterwards.
        pipeline.handle_press(Mode.REFINE)
        self.assertEqual(pipeline.handle_release(), "local refinement")


# --------------------------------------------------------------------------- #
# The API key must never leak
# --------------------------------------------------------------------------- #


class TestApiKeyIsNeverLeaked(EnvMixin):
    SECRET = "gsk_supersecretvalue_1234567890"

    def test_key_is_not_in_the_name(self):
        processor = GroqProcessor(_config(), Mode.GENERATE, api_key=self.SECRET)
        self.assertNotIn(self.SECRET, processor.name)

    def test_key_is_not_in_repr(self):
        processor = GroqProcessor(_config(), Mode.GENERATE, api_key=self.SECRET)
        self.assertNotIn(self.SECRET, repr(processor))

    def test_key_is_not_in_the_config_repr(self):
        os.environ[GROQ_API_KEY_ENV_VAR] = self.SECRET
        config = AppConfig()
        self.assertNotIn(self.SECRET, repr(config))
        self.assertNotIn(self.SECRET, config.summary())

    def test_key_is_not_in_error_messages(self):
        client = FakeClient(
            error=_groq_exc("InternalServerError")(f"boom with {self.SECRET}")
        )
        processor = GroqProcessor(
            _config(), Mode.GENERATE, api_key=self.SECRET, client=client
        )
        with self.assertRaises(ProcessingError) as ctx:
            processor.process("hi", Mode.GENERATE)
        self.assertNotIn(self.SECRET, str(ctx.exception))
        self.assertIn("redacted", str(ctx.exception))

    def test_key_is_not_logged_during_processing_or_warmup(self):
        client = FakeClient()
        processor = GroqProcessor(
            _config(), Mode.GENERATE, api_key=self.SECRET, client=client
        )
        messages: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = Capture()
        logging.getLogger("micdrop").addHandler(handler)
        previous = logging.getLogger("micdrop").level
        logging.getLogger("micdrop").setLevel(logging.DEBUG)
        try:
            # A freshly built processor (no injected client) logs on warmup.
            GroqProcessor(
                _config(), Mode.GENERATE, api_key=self.SECRET
            ).warmup()
            processor.process("hi", Mode.GENERATE)
        finally:
            logging.getLogger("micdrop").removeHandler(handler)
            logging.getLogger("micdrop").setLevel(previous)

        self.assertTrue(messages, "expected some log output")
        self.assertNotIn(self.SECRET, "\n".join(messages))

    def test_redact_helper_catches_truncated_keys(self):
        self.assertNotIn("gsk_abcdefgh", _redact("token gsk_abcdefgh rest", None))
        self.assertNotIn(self.SECRET, _redact(f"x{self.SECRET}y", self.SECRET))

    def test_key_is_never_written_to_the_config_file(self):
        from ..app.config import default_config_example

        self.assertNotIn("gsk_", default_config_example().replace("gsk_...", ""))


# --------------------------------------------------------------------------- #
# Mixed local/cloud configuration
# --------------------------------------------------------------------------- #


class TestMixedProviders(EnvMixin):
    def test_local_refine_plus_cloud_generate(self):
        config = AppConfig.from_dict(
            {
                "models": {
                    "refine_provider": "local_qwen",
                    "generate_provider": "groq",
                }
            }
        )
        router = create_router(config)
        self.assertEqual(
            type(router.processor_for(Mode.REFINE)).__name__, "LocalQwenProcessor"
        )
        self.assertEqual(
            type(router.processor_for(Mode.GENERATE)).__name__, "GroqProcessor"
        )

    def test_cloud_refine_plus_local_generate(self):
        config = AppConfig.from_dict(
            {
                "models": {
                    "refine_provider": "groq",
                    "generate_provider": "local_qwen",
                }
            }
        )
        router = create_router(config)
        self.assertEqual(
            type(router.processor_for(Mode.REFINE)).__name__, "GroqProcessor"
        )
        self.assertEqual(
            type(router.processor_for(Mode.GENERATE)).__name__, "LocalQwenProcessor"
        )

    def test_all_local_is_the_offline_default(self):
        config = AppConfig()
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")


if __name__ == "__main__":
    unittest.main()