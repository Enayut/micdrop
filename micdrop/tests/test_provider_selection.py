"""Provider selection: REFINE and GENERATE can be configured independently.

This is the "make model selection configurable" requirement. Two providers, two
slots, and the ability to point them at different backends without touching the
pipeline — including, later, at a cloud provider that does not exist yet.

What is deliberately NOT here: any Groq implementation. The registry refuses the
name with an actionable message, and that refusal is tested.
"""

import unittest

from ..app.config import AppConfig
from ..app.errors import UnknownProviderError
from ..app.modes import Mode
from ..refinement import (
    PLANNED_PROVIDERS,
    PROVIDERS,
    create_processor,
    create_router,
)
from ..refinement.local_qwen import LocalQwenProcessor
from ..refinement.passthrough import PassthroughProcessor
from ..refinement.router import ModeProcessorRouter


class TestProviderResolution(unittest.TestCase):
    def test_defaults_are_local_qwen_for_both_modes(self):
        config = AppConfig()
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_modes_can_use_different_providers(self):
        config = AppConfig.from_dict(
            {"models": {"refine_provider": "none", "generate_provider": "local_qwen"}}
        )
        self.assertEqual(config.provider_for(Mode.REFINE), "none")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_providers_helper_reports_both(self):
        self.assertEqual(
            AppConfig().providers(), {Mode.REFINE: "local_qwen", Mode.GENERATE: "local_qwen"}
        )

    def test_token_budget_is_per_mode(self):
        config = AppConfig()
        # GENERATE must have room to emit a whole artifact; REFINE must not
        # balloon away from the input.
        self.assertGreater(
            config.max_new_tokens_for(Mode.GENERATE),
            config.max_new_tokens_for(Mode.REFINE),
        )

    def test_token_budgets_are_configurable(self):
        config = AppConfig.from_dict(
            {"models": {"refine_max_new_tokens": 64, "generate_max_new_tokens": 900}}
        )
        self.assertEqual(config.max_new_tokens_for(Mode.REFINE), 64)
        self.assertEqual(config.max_new_tokens_for(Mode.GENERATE), 900)


class TestFactoryBuildsModeBoundProcessors(unittest.TestCase):
    def test_create_processor_requires_a_mode(self):
        with self.assertRaises(TypeError):
            create_processor(AppConfig())  # mode is not optional

    def test_none_provider_builds_passthrough(self):
        config = AppConfig.from_dict({"models": {"refine_provider": "none"}})
        processor = create_processor(config, Mode.REFINE)
        self.assertIsInstance(processor, PassthroughProcessor)
        self.assertEqual(processor.mode, Mode.REFINE)

    def test_local_qwen_provider_is_mode_bound(self):
        config = AppConfig.from_dict({"models": {"generate_provider": "local_qwen"}})
        processor = create_processor(config, Mode.GENERATE)
        self.assertIsInstance(processor, LocalQwenProcessor)
        self.assertEqual(processor.mode, Mode.GENERATE)

    def test_processor_receives_the_modes_token_budget(self):
        config = AppConfig.from_dict({"models": {"generate_max_new_tokens": 777}})
        processor = create_processor(config, Mode.GENERATE)
        self.assertEqual(processor.max_new_tokens, 777)

    def test_router_builds_one_processor_per_mode(self):
        config = AppConfig.from_dict(
            {"models": {"refine_provider": "none", "generate_provider": "none"}}
        )
        router = create_router(config)
        self.assertIsInstance(router, ModeProcessorRouter)
        self.assertTrue(router.supports(Mode.REFINE))
        self.assertTrue(router.supports(Mode.GENERATE))
        self.assertIsInstance(router.processor_for(Mode.REFINE), PassthroughProcessor)

    def test_router_name_shows_the_wiring(self):
        config = AppConfig.from_dict({"models": {"refine_provider": "none"}})
        router = create_router(config)
        self.assertIn("refine=passthrough", router.name)
        self.assertIn("generate=", router.name)

    def test_router_can_be_built_for_one_mode_only(self):
        config = AppConfig()
        router = create_router(config, modes=[Mode.REFINE])
        self.assertTrue(router.supports(Mode.REFINE))
        self.assertFalse(router.supports(Mode.GENERATE))


class TestGroqProviderIsWired(unittest.TestCase):
    """Groq is now a real provider, not a placeholder."""

    def test_groq_is_registered(self):
        self.assertIn("groq", PROVIDERS)
        self.assertNotIn("groq", PLANNED_PROVIDERS)

    def test_groq_can_back_either_mode(self):
        config = AppConfig.from_dict(
            {
                "models": {
                    "refine_provider": "groq",
                    "generate_provider": "local_qwen",
                }
            }
        )
        refine = create_processor(config, Mode.REFINE)
        generate = create_processor(config, Mode.GENERATE)
        self.assertEqual(type(refine).__name__, "GroqProcessor")
        self.assertEqual(type(generate).__name__, "LocalQwenProcessor")

    def test_groq_is_still_mode_bound(self):
        from ..app.errors import ModeMismatchError
        from ..refinement.groq import GroqProcessor

        processor = GroqProcessor(AppConfig(), Mode.GENERATE)
        with self.assertRaises(ModeMismatchError):
            processor.process("hello", Mode.REFINE)

    def test_groq_and_local_can_back_opposite_modes(self):
        config = AppConfig.from_dict(
            {"models": {"refine_provider": "local_qwen", "generate_provider": "groq"}}
        )
        router = create_router(config)
        self.assertEqual(router.processor_for(Mode.REFINE).mode, Mode.REFINE)
        self.assertEqual(router.processor_for(Mode.GENERATE).mode, Mode.GENERATE)
        self.assertIn("groq", router.name)
        self.assertIn("Qwen3", router.name)


class TestUnknownProviders(unittest.TestCase):
    def test_unknown_provider_is_reported_clearly(self):
        config = AppConfig.from_dict({"models": {"refine_provider": "openai"}})
        with self.assertRaises(UnknownProviderError) as ctx:
            create_processor(config, Mode.REFINE)
        self.assertIn("openai", str(ctx.exception))
        self.assertIn("local_qwen", str(ctx.exception))

    def test_unknown_provider_message_lists_every_option(self):
        config = AppConfig.from_dict({"models": {"refine_provider": "openai"}})
        with self.assertRaises(UnknownProviderError) as ctx:
            create_processor(config, Mode.REFINE)
        for known in ("local_qwen", "groq", "none"):
            self.assertIn(known, str(ctx.exception))

    def test_a_bad_provider_fails_the_whole_router(self):
        # Building the app must fail loudly at startup rather than come up
        # with one working mode and one silently missing mode.
        config = AppConfig.from_dict({"models": {"generate_provider": "openai"}})
        with self.assertRaises(UnknownProviderError) as ctx:
            create_router(config)
        self.assertIn("generate", str(ctx.exception).lower())
        # ... while the other mode on its own is still fine.
        self.assertEqual(create_processor(config, Mode.REFINE).mode, Mode.REFINE)


class TestLegacyConfigStillWorks(unittest.TestCase):
    """An old config with [refinement] provider keeps working."""

    def test_legacy_provider_applies_to_both_modes(self):
        config = AppConfig.from_dict({"refinement": {"provider": "none"}})
        self.assertEqual(config.provider_for(Mode.REFINE), "none")
        self.assertEqual(config.provider_for(Mode.GENERATE), "none")

    def test_models_section_overrides_the_legacy_key(self):
        config = AppConfig.from_dict(
            {
                "refinement": {"provider": "none"},
                "models": {"refine_provider": "local_qwen"},
            }
        )
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        # generate_provider defaults, and is not forced to the legacy value
        # because [models] was explicitly present.
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_legacy_token_budget_applies_to_both_modes(self):
        config = AppConfig.from_dict({"refinement": {"max_new_tokens": 200}})
        self.assertEqual(config.max_new_tokens_for(Mode.REFINE), 200)
        self.assertEqual(config.max_new_tokens_for(Mode.GENERATE), 200)


class TestProviderInterfaceIsTheOnlyContract(unittest.TestCase):
    """The pipeline depends on TextProcessor, not on any concrete provider."""

    def test_custom_provider_can_be_registered_without_pipeline_changes(self):
        from ..refinement.base import TextProcessor

        class EchoProcessor(TextProcessor):
            @property
            def name(self):
                return "echo"

            def process(self, text, mode):
                self._check_mode(mode)
                return f"{mode.value}:{text}"

        # Simulate adding a provider to the registry.
        PROVIDERS["echo"] = lambda config, mode: EchoProcessor(mode)
        try:
            config = AppConfig.from_dict(
                {"models": {"refine_provider": "echo", "generate_provider": "echo"}}
            )
            router = create_router(config)
            self.assertEqual(router.process("hi", Mode.REFINE), "refine:hi")
            self.assertEqual(router.process("hi", Mode.GENERATE), "generate:hi")
        finally:
            del PROVIDERS["echo"]

    def test_unknown_mode_cannot_select_a_provider(self):
        from ..app.errors import UnknownModeError

        with self.assertRaises(UnknownModeError):
            create_processor(AppConfig(), "refine")


if __name__ == "__main__":
    unittest.main()
