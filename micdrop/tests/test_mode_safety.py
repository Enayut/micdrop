"""The mode safety invariant.

    The REFINE processor must never silently fall back to GENERATE.
    GENERATE must never accidentally be handled with the REFINE prompt.

"Never silently" is the operative word: an explicit, loud refusal is the correct
behaviour. These tests pin down every place a cross-mode substitution could be
introduced — the router, the processors, and the pipeline — and require each one
to raise or to do nothing at all.
"""

import unittest

from ..app.errors import (
    ModeMismatchError,
    ModeProcessorUnavailableError,
    UnknownModeError,
)
from ..app.modes import Mode, parse_mode, require_mode
from ..refinement import create_processor, create_router
from ..refinement.passthrough import PassthroughProcessor
from ..refinement.prompts import GENERATE_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT
from ..refinement.router import ModeProcessorRouter
from .fakes import FakeProcessor, RecordingProcessor, make_pipeline


class TestProcessorsRefuseCrossMode(unittest.TestCase):
    def test_refine_processor_refuses_generate(self):
        processor = FakeProcessor(Mode.REFINE)
        with self.assertRaises(ModeMismatchError):
            processor.process("write a poem", Mode.GENERATE)

    def test_generate_processor_refuses_refine(self):
        processor = FakeProcessor(Mode.GENERATE)
        with self.assertRaises(ModeMismatchError):
            processor.process("clean this up", Mode.REFINE)

    def test_refused_call_is_not_recorded(self):
        processor = FakeProcessor(Mode.REFINE)
        with self.assertRaises(ModeMismatchError):
            processor.process("write a poem", Mode.GENERATE)
        self.assertEqual(processor.calls, [])

    def test_passthrough_also_refuses_cross_mode(self):
        processor = PassthroughProcessor(Mode.REFINE)
        with self.assertRaises(ModeMismatchError):
            processor.process("hello", Mode.GENERATE)

    def test_router_dispatch_is_checked_too(self):
        # A processor whose mode attribute is mutated after registration must
        # still be refused by the router.
        processor = RecordingProcessor(Mode.REFINE)
        router = ModeProcessorRouter({Mode.REFINE: processor})
        processor._mode = Mode.GENERATE  # simulate a miswiring bug
        with self.assertRaises(ModeProcessorUnavailableError):
            router.process("hello", Mode.REFINE)


class TestRouterNeverFallsBack(unittest.TestCase):
    def test_missing_mode_raises_instead_of_borrowing(self):
        router = ModeProcessorRouter({Mode.REFINE: RecordingProcessor(Mode.REFINE)})
        with self.assertRaises(ModeProcessorUnavailableError):
            router.process("write an email", Mode.GENERATE)

    def test_missing_mode_is_not_delivered_to_the_other_processor(self):
        refine = RecordingProcessor(Mode.REFINE)
        router = ModeProcessorRouter({Mode.REFINE: refine})
        with self.assertRaises(ModeProcessorUnavailableError):
            router.process("write an email", Mode.GENERATE)
        self.assertEqual(refine.calls, [])

    def test_registering_a_mismatched_processor_is_rejected(self):
        with self.assertRaises(ModeProcessorUnavailableError):
            ModeProcessorRouter({Mode.REFINE: RecordingProcessor(Mode.GENERATE)})

    def test_empty_router_refuses_everything(self):
        router = ModeProcessorRouter({})
        for mode in Mode:
            with self.assertRaises(ModeProcessorUnavailableError):
                router.process("anything", mode)

    def test_router_has_no_default_mode(self):
        router = ModeProcessorRouter({Mode.REFINE: RecordingProcessor(Mode.REFINE)})
        with self.assertRaises(ModeProcessorUnavailableError):
            _ = router.mode


class TestPipelineNeverSubstitutes(unittest.TestCase):
    def test_pipeline_does_not_fall_back_on_missing_mode(self):
        """A GENERATE press with no GENERATE processor delivers nothing."""
        refine = RecordingProcessor(Mode.REFINE)
        pipeline, _, _, _, output = make_pipeline(
            recognizer="write an email", processor=ModeProcessorRouter({Mode.REFINE: refine})
        )

        pipeline.handle_press(Mode.GENERATE)
        result = pipeline.handle_release()  # contained, returns None

        self.assertIsNone(result)
        self.assertEqual(output.delivered, [])
        self.assertEqual(refine.calls, [], "REFINE processor must not be used for GENERATE")

    def test_pipeline_contains_the_mismatch_without_raising(self):
        router = ModeProcessorRouter({Mode.REFINE: RecordingProcessor(Mode.REFINE)})
        pipeline, _, _, _, output = make_pipeline(recognizer="hello", processor=router)

        with self.assertLogs("micdrop", level="ERROR"):
            result = pipeline.process(_audio(), 16000, Mode.GENERATE)

        self.assertIsNone(result)
        self.assertEqual(output.delivered, [])


class TestModeValidation(unittest.TestCase):
    def test_string_modes_are_not_accepted_as_a_mode(self):
        with self.assertRaises(UnknownModeError):
            require_mode("refine")

    def test_pipeline_rejects_a_non_mode(self):
        pipeline, _, _, _, output = make_pipeline(recognizer="hello")
        self.assertIsNone(pipeline.process(_audio(), 16000, "refine"))
        self.assertEqual(output.delivered, [])

    def test_press_with_a_non_mode_does_not_start_recording(self):
        pipeline, recorder, *_ = make_pipeline()
        pipeline.handle_press("refine")
        self.assertFalse(recorder.is_recording)
        self.assertIsNone(pipeline.active_mode)

    def test_release_with_a_non_mode_is_ignored(self):
        pipeline, recorder, *_ = make_pipeline()
        pipeline.handle_press(Mode.REFINE)
        self.assertIsNone(pipeline.handle_release("generate"))
        self.assertTrue(recorder.is_recording)

    def test_unknown_mode_names_are_rejected(self):
        with self.assertRaises(UnknownModeError):
            parse_mode("expand")
        with self.assertRaises(UnknownModeError):
            parse_mode("auto")

    def test_there_is_still_exactly_two_modes(self):
        self.assertEqual({m.value for m in Mode}, {"refine", "generate"})


class TestPromptsAreModeSpecific(unittest.TestCase):
    def test_prompts_are_distinct(self):
        self.assertNotEqual(REFINE_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT)

    def test_prompt_lookup_is_exact(self):
        from ..refinement.prompts import SYSTEM_PROMPTS, system_prompt_for

        self.assertEqual(set(SYSTEM_PROMPTS), {Mode.REFINE, Mode.GENERATE})
        self.assertIs(system_prompt_for(Mode.REFINE), REFINE_SYSTEM_PROMPT)
        self.assertIs(system_prompt_for(Mode.GENERATE), GENERATE_SYSTEM_PROMPT)

    def test_unknown_mode_has_no_prompt(self):
        from ..refinement.prompts import system_prompt_for

        with self.assertRaises(UnknownModeError):
            system_prompt_for("refine")

    def test_refine_prompt_forbids_answering(self):
        self.assertIn("Do not answer", REFINE_SYSTEM_PROMPT)

    def test_generate_prompt_forbids_meta_commentary(self):
        self.assertIn("Do not discuss the transformation", GENERATE_SYSTEM_PROMPT)


class TestConfiguredProvidersCannotCrossModes(unittest.TestCase):
    """Configuration-level wiring is checked too, not just object wiring."""

    def test_router_from_config_uses_each_mode_own_provider(self):
        from ..app.config import AppConfig

        config = AppConfig.from_dict(
            {"models": {"refine_provider": "none", "generate_provider": "local_qwen"}}
        )
        router = create_router(config)
        self.assertIsInstance(router.processor_for(Mode.REFINE), PassthroughProcessor)
        self.assertEqual(router.processor_for(Mode.GENERATE).mode, Mode.GENERATE)

    def test_create_processor_is_bound_to_the_requested_mode(self):
        from ..app.config import AppConfig

        config = AppConfig.from_dict({"models": {"refine_provider": "none"}})
        processor = create_processor(config, Mode.REFINE)
        self.assertEqual(processor.mode, Mode.REFINE)
        with self.assertRaises(ModeMismatchError):
            processor.process("hello", Mode.GENERATE)


def _audio():
    import numpy as np

    return np.zeros((1600, 1), dtype="float32")


if __name__ == "__main__":
    unittest.main()
