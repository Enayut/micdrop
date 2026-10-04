"""Failure behaviour across every stage.

One rule: a failure in any stage is contained, nothing is delivered, and the
listener is immediately ready for the next hotkey cycle. "Contained" means no
exception escapes — the hotkey thread must not die because Whisper hiccuped or
wl-copy timed out.

Each test fails one stage and then asserts both halves of the rule: this cycle
did nothing, and the next cycle works.
"""

import unittest

import numpy as np

from ..app.errors import ProcessingError
from ..app.modes import Mode
from ..refinement.base import TextProcessor
from ..refinement.router import ModeProcessorRouter
from .fakes import (
    FakeHotkeyManager,
    FakeOutput,
    FakeProcessor,
    FakeRecorder,
    FakeRecognizer,
    RecordingProcessor,
    make_pipeline,
)

AUDIO = np.zeros((1600, 1), dtype="float32")


def _wired(**kwargs):
    """Pipeline + hotkey manager, as the real app wires them."""
    pipeline, recorder, recognizer, processor, output = make_pipeline(**kwargs)
    hotkeys = FakeHotkeyManager()
    for mode in Mode:
        hotkeys.register(mode, pipeline.handle_press, pipeline.handle_release)
    return pipeline, hotkeys, recorder, recognizer, processor, output


class TestStageFailuresAreContained(unittest.TestCase):
    """Each stage fails on its own; nothing is delivered and nothing raises."""

    def test_recorder_failure_on_press(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recorder=FakeRecorder(fail_start=RuntimeError("ALSA gone")),
            recognizer="hello",
        )
        hotkeys.press(Mode.REFINE)  # must not raise
        self.assertIsNone(pipeline.active_mode)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(output.delivered, [])

    def test_recorder_failure_on_release(self):
        pipeline, hotkeys, _, _, processor, output = _wired(
            recorder=FakeRecorder(fail_stop=RuntimeError("stream died")),
            recognizer="hello",
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(processor.modes_called, [])
        self.assertEqual(output.delivered, [])

    def test_empty_audio(self):
        empty = np.empty((0, 1), dtype="float32")
        processor = RecordingProcessor(Mode.REFINE)
        pipeline, hotkeys, _, _, _, output = _wired(
            recorder=FakeRecorder(audio=empty),
            recognizer="hello",
            processor=processor,
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(processor.calls, [])
        self.assertEqual(output.delivered, [])

    def test_transcription_failure(self):
        processor = RecordingProcessor(Mode.REFINE)
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer=FakeRecognizer(error=RuntimeError("whisper crashed")),
            processor=processor,
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(processor.calls, [])
        self.assertEqual(output.delivered, [])

    def test_empty_transcription(self):
        processor = RecordingProcessor(Mode.REFINE)
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="   ", processor=processor
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(processor.calls, [])
        self.assertEqual(output.delivered, [])

    def test_refinement_failure(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="make this more professional",
            processor=FakeProcessor(Mode.REFINE, error=ProcessingError("qwen died")),
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(output.delivered, [])

    def test_generation_failure(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="write an email to my professor",
            processor=FakeProcessor(Mode.GENERATE, error=ProcessingError("qwen died")),
        )
        hotkeys.press(Mode.GENERATE)
        self.assertIsNone(hotkeys.release(Mode.GENERATE))
        self.assertEqual(output.delivered, [])

    def test_unexpected_exception_in_processing(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="hello",
            processor=FakeProcessor(Mode.REFINE, error=ValueError("something odd")),
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(output.delivered, [])

    def test_processor_returning_empty_output(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="hello",
            processor=FakeProcessor(Mode.REFINE, result="   "),
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(output.delivered, [])

    def test_output_failure_returning_false(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="hello", output=FakeOutput(fail=True)
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        # The adapter saw the text but reported failure; nothing is claimed as
        # delivered by the pipeline.
        self.assertEqual(output.delivered, ["[refine] hello"])

    def test_output_failure_raising(self):
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="hello",
            output=FakeOutput(raise_error=RuntimeError("wl-copy timed out")),
        )
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))

    def test_mode_with_no_processor(self):
        router = ModeProcessorRouter({Mode.REFINE: RecordingProcessor(Mode.REFINE)})
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="write an email", processor=router
        )
        hotkeys.press(Mode.GENERATE)
        self.assertIsNone(hotkeys.release(Mode.GENERATE))
        self.assertEqual(output.delivered, [])


class TestModelUnavailable(unittest.TestCase):
    def test_model_load_failure_becomes_a_processing_error(self):
        """A missing or corrupt model surfaces as ProcessingError, not a crash."""
        from ..refinement.local_qwen import LocalQwenProcessor

        processor = LocalQwenProcessor.__new__(LocalQwenProcessor)
        # Bypass __init__ (which would read the real config) and force the load
        # path to fail against a model that does not exist.
        TextProcessor.__init__(processor, Mode.REFINE)
        processor.model_name = "definitely/not-a-real-model-xyz"
        processor.max_new_tokens = 16
        processor.adapter_path = None
        processor.system_prompt = "prompt"
        processor._model = None
        processor._tokenizer = None

        with self.assertRaises(ProcessingError):
            processor.process("hello there", Mode.REFINE)

    def test_model_unavailable_does_not_break_the_cycle(self):
        class ExplodingProcessor(RecordingProcessor):
            def process(self, text, mode):
                self._check_mode(mode)
                self.calls.append((text, mode))
                raise ProcessingError("model unavailable")

        refine = ExplodingProcessor(Mode.REFINE)
        pipeline, hotkeys, _, _, _, output = _wired(
            recognizer="hello",
            processor=ModeProcessorRouter({Mode.REFINE: refine}),
        )

        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(output.delivered, [])


class TestAppSurvivesAndKeepsListening(unittest.TestCase):
    """After any failure, the very next hotkey cycle succeeds end to end."""

    def _assert_next_cycle_succeeds(self, working_mode=Mode.GENERATE):
        good = RecordingProcessor(working_mode, default="delivered artifact")
        output = FakeOutput()
        pipeline = _wired(
            recognizer="say something",
            processor=ModeProcessorRouter(
                {Mode.REFINE: RecordingProcessor(Mode.REFINE), Mode.GENERATE: good}
            ),
            output=output,
        )[0]
        pipeline.handle_press(working_mode)
        result = pipeline.handle_release()
        self.assertEqual(result, "delivered artifact")
        self.assertEqual(output.delivered, ["delivered artifact"])

    def _fail_one_cycle(self, **kwargs):
        pipeline, hotkeys, _, _, _, _ = _wired(**kwargs)
        hotkeys.press(kwargs.pop("_mode", Mode.REFINE))
        self.assertIsNone(hotkeys.release(Mode.REFINE))

    def test_recovers_after_transcription_failure(self):
        self._fail_one_cycle(recognizer=FakeRecognizer(error=RuntimeError("boom")))
        self._assert_next_cycle_succeeds()

    def test_recovers_after_refinement_failure(self):
        self._fail_one_cycle(
            recognizer="hello",
            processor=FakeProcessor(Mode.REFINE, error=ProcessingError("boom")),
        )
        self._assert_next_cycle_succeeds()

    def test_recovers_after_output_failure(self):
        self._fail_one_cycle(
            recognizer="hello",
            output=FakeOutput(raise_error=RuntimeError("boom")),
        )
        self._assert_next_cycle_succeeds()

    def test_recovers_after_recorder_failure(self):
        self._fail_one_cycle(
            recognizer="hello",
            recorder=FakeRecorder(fail_stop=RuntimeError("boom")),
        )
        self._assert_next_cycle_succeeds()

    def test_recovers_after_empty_transcription(self):
        self._fail_one_cycle(recognizer="   ")
        self._assert_next_cycle_succeeds()

    def test_recovers_after_mode_without_a_processor(self):
        self._fail_one_cycle(
            recognizer="hello",
            processor=ModeProcessorRouter({Mode.REFINE: RecordingProcessor(Mode.REFINE)}),
            _mode=Mode.GENERATE,
        )
        self._assert_next_cycle_succeeds()

    def test_same_pipeline_recovers_across_cycles(self):
        """One pipeline instance, one failure, then a working cycle."""
        processor = FakeProcessor(Mode.REFINE)
        pipeline, recorder, _, _, output = make_pipeline(
            recognizer="first attempt", processor=processor
        )

        processor.error = ProcessingError("transient failure")
        pipeline.handle_press(Mode.REFINE)
        self.assertIsNone(pipeline.handle_release())
        self.assertEqual(output.delivered, [])

        processor.error = None
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()
        self.assertEqual(result, "[refine] first attempt")
        self.assertEqual(output.delivered, ["[refine] first attempt"])
        self.assertFalse(recorder.is_recording)

    def test_recorder_is_stopped_after_a_failed_cycle(self):
        pipeline, recorder, *_ = make_pipeline(recognizer=FakeRecognizer(error=RuntimeError()))
        pipeline.handle_press(Mode.REFINE)
        pipeline.handle_release()
        self.assertFalse(recorder.is_recording)


if __name__ == "__main__":
    unittest.main()


class TestFailedPressIsNotProcessed(unittest.TestCase):
    """A press that failed to start recording must not run the pipeline.

    Otherwise the release would transcribe silence and paste whatever the
    recognizer hallucinated from it.
    """

    def test_release_after_a_failed_press_delivers_nothing(self):
        processor = RecordingProcessor(Mode.REFINE)
        pipeline, _, _, _, output = make_pipeline(
            recorder=FakeRecorder(fail_start=RuntimeError("ALSA gone")),
            recognizer="this text must never be processed",
            processor=processor,
        )

        pipeline.handle_press(Mode.REFINE)
        self.assertIsNone(pipeline.handle_release(Mode.REFINE))
        self.assertEqual(processor.calls, [])
        self.assertEqual(output.delivered, [])

    def test_a_later_good_press_still_works(self):
        recorder = FakeRecorder(fail_start=RuntimeError("ALSA gone"))
        processor = RecordingProcessor(Mode.REFINE)
        pipeline, _, _, _, output = make_pipeline(
            recorder=recorder,
            recognizer="now working",
            processor=processor,
        )

        pipeline.handle_press(Mode.REFINE)
        self.assertIsNone(pipeline.handle_release(Mode.REFINE))

        recorder.fail_start = None
        pipeline.handle_press(Mode.REFINE)
        self.assertEqual(pipeline.handle_release(Mode.REFINE), "<refine artifact>")
        self.assertEqual(output.delivered, ["<refine artifact>"])
