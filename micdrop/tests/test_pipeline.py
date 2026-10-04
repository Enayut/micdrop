"""Tests for the core pipeline: mode routing and failure handling."""

import unittest

import numpy as np

from ..app.errors import ProcessingError, RecordingError, TranscriptionError
from ..app.modes import Mode
from .fakes import (
    PROCESSOR_ERROR,
    RECORDER_ERROR,
    TRANSCRIBE_ERROR,
    FakeOutput,
    FakeProcessor,
    FakeRecorder,
    FakeRecognizer,
    make_pipeline,
)


class TestSuccessfulPipeline(unittest.TestCase):
    def test_successful_pipeline_refine(self):
        pipeline, recorder, recognizer, processor, output = make_pipeline(
            recognizer="um hello world the the",
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertEqual(result, "[refine] um hello world the the")
        self.assertEqual(output.delivered, ["[refine] um hello world the the"])
        # Audio reached the recognizer with the recorder's sample rate.
        self.assertEqual(len(recognizer.calls), 1)
        self.assertEqual(recognizer.calls[0][1], 16000)

    def test_successful_pipeline_generate(self):
        pipeline, _, _, processor, output = make_pipeline(
            recognizer="write me a poem about the sea",
        )
        pipeline.handle_press(Mode.GENERATE)
        result = pipeline.handle_release()

        self.assertEqual(result, "[generate] write me a poem about the sea")
        self.assertEqual(len(output.delivered), 1)

    def test_release_returns_none_without_press(self):
        pipeline, *_ = make_pipeline()
        self.assertIsNone(pipeline.handle_release())


class TestModeSelectionHappensBeforeProcessing(unittest.TestCase):
    """The mode is chosen at press time, not inferred from the audio or text."""

    def test_mode_is_fixed_at_press_time(self):
        pipeline, recorder, *_ = make_pipeline()
        pipeline.handle_press(Mode.GENERATE)
        self.assertEqual(pipeline.active_mode, Mode.GENERATE)
        self.assertTrue(recorder.is_recording)

    def test_active_mode_is_cleared_after_release(self):
        pipeline, *_ = make_pipeline()
        pipeline.handle_press(Mode.REFINE)
        pipeline.handle_release()
        self.assertIsNone(pipeline.active_mode)

    def test_recording_is_stopped_after_release(self):
        pipeline, recorder, *_ = make_pipeline()
        pipeline.handle_press(Mode.REFINE)
        pipeline.handle_release()
        self.assertFalse(recorder.is_recording)


class TestModeRouting(unittest.TestCase):
    """The mode is explicit and passed through to the processor."""

    def test_refine_mode_routed_to_processor(self):
        pipeline, _, _, processor, _ = make_pipeline()
        pipeline.handle_press(Mode.REFINE)
        pipeline.handle_release()

        self.assertEqual(len(processor.calls_for(Mode.REFINE)), 1)
        text, mode = processor.calls_for(Mode.REFINE)[0]
        self.assertEqual(mode, Mode.REFINE)
        self.assertEqual(text, "hello world")

    def test_generate_mode_routed_to_processor(self):
        pipeline, _, _, processor, _ = make_pipeline()
        pipeline.handle_press(Mode.GENERATE)
        pipeline.handle_release()

        self.assertEqual(len(processor.calls_for(Mode.GENERATE)), 1)
        _, mode = processor.calls_for(Mode.GENERATE)[0]
        self.assertEqual(mode, Mode.GENERATE)

    def test_mode_is_never_inferred_from_text(self):
        # Even request-like text must keep the mode it was triggered with.
        pipeline, _, _, processor, _ = make_pipeline(
            recognizer="please write a poem",
        )
        pipeline.handle_press(Mode.REFINE)
        pipeline.handle_release()
        self.assertEqual(processor.calls_for(Mode.REFINE)[0][1], Mode.REFINE)

    def test_release_of_the_other_mode_is_ignored(self):
        pipeline, recorder, _, processor, _ = make_pipeline()
        pipeline.handle_press(Mode.REFINE)
        # Right Alt released while Right Ctrl is still held.
        self.assertIsNone(pipeline.handle_release(Mode.GENERATE))
        self.assertTrue(recorder.is_recording)
        self.assertEqual(processor.modes_called, [])


class TestFailureHandling(unittest.TestCase):
    """Every stage failure is contained: no raise, no output, listener alive."""

    def test_recording_start_failure_does_not_raise(self):
        pipeline, recorder, recognizer, processor, output = make_pipeline(
            recorder=FakeRecorder(fail_start=RECORDER_ERROR),
        )
        pipeline.handle_press(Mode.REFINE)  # must not raise
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(len(recognizer.calls), 0)
        self.assertEqual(processor.modes_called, [])
        self.assertEqual(output.delivered, [])

    def test_recording_stop_failure_does_not_raise(self):
        pipeline, _, recognizer, processor, output = make_pipeline(
            recorder=FakeRecorder(fail_stop=RECORDER_ERROR),
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(len(recognizer.calls), 0)
        self.assertEqual(processor.modes_called, [])
        self.assertEqual(output.delivered, [])

    def test_empty_audio_is_skipped(self):
        empty = np.empty((0, 1), dtype="float32")
        pipeline, _, recognizer, processor, output = make_pipeline(
            recorder=FakeRecorder(audio=empty),
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(len(recognizer.calls), 0)
        self.assertEqual(processor.modes_called, [])
        self.assertEqual(output.delivered, [])

    def test_empty_transcription_skips_processing_and_output(self):
        pipeline, _, recognizer, processor, output = make_pipeline(
            recognizer_text="   ",
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(processor.modes_called, [])  # never reached the processor
        self.assertEqual(output.delivered, [])     # and never the output

    def test_transcription_failure_is_contained(self):
        pipeline, _, _, processor, output = make_pipeline(
            recognizer=FakeRecognizer(error=TRANSCRIBE_ERROR),
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(processor.modes_called, [])
        self.assertEqual(output.delivered, [])

    def test_transcription_unexpected_error_is_contained(self):
        pipeline, _, _, processor, output = make_pipeline(
            recognizer=FakeRecognizer(error=RuntimeError("segfault-ish")),
        )
        pipeline.handle_press(Mode.GENERATE)
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(processor.modes_called, [])
        self.assertEqual(output.delivered, [])

    def test_processor_failure_is_contained(self):
        pipeline, _, _, _, output = make_pipeline(
            processor=FakeProcessor(Mode.REFINE, error=PROCESSOR_ERROR),
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertIsNone(result)
        self.assertEqual(output.delivered, [])

    def test_output_false_result_is_reported(self):
        pipeline, *_ = make_pipeline(output=FakeOutput(fail=True))
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertIsNone(result)

    def test_output_exception_is_contained(self):
        pipeline, *_ = make_pipeline(
            output=FakeOutput(raise_error=RuntimeError("clipboard exploded")),
        )
        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()  # must not raise

        self.assertIsNone(result)

    def test_pipeline_recovers_after_failure(self):
        # A failed cycle must not leave the pipeline in a broken state.
        failing = make_pipeline(recognizer=FakeRecognizer(error=TRANSCRIBE_ERROR))[0]
        failing.handle_press(Mode.REFINE)
        self.assertIsNone(failing.handle_release())

        working = make_pipeline(recognizer="second attempt")[0]
        working.handle_press(Mode.GENERATE)
        result = working.handle_release()
        self.assertEqual(result, "[generate] second attempt")


if __name__ == "__main__":
    unittest.main()
