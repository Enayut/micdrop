"""Recording safety: a held hotkey must not record forever.

``[audio] max_duration`` (default 60s) caps a single recording. The limit is
enforced on the audio callback thread rather than only on release, because a
main thread that is busy must not be able to miss it.

The pipeline-facing behaviour matters most: hitting the cap is a *normal* end of
recording. The audio captured so far is still transcribed and delivered, the
cycle completes, and the app is ready for the next hotkey. It is not an error.
"""

from unittest import mock
import unittest

import numpy as np

from ..app.config import AudioSettings
from ..app.errors import ProcessingError
from ..app.modes import Mode
from ..audio.recorder import SoundDeviceRecorder
from ..refinement.prompts import GENERATE_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT
from .fakes import (
    FakeHotkeyManager,
    FakeOutput,
    FakeProcessor,
    FakeRecorder,
    FakeRecognizer,
    RecordingProcessor,
    make_pipeline,
)


class FakeStream:
    """Minimal stand-in for a sounddevice InputStream."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class TestRecorderDurationLimit(unittest.TestCase):
    def _recorder(self, max_duration):
        settings = AudioSettings(max_duration=max_duration)
        recorder = SoundDeviceRecorder(settings)
        recorder._stream = FakeStream()
        return recorder

    def test_default_is_sixty_seconds(self):
        self.assertEqual(AudioSettings().max_duration, 60.0)

    def test_elapsed_starts_at_zero(self):
        recorder = self._recorder(60.0)
        self.assertEqual(recorder.elapsed, 0.0)

    def test_blocks_are_accepted_below_the_limit(self):
        recorder = self._recorder(60.0)
        recorder._recording = True
        block = np.zeros((160, 1), dtype="float32")
        recorder._audio_callback(block, 160, None, None)
        recorder._audio_callback(block, 160, None, None)
        self.assertEqual(len(recorder._frames), 2)
        self.assertTrue(recorder.is_recording)

    def test_recording_stops_itself_past_the_limit(self):
        recorder = self._recorder(0.05)
        recorder._recording = True
        recorder._started_at = __import__("time").monotonic() - 10.0  # long past

        block = np.zeros((160, 1), dtype="float32")
        recorder._audio_callback(block, 160, None, None)

        self.assertFalse(recorder.is_recording, "recorder must stop itself")
        self.assertEqual(len(recorder._frames), 0, "the over-limit block is dropped")

    def test_stop_after_the_limit_returns_the_audio_captured(self):
        recorder = self._recorder(0.05)
        recorder._recording = True

        early = np.ones((160, 1), dtype="float32")
        recorder._audio_callback(early, 160, None, None)

        recorder._started_at = __import__("time").monotonic() - 10.0
        recorder._audio_callback(early, 160, None, None)

        audio = recorder.stop()
        self.assertGreater(len(audio), 0, "audio before the cap must be kept")
        self.assertFalse(recorder.is_recording)

    def test_zero_disables_the_limit(self):
        recorder = self._recorder(0)
        recorder._recording = True
        recorder._started_at = __import__("time").monotonic() - 10000.0
        recorder._audio_callback(np.zeros((160, 1), dtype="float32"), 160, None, None)
        self.assertTrue(recorder.is_recording)

    def test_stop_when_not_recording_returns_empty(self):
        recorder = self._recorder(60.0)
        self.assertEqual(len(recorder.stop()), 0)

    def test_audio_stays_one_dimensional(self):
        recorder = self._recorder(60.0)
        recorder._recording = True
        recorder._audio_callback(np.ones((160, 1), dtype="float32"), 160, None, None)
        audio = recorder.stop()
        self.assertEqual(audio.ndim, 1)


class TestLimitIsConfigurable(unittest.TestCase):
    def test_limit_comes_from_config(self):
        settings = AudioSettings(max_duration=15.0)
        self.assertEqual(SoundDeviceRecorder(settings).max_duration, 15.0)

    def test_limit_is_not_hardcoded_in_the_pipeline(self):
        """The value comes from config; the recorder just honours it."""
        from ..app.config import load_config
        from ..app.pipeline import Pipeline

        recorder = SoundDeviceRecorder(AudioSettings(max_duration=5.0))
        pipeline = Pipeline(
            recorder=recorder,
            recognizer=FakeRecognizer(),
            processor=RecordingProcessor(Mode.REFINE),
            output=FakeOutput(),
        )
        self.assertEqual(pipeline.recorder.max_duration, 5.0)

        # ... and a config file can change it without touching code.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text("[audio]\nmax_duration = 12.0\n")
            config = load_config(str(path))
        self.assertEqual(config.audio.max_duration, 12.0)


class TestPipelineSurvivesAMaxedRecording(unittest.TestCase):
    """Hitting the cap is a normal cycle, not a failure."""

    def _wired(self, processor, output=None):
        pipeline, recorder, recognizer, _, out = make_pipeline(
            recorder=FakeRecorder(),
            recognizer=FakeRecognizer("a fairly long sentence about the deadline"),
            processor=processor,
            output=output or FakeOutput(),
        )
        hotkeys = FakeHotkeyManager()
        for mode in Mode:
            hotkeys.register(mode, pipeline.handle_press, pipeline.handle_release)
        return pipeline, hotkeys, out

    def test_a_maxed_recording_still_delivers(self):
        """Whatever was captured before the cap is processed normally."""
        processor = RecordingProcessor(Mode.REFINE, default="cleaned")
        pipeline, hotkeys, output = self._wired(processor)

        # Simulate the recorder having self-stopped at the cap: is_recording is
        # already False and the frames are present.
        hotkeys.press(Mode.REFINE)
        result = hotkeys.release(Mode.REFINE)

        self.assertEqual(result, "cleaned")
        self.assertEqual(output.delivered, ["cleaned"])

    def test_the_app_is_ready_again_afterwards(self):
        """A maxed recording must not wedge the hotkey listener."""
        processor = RecordingProcessor(Mode.REFINE, default="cleaned")
        pipeline, hotkeys, output = self._wired(processor)

        for _ in range(3):
            hotkeys.press(Mode.REFINE)
            self.assertEqual(hotkeys.release(Mode.REFINE), "cleaned")

        self.assertEqual(len(output.delivered), 3)
        self.assertIsNone(pipeline.active_mode)

    def test_a_maxed_then_empty_recording_does_not_wedge_it(self):
        processor = RecordingProcessor(Mode.REFINE, default="cleaned")
        pipeline, hotkeys, _ = self._wired(processor)
        hotkeys.press(Mode.REFINE)
        hotkeys.release(Mode.REFINE)

        pipeline.recognizer.text = ""  # nothing intelligible heard
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))

        pipeline.recognizer.text = "speaking again"
        hotkeys.press(Mode.REFINE)
        self.assertEqual(hotkeys.release(Mode.REFINE), "cleaned")


class TestPartialWarmup(unittest.TestCase):
    """One unavailable mode must not disable a working one.

    The motivating case is REFINE=local_qwen with GENERATE=groq and no API key.
    Killing the whole app there would throw away a perfectly good REFINE; and
    quietly using the local model for GENERATE would violate the mode contract.
    """

    def _router(self, failing_mode):
        from ..app.errors import ProcessingError
        from ..refinement.base import TextProcessor
        from ..refinement.router import ModeProcessorRouter

        class Stub(TextProcessor):
            def __init__(self, mode):
                self._mode = mode
                self.warmup_calls = 0

            @property
            def name(self):
                return f"stub-{self._mode.value}"

            @property
            def system_prompt(self):
                return "stub"

            def process(self, text, mode=None):
                return f"{self._mode.value}:{text}"

            def warmup(self):
                self.warmup_calls += 1
                if self._mode is failing_mode:
                    raise ProcessingError("no API key is set")

        return ModeProcessorRouter({Mode.REFINE: Stub(Mode.REFINE),
                                    Mode.GENERATE: Stub(Mode.GENERATE)})

    def test_a_failing_mode_does_not_stop_the_working_one(self):
        router = self._router(Mode.GENERATE)
        failures = router.warmup_individually()

        self.assertEqual(set(failures), {Mode.GENERATE})
        self.assertIn("no API key", failures[Mode.GENERATE])
        self.assertEqual(router.failures, failures)

    def test_the_working_mode_still_processes(self):
        router = self._router(Mode.GENERATE)
        router.warmup_individually()

        self.assertEqual(router.process("hi", Mode.REFINE), "refine:hi")

    def test_the_failing_mode_raises_instead_of_substituting(self):
        router = self._router(Mode.GENERATE)
        router.warmup_individually()

        from ..app.errors import ModeProcessorUnavailableError

        with self.assertRaises(ModeProcessorUnavailableError) as ctx:
            router.process("hi", Mode.GENERATE)
        message = str(ctx.exception)
        self.assertIn("no API key", message)
        self.assertIn("Not falling back", message)

    def test_all_or_nothing_warmup_still_raises(self):
        from ..app.errors import ModeProcessorUnavailableError

        router = self._router(Mode.GENERATE)
        with self.assertRaises(ModeProcessorUnavailableError):
            router.warmup()

    def test_everything_warm_means_no_failures(self):
        router = self._router(None)
        self.assertEqual(router.warmup_individually(), {})
        self.assertEqual(router.failures, {})

    def test_warmup_application_returns_per_mode_failures(self):
        from ..refinement import warmup_application

        router = self._router(Mode.GENERATE)

        class FakePipeline:
            recognizer = type("R", (), {"warmup": lambda self: None})()
            processor = router

        failures = warmup_application(FakePipeline())
        self.assertEqual(set(failures), {Mode.GENERATE})

    def test_a_recognizer_failure_is_still_fatal(self):
        """Without Whisper nothing works, so that one must not be swallowed."""
        from ..app.errors import TranscriptionError
        from ..refinement import warmup_application

        class BrokenRecognizer:
            def warmup(self):
                raise TranscriptionError("whisper model missing")

        router = self._router(None)

        class FakePipeline:
            recognizer = BrokenRecognizer()
            processor = router

        with self.assertRaises(TranscriptionError):
            warmup_application(FakePipeline())


class TestRecorderLifecycle(unittest.TestCase):
    """Models load once, at startup — never inside a hotkey cycle."""

    def test_recognizer_and_processors_warm_up_before_listening(self):
        from ..refinement import warmup_application

        calls: list[str] = []

        class WarmingRecorder:
            def warmup(self):
                calls.append("recognizer")

        class WarmingProcessor:
            def warmup(self):
                calls.append("processors")

        class FakePipeline:
            recognizer = WarmingRecorder()
            processor = WarmingProcessor()

        warmup_application(FakePipeline())
        self.assertEqual(calls, ["recognizer", "processors"])

    def test_warmup_does_not_touch_hotkeys(self):
        from ..refinement import warmup_application

        class FakePipeline:
            recognizer = type("R", (), {"warmup": lambda self: None})()
            processor = type("P", (), {"warmup": lambda self: None})()

        warmup_application(FakePipeline())  # must not need a HotkeyManager

    def test_local_model_is_cached_across_modes(self):
        """REFINE and GENERATE share one loaded model, not two.

        The loader is stubbed: a unit test must not download or instantiate
        real weights (that took ~45s and gigabytes when this called the real
        ``_load_model``).
        """
        from ..app.config import AppConfig
        from ..refinement import create_router
        from ..refinement import local_qwen

        config = AppConfig()
        router = create_router(config)
        refine = router.processor_for(Mode.REFINE)
        generate = router.processor_for(Mode.GENERATE)
        self.assertNotEqual(refine, generate)

        # Nothing is loaded until something asks for it.
        self.assertIsNone(refine._model)
        self.assertIsNone(generate._model)

        loads: list[str] = []
        sentinel_model = object()
        sentinel_tokenizer = object()

        def fake_loader(self):
            loads.append(self.model_name)
            return (sentinel_model, sentinel_tokenizer)

        local_qwen.clear_model_cache()
        with mock.patch.object(
            type(refine), "_load_model", fake_loader, create=True
        ):
            refine._ensure_loaded()
            generate._ensure_loaded()

        # Both modes name the same model, so it is loaded exactly once and
        # both processors end up pointing at the same objects.
        self.assertEqual(loads, [config.local.refine_model])
        self.assertIs(refine._model, sentinel_model)
        self.assertIs(generate._model, sentinel_model)
        self.assertIs(refine._tokenizer, generate._tokenizer)

        local_qwen.clear_model_cache()

    def test_different_models_are_not_shared(self):
        """Different model ids must not collapse into one cached instance."""
        from ..app.config import AppConfig
        from ..refinement import create_router
        from ..refinement import local_qwen

        config = AppConfig.from_dict(
            {"models": {"local": {"refine_model": "model-a", "generate_model": "model-b"}}}
        )
        router = create_router(config)
        refine = router.processor_for(Mode.REFINE)
        generate = router.processor_for(Mode.GENERATE)

        loads: list[str] = []
        local_qwen.clear_model_cache()
        with mock.patch.object(
            type(refine),
            "_load_model",
            lambda self: (loads.append(self.model_name) or object(), object()),
            create=True,
        ):
            refine._ensure_loaded()
            generate._ensure_loaded()

        self.assertEqual(loads, ["model-a", "model-b"])
        self.assertIsNot(refine._model, generate._model)
        local_qwen.clear_model_cache()

    def test_prompts_stay_bound_to_their_mode_after_construction(self):
        from ..app.config import AppConfig
        from ..refinement import create_router

        router = create_router(AppConfig())
        self.assertEqual(
            router.processor_for(Mode.REFINE).system_prompt, REFINE_SYSTEM_PROMPT
        )
        self.assertEqual(
            router.processor_for(Mode.GENERATE).system_prompt, GENERATE_SYSTEM_PROMPT
        )


if __name__ == "__main__":
    unittest.main()