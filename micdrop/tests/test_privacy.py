"""Privacy and debug logging.

Two guarantees, both testable without touching a microphone:

1. Raw microphone audio is never logged.
2. Transcriptions are not persisted anywhere — not in the log, not on disk —
   unless debug mode is explicitly enabled.

And one usability guarantee: with debug off, the logs still say enough to tell
"the microphone heard nothing" from "the model produced nothing".
"""

import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..app.config import AppConfig, DebugSettings
from ..app.modes import Mode
from ..refinement.prompts import GENERATE_SYSTEM_PROMPT, REFINE_SYSTEM_PROMPT
from .fakes import FakeRecorder, FakeRecognizer, RecordingProcessor, make_pipeline

AUDIO = np.full((1600, 1), 0.25, dtype="float32")
SECRET_TRANSCRIPT = "my bank code is four three five two one"


def _pipeline(tmp: Path, debug: bool):
    config = AppConfig()
    config = replace(config, debug=DebugSettings(enabled=debug, debug_dir=str(tmp)))
    processor = RecordingProcessor(Mode.REFINE, default="cleaned text")
    pipeline, _, _, _, output = make_pipeline(
        recognizer=SECRET_TRANSCRIPT,
        processor=processor,
        config=config,
        debug_dir=str(tmp),
    )
    return pipeline, output


class TestNothingIsPersistedByDefault(unittest.TestCase):
    def test_no_files_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=False)
            pipeline.handle_press(Mode.REFINE)
            pipeline.handle_release()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_transcript_text_is_not_in_the_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=False)
            with self.assertLogs("micdrop", level="INFO") as captured:
                pipeline.handle_press(Mode.REFINE)
                pipeline.handle_release()
        self.assertNotIn(SECRET_TRANSCRIPT, "\n".join(captured.output))

    def test_result_text_is_not_in_the_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=False)
            with self.assertLogs("micdrop", level="INFO") as captured:
                pipeline.handle_press(Mode.REFINE)
                pipeline.handle_release()
        self.assertNotIn("cleaned text", "\n".join(captured.output))

    def test_logs_still_report_mode_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=False)
            with self.assertLogs("micdrop", level="INFO") as captured:
                pipeline.handle_press(Mode.REFINE)
                pipeline.handle_release()
        joined = "\n".join(captured.output)
        self.assertIn("REFINE", joined)
        self.assertIn("chars", joined)
        self.assertIn("hidden", joined)

    def test_generate_mode_is_also_hidden_by_default(self):
        from .fakes import FakeRouter

        with tempfile.TemporaryDirectory() as tmp:
            config = replace(AppConfig(), debug=DebugSettings(False, str(tmp)))
            pipeline, _, _, _, _ = make_pipeline(
                recognizer=SECRET_TRANSCRIPT,
                processor=FakeRouter(),
                config=config,
                debug_dir=str(tmp),
            )
            with self.assertLogs("micdrop", level="INFO") as captured:
                pipeline.handle_press(Mode.GENERATE)
                pipeline.handle_release()
        joined = "\n".join(captured.output)
        self.assertNotIn(SECRET_TRANSCRIPT, joined)
        self.assertIn("GENERATE", joined)


class TestDebugModePersistsWhenEnabled(unittest.TestCase):
    def test_transcript_text_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=True)
            with self.assertLogs("micdrop", level="INFO") as captured:
                pipeline.handle_press(Mode.REFINE)
                pipeline.handle_release()
        self.assertIn(SECRET_TRANSCRIPT, "\n".join(captured.output))

    def test_transcript_text_is_saved_to_the_debug_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=True)
            pipeline.handle_press(Mode.REFINE)
            pipeline.handle_release()
            contents = [p.read_text() for p in Path(tmp).glob("*.txt")]
            self.assertTrue(contents, "expected a debug transcript file")
            self.assertTrue(
                any(SECRET_TRANSCRIPT in text for text in contents),
                "the transcript should be saved when debug is on",
            )

    def test_recording_is_saved_to_the_debug_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=True)
            pipeline.handle_press(Mode.REFINE)
            pipeline.handle_release()
            self.assertTrue(list(Path(tmp).glob("*.wav")), "expected a debug wav")

    def test_debug_files_are_named_per_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=True)
            pipeline.handle_press(Mode.REFINE)
            pipeline.handle_release()
            names = [p.name for p in Path(tmp).iterdir()]
        self.assertTrue(any("refine" in name for name in names), names)


class TestAudioIsNeverLogged(unittest.TestCase):
    def _records(self, debug: bool) -> list[str]:
        messages: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = Capture()
        logger = logging.getLogger("micdrop")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pipeline, _ = _pipeline(Path(tmp), debug=debug)
                pipeline.handle_press(Mode.REFINE)
                pipeline.handle_release()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        return messages

    def test_no_audio_samples_in_logs_even_when_debugging(self):
        for debug in (False, True):
            for message in self._records(debug):
                self.assertNotIn("array(", message, message)
                self.assertNotIn("0.25", message, message)

    def test_audio_is_in_memory_only_without_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=False)
            # The recorder hands back a numpy array; nothing is written.
            self.assertIsInstance(pipeline.recorder.stop(), np.ndarray)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class TestPromptsNeverReachTheLog(unittest.TestCase):
    def test_system_prompts_are_not_logged_during_a_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _ = _pipeline(Path(tmp), debug=True)
            with self.assertLogs("micdrop", level="DEBUG") as captured:
                pipeline.handle_press(Mode.REFINE)
                pipeline.handle_release()
        joined = "\n".join(captured.output)
        self.assertNotIn(REFINE_SYSTEM_PROMPT[:60], joined)
        self.assertNotIn(GENERATE_SYSTEM_PROMPT[:60], joined)


class TestDebugEnvOverride(unittest.TestCase):
    def test_env_var_enables_debug(self):
        import os

        from ..app.config import DEBUG_ENV_VAR, load_config

        previous = os.environ.get(DEBUG_ENV_VAR)
        os.environ[DEBUG_ENV_VAR] = "1"
        try:
            self.assertTrue(load_config("/nonexistent").debug.enabled)
        finally:
            if previous is None:
                os.environ.pop(DEBUG_ENV_VAR, None)
            else:
                os.environ[DEBUG_ENV_VAR] = previous


if __name__ == "__main__":
    unittest.main()
