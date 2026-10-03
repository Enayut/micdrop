"""Tests for wiring a HotkeyManager to the Pipeline (platform-independent).

Uses a fake HotkeyManager implementing the same interface as
LinuxHotkeyManager, proving that the pipeline can be driven by *any*
platform's hotkey handling without modification.
"""

import unittest

from ..app.modes import Mode
from ..platform.base import HotkeyManager
from .fakes import FakeOutput, make_pipeline


class FakeHotkeyManager(HotkeyManager):
    """Minimal stand-in for LinuxHotkeyManager."""

    def __init__(self):
        self.handlers = {}
        self.running = False

    def register(self, mode, on_press, on_release):
        self.handlers[mode] = (on_press, on_release)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    # test helpers ------------------------------------------------------- #
    def press(self, mode):
        return self.handlers[mode][0](mode)

    def release(self, mode):
        return self.handlers[mode][1](mode)


class TestHotkeyWiring(unittest.TestCase):
    def _wired(self, **kwargs):
        pipeline, recorder, recognizer, processor, output = make_pipeline(**kwargs)
        hotkeys = FakeHotkeyManager()
        for mode in Mode:
            hotkeys.register(
                mode,
                on_press=pipeline.handle_press,
                on_release=pipeline.handle_release,
            )
        return pipeline, hotkeys, recorder, processor, output

    def test_refine_hotkey_runs_refine_pipeline(self):
        pipeline, hotkeys, recorder, processor, output = self._wired(
            recognizer_text="hello there the the"
        )
        hotkeys.press(Mode.REFINE)
        self.assertTrue(recorder.is_recording)
        hotkeys.release(Mode.REFINE)

        self.assertEqual(processor.calls[0][1], Mode.REFINE)
        self.assertEqual(len(output.delivered), 1)
        self.assertFalse(recorder.is_recording)

    def test_generate_hotkey_runs_generate_pipeline(self):
        pipeline, hotkeys, recorder, processor, output = self._wired(
            recognizer_text="make me a poem"
        )
        hotkeys.press(Mode.GENERATE)
        hotkeys.release(Mode.GENERATE)

        self.assertEqual(processor.calls[0][1], Mode.GENERATE)
        self.assertEqual(len(output.delivered), 1)

    def test_different_hotkeys_reach_different_modes(self):
        pipeline, hotkeys, _, processor, _ = self._wired()
        hotkeys.press(Mode.REFINE)
        hotkeys.release(Mode.REFINE)
        hotkeys.press(Mode.GENERATE)
        hotkeys.release(Mode.GENERATE)

        self.assertEqual([m for _, m in processor.calls], [Mode.REFINE, Mode.GENERATE])

    def test_failure_in_pipeline_does_not_break_hotkeys(self):
        pipeline, hotkeys, recorder, processor, output = self._wired(
            recognizer_text="hello",
            output=FakeOutput(fail=True),
        )
        # First cycle: output failure -> contained, returns None.
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(len(output.delivered), 1)

        # The listener (hotkey manager) must still dispatch afterwards:
        # the second cycle succeeds end to end.
        hotkeys.press(Mode.GENERATE)
        self.assertTrue(recorder.is_recording)
        result = hotkeys.release(Mode.GENERATE)
        self.assertEqual(result, "[generate] hello")
        self.assertEqual(len(output.delivered), 2)
        self.assertFalse(recorder.is_recording)


if __name__ == "__main__":
    unittest.main()
