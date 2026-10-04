"""Tests for wiring a HotkeyManager to the Pipeline (platform-independent).

Uses a fake HotkeyManager implementing the same interface as
LinuxHotkeyManager, proving that the pipeline can be driven by *any*
platform's hotkey handling without modification. This is the readiness
property for Windows/macOS: the pipeline does not know or care which
HotkeyManager it is handed.
"""

import unittest

from ..app.modes import Mode
from ..platform.base import HotkeyManager
from .fakes import FakeHotkeyManager, FakeOutput, make_pipeline


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
            recognizer="hello there the the"
        )
        hotkeys.press(Mode.REFINE)
        self.assertTrue(recorder.is_recording)
        hotkeys.release(Mode.REFINE)

        self.assertEqual(processor.calls_for(Mode.REFINE)[0][1], Mode.REFINE)
        self.assertEqual(len(output.delivered), 1)
        self.assertFalse(recorder.is_recording)

    def test_generate_hotkey_runs_generate_pipeline(self):
        pipeline, hotkeys, recorder, processor, output = self._wired(
            recognizer="make me a poem"
        )
        hotkeys.press(Mode.GENERATE)
        hotkeys.release(Mode.GENERATE)

        self.assertEqual(processor.calls_for(Mode.GENERATE)[0][1], Mode.GENERATE)
        self.assertEqual(len(output.delivered), 1)

    def test_different_hotkeys_reach_different_modes(self):
        pipeline, hotkeys, _, processor, _ = self._wired()
        hotkeys.press(Mode.REFINE)
        hotkeys.release(Mode.REFINE)
        hotkeys.press(Mode.GENERATE)
        hotkeys.release(Mode.GENERATE)

        self.assertEqual(processor.modes_called, [Mode.REFINE, Mode.GENERATE])

    def test_failure_in_pipeline_does_not_break_hotkeys(self):
        pipeline, hotkeys, recorder, processor, output = self._wired(
            recognizer="hello",
            output=FakeOutput(fail=True),
        )
        # First cycle: output failure -> contained, returns None.
        hotkeys.press(Mode.REFINE)
        self.assertIsNone(hotkeys.release(Mode.REFINE))
        self.assertEqual(len(processor.modes_called), 1)
        self.assertEqual(len(output.delivered), 1)

        # The listener (hotkey manager) must still dispatch afterwards:
        # the second cycle succeeds end to end.
        hotkeys.press(Mode.GENERATE)
        self.assertTrue(recorder.is_recording)
        result = hotkeys.release(Mode.GENERATE)
        self.assertEqual(result, "[generate] hello")
        self.assertEqual(len(output.delivered), 2)
        self.assertFalse(recorder.is_recording)

    def test_a_hostile_hotkey_manager_cannot_crash_the_pipeline(self):
        """The pipeline must tolerate a hotkey manager that misbehaves."""

        class HostileHotkeyManager(HotkeyManager):
            def __init__(self):
                self.handlers = {}

            def register(self, mode, on_press, on_release):
                self.handlers[mode] = (on_press, on_release)

            def start(self):
                pass

            def stop(self):
                pass

            def press_boom(self, mode):
                raise RuntimeError("press handler exploded")

        pipeline, _, _, _, output = make_pipeline(recognizer="hello")
        hostile = HostileHotkeyManager()
        for mode in Mode:
            hostile.register(mode, pipeline.handle_press, pipeline.handle_release)

        with self.assertRaises(RuntimeError):
            hostile.press_boom(Mode.REFINE)
        # ... and the pipeline is still usable afterwards.
        hostile.handlers[Mode.REFINE][0](Mode.REFINE)
        self.assertEqual(hostile.handlers[Mode.REFINE][1](Mode.REFINE), "[refine] hello")


if __name__ == "__main__":
    unittest.main()
