"""Tests for Linux hotkey configuration plumbing.

These test the platform-independent logic inside the Linux module (key-name
resolution, binding conflicts, device auto-detection). They are skipped on
machines without evdev.
"""

import unittest

try:
    import evdev  # noqa: F401

    HAVE_EVDEV = True
except ImportError:
    HAVE_EVDEV = False

from ..app.config import HotkeySettings
from ..app.modes import Mode

if HAVE_EVDEV:
    from ..platform.linux.keyboard import (
        LinuxHotkeyManager,
        find_keyboard_device,
        resolve_key_code,
    )


@unittest.skipUnless(HAVE_EVDEV, "evdev not available")
class TestKeyResolution(unittest.TestCase):
    def test_known_key_names(self):
        self.assertEqual(resolve_key_code("KEY_RIGHTCTRL"), 97)
        self.assertEqual(resolve_key_code("KEY_RIGHTALT"), 100)
        self.assertEqual(resolve_key_code("KEY_F5"), 63)

    def test_unknown_key_name_raises(self):
        with self.assertRaises(ValueError):
            resolve_key_code("KEY_NOT_A_REAL_KEY")


@unittest.skipUnless(HAVE_EVDEV, "evdev not available")
class TestBindings(unittest.TestCase):
    def _manager(self, **overrides):
        settings = HotkeySettings(refine="KEY_RIGHTCTRL", generate="KEY_RIGHTALT")
        for key, value in overrides.items():
            setattr(settings, key, value)
        return LinuxHotkeyManager(settings)

    def test_register_both_modes(self):
        manager = self._manager()
        manager.register(Mode.REFINE, lambda m: None, lambda m: None)
        manager.register(Mode.GENERATE, lambda m: None, lambda m: None)
        self.assertEqual(len(manager._bindings), 2)

    def test_conflicting_hotkeys_rejected(self):
        manager = self._manager(refine="KEY_F1", generate="KEY_F1")
        manager.register(Mode.REFINE, lambda m: None, lambda m: None)
        with self.assertRaises(ValueError):
            manager.register(Mode.GENERATE, lambda m: None, lambda m: None)


@unittest.skipUnless(HAVE_EVDEV, "evdev not available")
class TestDeviceDetection(unittest.TestCase):
    def test_find_keyboard_device_returns_safe_result(self):
        keys = [resolve_key_code("KEY_RIGHTCTRL"), resolve_key_code("KEY_RIGHTALT")]
        result = find_keyboard_device(keys)
        # On a normal Fedora desktop a matching keyboard exists; on a
        # headless box None is acceptable. It must never raise.
        self.assertTrue(result is None or isinstance(result, str))


if __name__ == "__main__":
    unittest.main()
