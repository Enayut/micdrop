"""Tests for the Linux hotkey layer.

Skipped when evdev is unavailable, except for the pure key-name normalization
helpers, which need no input device.

These tests check the *bindings*, not the event loop: Right Ctrl must resolve to
REFINE and Right Alt to GENERATE by default, the same key must never be bound to
both modes, and the config spelling must not matter.
"""

import unittest

from ..app.config import HotkeySettings
from ..app.modes import Mode

try:
    import evdev

    HAS_EVDEV = True
except ImportError:  # pragma: no cover
    evdev = None
    HAS_EVDEV = False


class TestKeyNameNormalization(unittest.TestCase):
    """Key names are normalized without needing a device."""

    def test_normalize_adds_the_prefix(self):
        from ..platform.linux.keyboard import normalize_key_name

        self.assertEqual(normalize_key_name("F5"), "KEY_F5")
        self.assertEqual(normalize_key_name("KEY_F5"), "KEY_F5")
        self.assertEqual(normalize_key_name("key_f5"), "KEY_F5")
        self.assertEqual(normalize_key_name("RIGHTCTRL"), "KEY_RIGHTCTRL")
        self.assertEqual(normalize_key_name("right-ctrl").replace("_", ""), "KEYRIGHTCTRL")

    def test_normalize_rejects_empty(self):
        from ..platform.linux.keyboard import normalize_key_name

        with self.assertRaises(ValueError):
            normalize_key_name("")
        with self.assertRaises(ValueError):
            normalize_key_name("   ")

    @unittest.skipUnless(HAS_EVDEV, "evdev not installed")
    def test_every_spelling_of_right_ctrl_resolves_to_the_same_code(self):
        from ..platform.linux.keyboard import resolve_key_code

        for name in ("RIGHTCTRL", "KEY_RIGHTCTRL", "rightctrl", "Right Ctrl", "right-ctrl"):
            self.assertEqual(resolve_key_code(name), 97, name)

    @unittest.skipUnless(HAS_EVDEV, "evdev not installed")
    def test_every_spelling_of_right_alt_resolves_to_the_same_code(self):
        from ..platform.linux.keyboard import resolve_key_code

        for name in ("RIGHTALT", "KEY_RIGHTALT", "right alt"):
            self.assertEqual(resolve_key_code(name), 100, name)

    @unittest.skipUnless(HAS_EVDEV, "evdev not installed")
    def test_unknown_key_is_rejected(self):
        from ..platform.linux.keyboard import resolve_key_code

        with self.assertRaises(ValueError):
            resolve_key_code("KEY_NONSENSE")

    def test_pretty_names_for_display(self):
        from ..platform.linux.keyboard import pretty_key_name

        self.assertEqual(pretty_key_name("KEY_RIGHTCTRL"), "Right Ctrl")
        self.assertEqual(pretty_key_name("RIGHTALT"), "Right Alt")
        self.assertEqual(pretty_key_name("KEY_F5"), "F5")
        self.assertEqual(pretty_key_name("KEY_LEFTSHIFT"), "Left Shift")


class TestDefaultBindings(unittest.TestCase):
    """Right Ctrl = REFINE, Right Alt = GENERATE."""

    def test_default_settings_name_the_right_keys(self):
        settings = HotkeySettings()
        self.assertEqual(settings.key_for(Mode.REFINE), "RIGHTCTRL")
        self.assertEqual(settings.key_for(Mode.GENERATE), "RIGHTALT")

    @unittest.skipUnless(HAS_EVDEV, "evdev not installed")
    def test_default_keys_resolve_to_the_expected_codes(self):
        from ..platform.linux.keyboard import resolve_key_code

        self.assertEqual(resolve_key_code(HotkeySettings().key_for(Mode.REFINE)), 97)
        self.assertEqual(resolve_key_code(HotkeySettings().key_for(Mode.GENERATE)), 100)

    @unittest.skipUnless(HAS_EVDEV, "evdev not installed")
    def test_the_two_modes_resolve_to_different_keys(self):
        from ..platform.linux.keyboard import resolve_key_code

        settings = HotkeySettings()
        self.assertNotEqual(
            resolve_key_code(settings.key_for(Mode.REFINE)),
            resolve_key_code(settings.key_for(Mode.GENERATE)),
        )

    @unittest.skipUnless(HAS_EVDEV, "evdev not installed")
    def test_hotkeys_are_configurable(self):
        from dataclasses import replace

        from ..platform.linux.keyboard import resolve_key_code

        settings = replace(HotkeySettings(), refine="KEY_F9", generate="KEY_F10")
        self.assertEqual(resolve_key_code(settings.key_for(Mode.REFINE)), 67)
        self.assertEqual(resolve_key_code(settings.key_for(Mode.GENERATE)), 68)


@unittest.skipUnless(HAS_EVDEV, "evdev not installed")
class TestRegistration(unittest.TestCase):
    def _manager(self, settings=None):
        from ..platform.linux.keyboard import LinuxHotkeyManager

        return LinuxHotkeyManager(settings or HotkeySettings())

    def test_both_modes_register_with_distinct_codes(self):
        manager = self._manager()
        manager.register(Mode.REFINE, lambda m: None, lambda m: None)
        manager.register(Mode.GENERATE, lambda m: None, lambda m: None)
        codes = sorted(manager._bindings)
        self.assertEqual(len(codes), 2)
        self.assertNotEqual(codes[0], codes[1])

    def test_registration_uses_the_configured_key(self):
        manager = self._manager()
        manager.register(Mode.REFINE, lambda m: None, lambda m: None)
        self.assertEqual(list(manager._bindings), [97])
        self.assertIs(manager._bindings[97][0], Mode.REFINE)

    def test_binding_one_key_to_both_modes_is_rejected(self):
        from dataclasses import replace

        settings = replace(HotkeySettings(), generate="KEY_RIGHTCTRL")
        manager = self._manager(settings)
        manager.register(Mode.REFINE, lambda m: None, lambda m: None)
        with self.assertRaises(ValueError):
            manager.register(Mode.GENERATE, lambda m: None, lambda m: None)

    def test_registering_an_unknown_mode_is_rejected(self):
        from ..app.errors import UnknownModeError

        manager = self._manager()
        with self.assertRaises(UnknownModeError):
            manager.register("refine", lambda m: None, lambda m: None)

    def test_registering_an_unknown_key_is_rejected(self):
        from dataclasses import replace

        manager = self._manager(replace(HotkeySettings(), refine="KEY_NONSENSE"))
        with self.assertRaises(ValueError):
            manager.register(Mode.REFINE, lambda m: None, lambda m: None)


if __name__ == "__main__":
    unittest.main()
