"""Tests for the explicit REFINE/GENERATE mode model."""

import unittest

from ..app.errors import UnknownModeError
from ..app.modes import (
    DESCRIPTIONS,
    DISPLAY_NAMES,
    HOTKEY_KEYS,
    MAX_NEW_TOKENS_KEYS,
    MODE_CONTRACTS,
    PROVIDER_KEYS,
    Mode,
    parse_mode,
    require_mode,
)


class TestModes(unittest.TestCase):
    def test_exactly_two_modes(self):
        self.assertEqual({m.value for m in Mode}, {"refine", "generate"})

    def test_modes_are_distinct(self):
        self.assertIsNot(Mode.REFINE, Mode.GENERATE)

    def test_parse_mode_valid(self):
        self.assertEqual(parse_mode("refine"), Mode.REFINE)
        self.assertEqual(parse_mode("GENERATE"), Mode.GENERATE)
        self.assertEqual(parse_mode(" Generate "), Mode.GENERATE)

    def test_parse_mode_invalid(self):
        for bad in ("expand", "auto", "", "rewrite"):
            with self.assertRaises(UnknownModeError):
                parse_mode(bad)

    def test_parse_mode_error_lists_the_valid_modes(self):
        with self.assertRaises(UnknownModeError) as ctx:
            parse_mode("expand")
        self.assertIn("refine", str(ctx.exception))
        self.assertIn("generate", str(ctx.exception))

    def test_parse_mode_is_idempotent_on_a_mode(self):
        self.assertIs(parse_mode(Mode.REFINE), Mode.REFINE)

    def test_require_mode_accepts_only_modes(self):
        self.assertIs(require_mode(Mode.GENERATE), Mode.GENERATE)
        for bad in ("generate", None, 0, Mode):
            with self.assertRaises(UnknownModeError):
                require_mode(bad)


class TestModeMetadata(unittest.TestCase):
    def test_every_mode_is_described(self):
        for mode in Mode:
            self.assertIn(mode, DISPLAY_NAMES)
            self.assertIn(mode, DESCRIPTIONS)
            self.assertTrue(DESCRIPTIONS[mode])

    def test_descriptions_state_the_difference(self):
        self.assertIn("meaning", DESCRIPTIONS[Mode.REFINE])
        self.assertIn("instruction", DESCRIPTIONS[Mode.GENERATE])

    def test_every_mode_has_a_contract(self):
        self.assertEqual(set(MODE_CONTRACTS), set(Mode))

    def test_config_keys_are_mode_specific(self):
        # REFINE and GENERATE must not share a config key, or one would
        # overwrite the other.
        self.assertEqual(HOTKEY_KEYS[Mode.REFINE], "refine")
        self.assertEqual(HOTKEY_KEYS[Mode.GENERATE], "generate")
        self.assertEqual(PROVIDER_KEYS[Mode.REFINE], "refine_provider")
        self.assertEqual(PROVIDER_KEYS[Mode.GENERATE], "generate_provider")
        self.assertEqual(MAX_NEW_TOKENS_KEYS[Mode.REFINE], "refine_max_new_tokens")
        self.assertEqual(MAX_NEW_TOKENS_KEYS[Mode.GENERATE], "generate_max_new_tokens")

    def test_no_mapping_collides(self):
        for mapping in (HOTKEY_KEYS, PROVIDER_KEYS, MAX_NEW_TOKENS_KEYS):
            self.assertEqual(len(set(mapping.values())), len(Mode))


if __name__ == "__main__":
    unittest.main()
