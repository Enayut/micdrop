"""Tests for the explicit REFINE/GENERATE mode model."""

import unittest

from ..app.modes import Mode, parse_mode


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
        with self.assertRaises(ValueError):
            parse_mode("expand")


if __name__ == "__main__":
    unittest.main()
