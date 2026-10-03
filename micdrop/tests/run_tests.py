"""Simple test runner: `python -m micdrop.tests.run_tests` (or unittest discover)."""

import unittest


def load_tests(loader, tests, pattern):
    return loader.discover(start_dir=__path__[0], pattern="test_*.py")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(start_dir=__path__[0], pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
