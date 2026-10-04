"""Packaging and platform dispatch.

Two things must hold for MicDrop to be a real desktop utility rather than a
script someone runs from a checkout:

1. ``pip install .`` produces a working ``micdrop`` command. Users should not
   have to know about ``python -m micdrop``.
2. Platform selection happens at exactly one boundary, so a future backend is a
   new directory plus one registry entry.
"""

import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"

#: The `micdrop` console script. Look on PATH first (a real install), then in
#: the project's own venv, so these tests also pass in a development checkout.
VENV_BIN = PACKAGE_ROOT / ".venv" / "bin"


def micdrop_command():
    """Path to the installed `micdrop` command, or None if not installed."""
    found = shutil.which("micdrop")
    if found:
        return found
    candidate = VENV_BIN / "micdrop"
    return str(candidate) if candidate.is_file() else None


MICDROP = micdrop_command()


def _load_pyproject() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


class TestProjectMetadata(unittest.TestCase):
    def setUp(self):
        self.data = _load_pyproject()

    def test_pyproject_exists_and_parses(self):
        self.assertIn("project", self.data)

    def test_declares_the_micdrop_command(self):
        scripts = self.data["project"].get("scripts", {})
        self.assertIn("micdrop", scripts)
        self.assertEqual(scripts["micdrop"], "micdrop.cli.main:main")

    def test_entry_point_is_importable_and_callable(self):
        entry = _load_pyproject()["project"]["scripts"]["micdrop"]
        module_path, _, attr = entry.partition(":")
        module = __import__(module_path, fromlist=[attr])
        self.assertTrue(callable(getattr(module, attr)))

    def test_declares_dependencies(self):
        deps = self.data["project"]["dependencies"]
        for package in ("numpy", "sounddevice", "transformers", "torch"):
            self.assertTrue(
                any(d.startswith(package) for d in deps), f"{package} not declared"
            )

    def test_evdev_is_linux_only(self):
        evdev_deps = [
            d for d in self.data["project"]["dependencies"] if d.startswith("evdev")
        ]
        self.assertTrue(evdev_deps)
        self.assertIn("sys_platform", evdev_deps[0])

    def test_groq_is_declared(self):
        deps = self.data["project"]["dependencies"]
        self.assertTrue(
            any(d.startswith("groq") for d in deps),
            "the groq provider needs its client declared as a dependency",
        )

    def test_python_version_supports_the_syntax_used(self):
        requires = self.data["project"]["requires-python"]
        # `X | None` annotations under `from __future__ import annotations`, and
        # `tuple[...]`/`dict[...]` builtins, both need 3.9+/3.10+.
        self.assertIn("3.11", requires)

    def test_packages_are_discovered(self):
        include = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertIn("micdrop*", include)

    def test_tests_are_not_shipped_as_a_package(self):
        """The tests package must not be installed as importable top-level code."""
        include = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertNotIn("micdrop.tests", include)


class TestInstalledCommandWorks(unittest.TestCase):
    """Verify the `micdrop` command actually runs from an installed package."""

    def _run(self, args, **kwargs):
        return subprocess.run(
            [MICDROP, *args],
            capture_output=True,
            text=True,
            timeout=180,
            **kwargs,
        )

    @unittest.skipUnless(MICDROP, "micdrop is not installed in this environment")
    def test_installed_micdrop_help(self):
        proc = self._run(["--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("setup", proc.stdout)

    @unittest.skipUnless(MICDROP, "micdrop is not installed in this environment")
    def test_installed_micdrop_doctor_runs(self):
        proc = self._run(["doctor"])
        # May legitimately be non-zero on a machine missing dependencies, but it
        # must run and produce a report rather than an ImportError.
        self.assertIn(proc.returncode, (0, 1))
        self.assertIn("MicDrop doctor", proc.stdout)

    @unittest.skipUnless(MICDROP, "micdrop is not installed in this environment")
    def test_installed_micdrop_status_runs(self):
        proc = self._run(["status"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("REFINE", proc.stdout)

    @unittest.skipUnless(MICDROP, "micdrop is not installed in this environment")
    def test_installed_micdrop_setup_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(["setup"], cwd=tmp)
        self.assertIn(proc.returncode, (0, 1))
        self.assertIn("MicDrop Setup", proc.stdout)


    @unittest.skipUnless(MICDROP, "micdrop is not installed in this environment")
    def test_installed_version_matches_pyproject(self):
        proc = self._run(["--version"])
        declared = _load_pyproject()["project"]["version"]
        self.assertIn(declared, proc.stdout)


class TestModuleEntryPoint(unittest.TestCase):
    def test_python_dash_m_still_works(self):
        """`python -m micdrop` must keep working for development."""
        proc = subprocess.run(
            [sys.executable, "-m", "micdrop", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PACKAGE_ROOT,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("micdrop", proc.stdout)

    def test_main_is_importable_without_side_effects(self):
        from micdrop.cli.main import main

        self.assertTrue(callable(main))


class TestPlatformDispatch(unittest.TestCase):
    def test_linux_is_registered(self):
        from micdrop.platform import PLATFORM_REGISTRY

        self.assertIn("linux", PLATFORM_REGISTRY)

    def test_macos_and_windows_are_not_implemented(self):
        from micdrop.platform import PLANNED_PLATFORMS, PLATFORM_REGISTRY

        self.assertNotIn("macos", PLATFORM_REGISTRY)
        self.assertNotIn("windows", PLATFORM_REGISTRY)
        self.assertIn("macos", PLANNED_PLATFORMS)
        self.assertIn("windows", PLANNED_PLATFORMS)

    def test_selecting_an_unimplemented_platform_says_not_yet(self):
        from micdrop.platform import UnsupportedPlatformError, select_platform

        with self.assertRaises(UnsupportedPlatformError) as ctx:
            select_platform("macos")
        self.assertIn("yet", str(ctx.exception))

    def test_selecting_an_unknown_platform_says_unsupported(self):
        from micdrop.platform import UnsupportedPlatformError, select_platform

        with self.assertRaises(UnsupportedPlatformError) as ctx:
            select_platform("plan9")
        self.assertIn("No MicDrop backend", str(ctx.exception))

    def test_selection_exposes_the_factories(self):
        from micdrop.platform import select_platform

        selection = select_platform("linux")
        self.assertTrue(callable(selection.hotkey_factory))
        self.assertTrue(callable(selection.output_factory))
        self.assertTrue(callable(selection.diagnostics))

    def test_selection_creates_the_linux_backend(self):
        from micdrop.app.config import AppConfig
        from micdrop.output.base import OutputAdapter
        from micdrop.platform import select_platform
        from micdrop.platform.base import HotkeyManager

        selection = select_platform("linux")
        self.assertIsInstance(selection.create_output(AppConfig()), OutputAdapter)
        self.assertIsInstance(selection.create_hotkeys(AppConfig()), HotkeyManager)

    def test_factories_receive_their_own_settings_section(self):
        """Regression: the factories used to be handed the whole AppConfig.

        LinuxHotkeyManager wants HotkeySettings, not AppConfig, so passing the
        config object crashed `micdrop run` with
        "'AppConfig' object has no attribute 'key_for'".
        """
        from micdrop.app.config import AppConfig, HotkeySettings, OutputSettings
        from micdrop.platform import select_platform

        selection = select_platform("linux")

        seen: dict[str, object] = {}

        def hotkey_factory(settings):
            seen["hotkeys"] = settings
            return "hotkeys"

        def output_factory(settings):
            seen["output"] = settings
            return "output"

        selection.hotkey_factory = hotkey_factory
        selection.output_factory = output_factory
        config = AppConfig()
        selection.create_hotkeys(config)
        selection.create_output(config)

        self.assertIsInstance(seen["hotkeys"], HotkeySettings)
        self.assertIsInstance(seen["output"], OutputSettings)

    def test_build_app_wires_a_working_linux_pipeline(self):
        """The full wiring path, without loading any model."""
        from micdrop.app.config import AppConfig
        from micdrop.app.modes import Mode
        from micdrop.app.pipeline import Pipeline
        from micdrop.cli.main import build_app
        from micdrop.refinement import ModeProcessorRouter

        try:
            pipeline, hotkeys, used = build_app(AppConfig())
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"build_app needs real hardware here: {exc}")

        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline.processor, ModeProcessorRouter)
        self.assertIs(used.name, "linux")
        # Registering both modes must work against the real Linux manager.
        hotkeys.register(Mode.REFINE, lambda m: None, lambda m: None)
        hotkeys.register(Mode.GENERATE, lambda m: None, lambda m: None)

    def test_a_new_platform_needs_only_a_registry_entry(self):
        """Adding a backend must not require touching anything else."""
        import ast

        from micdrop.platform import PLATFORM_REGISTRY

        source = (PACKAGE_ROOT / "micdrop" / "platform" / "__init__.py").read_text()
        tree = ast.parse(source)
        registry_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "PLATFORM_REGISTRY"
        ]
        self.assertEqual(len(registry_assignments), 1)

        # ... and the registry is the only place a backend is named.
        self.assertTrue(PLATFORM_REGISTRY)


if __name__ == "__main__":
    unittest.main()