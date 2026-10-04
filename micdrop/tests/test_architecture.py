"""Architecture boundaries, enforced mechanically.

Two structural claims that are easy to state and easy to break:

1. The core (app/, audio/, transcription/, refinement/, output/, platform/base.py)
   contains no Linux-specific assumptions. No evdev, no wl-copy, no ydotool, no
   /dev/input, no clipboard. Those live under platform/linux/ and are chosen in
   exactly one place: the CLI wiring.
2. Providers depend only on the TextProcessor interface, so a Windows or macOS
   backend can be added later without touching the pipeline.

These are import-graph tests rather than behavioural ones, which is the only way
to catch a stray `import evdev` that no functional test would ever notice.
"""

import ast
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Modules that must stay platform independent, and so must everything they
#: import (transitively). This is the code that a Windows/macOS port reuses.
CORE_PACKAGES = ("app", "audio", "transcription", "refinement", "output")

#: The diagnostics package is platform-*neutral* but is not core: it reaches
#: into the selected platform through one hook. It may name platform modules but
#: must not import them directly.
DIAGNOSTICS_PACKAGES = ("diagnostics",)

#: Modules allowed to name Linux/evdev/Wayland things.
PLATFORM_PACKAGES = ("platform", "cli")

#: Import names that would tie a module to this platform's tooling.
FORBIDDEN_IMPORTS = {
    "evdev",
    "fcntl",
    "termios",
    "pynput",
    "pyautogui",
}

#: Substrings that betray a Linux assumption even without an import.
FORBIDDEN_TEXT = (
    "wl-copy",
    "wl_copy",
    "ydotool",
    "/dev/input",
    "X11",
    "wayland",
    "Wayland",
)


def _python_files(package: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / package).rglob("*.py"))


def _imports_in(path: Path) -> set[str]:
    """Top-level module names imported by a file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _code_only(path: Path) -> str:
    """Source with comments and docstrings removed, for text checks.

    Documentation is allowed to name wl-copy and ydotool while explaining the
    boundary; code is not.
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    lines = [
        line.split("#")[0] for line in source.splitlines() if not line.strip().startswith("#")
    ]
    code = "\n".join(lines)
    for doc in docstrings:
        code = code.replace(doc, "")
    return code


def _relative_import_targets(path: Path) -> set[str]:
    """First path component of every relative import in a file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            targets.add(node.module.split(".")[0])
    return targets


def _relative_import_paths(path: Path) -> set[str]:
    """Full dotted path of every relative import in a file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            targets.add(node.module)
    return targets


class TestCoreIsPlatformIndependent(unittest.TestCase):
    def test_core_does_not_import_platform_modules(self):
        for package in CORE_PACKAGES:
            for path in _python_files(package):
                targets = _relative_import_targets(path)
                self.assertNotIn(
                    "platform",
                    targets,
                    f"{path.relative_to(PACKAGE_ROOT)} imports from platform/; "
                    "the core must depend only on the ABCs",
                )

    def test_diagnostics_does_not_import_platform_backends_directly(self):
        """Diagnostics may resolve the platform, but never a backend.

        `from ..platform import select_platform` is the sanctioned way in;
        `from ..platform.linux.keyboard import ...` is not.
        """
        for package in DIAGNOSTICS_PACKAGES:
            for path in _python_files(package):
                for imported in _relative_import_paths(path):
                    self.assertFalse(
                        imported.startswith("platform."),
                        f"{path.relative_to(PACKAGE_ROOT)} imports {imported!r}; "
                        "reach the backend through select_platform().diagnostics",
                    )

    def test_diagnostics_reaches_platform_only_via_the_hook(self):
        from ..platform import select_platform

        selection = select_platform()
        self.assertTrue(
            callable(selection.diagnostics),
            "the platform must expose a diagnostics hook",
        )
        # ... and the hook lives inside the platform package.
        self.assertTrue(
            "platform" in (selection.diagnostics.__module__ or ""),
            f"diagnostics hook came from {selection.diagnostics.__module__}",
        )

    def test_core_does_not_import_cli(self):
        for package in CORE_PACKAGES:
            for path in _python_files(package):
                self.assertNotIn(
                    "cli",
                    _relative_import_targets(path),
                    f"{path.relative_to(PACKAGE_ROOT)} imports from cli/",
                )

    def test_no_forbidden_third_party_imports_in_the_core(self):
        for package in CORE_PACKAGES:
            for path in _python_files(package):
                for imported in _imports_in(path):
                    self.assertNotIn(
                        imported,
                        FORBIDDEN_IMPORTS,
                        f"{path.relative_to(PACKAGE_ROOT)} imports {imported}",
                    )

    def test_no_linux_tool_names_in_the_core(self):
        for package in CORE_PACKAGES:
            for path in _python_files(package):
                code = _code_only(path)
                for needle in FORBIDDEN_TEXT:
                    self.assertNotIn(
                        needle,
                        code,
                        f"{path.relative_to(PACKAGE_ROOT)} mentions {needle!r} in code",
                    )

    def test_platform_base_is_the_only_platform_abstraction(self):
        # platform/base.py must stay free of evdev so a Windows implementation
        # can import the same ABC.
        base = PACKAGE_ROOT / "platform" / "base.py"
        for imported in _imports_in(base):
            self.assertNotIn(imported, FORBIDDEN_IMPORTS, "platform/base.py must stay pure")


class TestPlatformImplementationsAreIsolated(unittest.TestCase):
    #: Only these may name the Linux input stack. The CLI is included because
    #: choosing the platform implementation is exactly its job.
    ALLOWED_EVDEV = ("platform/linux/", "cli/")

    def test_evdev_is_only_imported_by_the_linux_backend_and_the_cli(self):
        offenders = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "tests" in path.parts:
                continue
            if "evdev" not in _imports_in(path):
                continue
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            if not relative.startswith(self.ALLOWED_EVDEV):
                offenders.append(relative)
        self.assertEqual(offenders, [], f"evdev imported outside the allowed set: {offenders}")

    #: clipboard.py holds the actual wl-copy/ydotool calls. The CLI may name the
    #: tools when reporting availability. The diagnostics package may name them
    #: in user-facing "fix this" advice; the pipeline may only mention them in
    #: prose, which _code_only() strips.
    ALLOWED_CLIPBOARD = ("platform/linux/", "cli/", "diagnostics/")

    def test_wl_copy_and_ydotool_only_appear_in_allowed_modules(self):
        offenders = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "tests" in path.parts:
                continue
            code = _code_only(path)
            if "wl-copy" not in code and "ydotool" not in code:
                continue
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            allowed = any(
                relative == entry or relative.endswith(entry) or relative.startswith(entry)
                for entry in self.ALLOWED_CLIPBOARD
            )
            if not allowed:
                offenders.append(relative)
        self.assertEqual(offenders, [], f"clipboard tooling referenced in: {offenders}")

    def test_platform_directory_has_a_linux_subpackage(self):
        self.assertTrue((PACKAGE_ROOT / "platform" / "linux").is_dir())
        self.assertTrue((PACKAGE_ROOT / "platform" / "base.py").is_file())

    def test_platform_selection_happens_in_one_place(self):
        """No `if linux:` branching scattered through the app."""
        # platform/__init__.py defines the registry; cli/main.py resolves it
        # once; diagnostics/__init__.py consumes the hook. Nothing else asks.
        allowed_callers = {
            "platform/__init__.py",
            "cli/main.py",
            "diagnostics/__init__.py",
        }
        offenders = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "tests" in path.parts:
                continue
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            if relative in allowed_callers:
                continue
            code = _code_only(path)
            if "select_platform(" in code or "PLATFORM_REGISTRY" in code:
                offenders.append(relative)
        self.assertEqual(
            offenders,
            [],
            f"platform selection should happen only at the boundary: {offenders}",
        )


class TestProviderInterfaceIsEnough(unittest.TestCase):
    """A new provider needs no change outside refinement/."""

    def test_providers_implement_the_abc(self):
        from ..refinement.base import TextProcessor
        from ..refinement.local_qwen import LocalQwenProcessor
        from ..refinement.passthrough import PassthroughProcessor

        for cls in (LocalQwenProcessor, PassthroughProcessor):
            self.assertTrue(issubclass(cls, TextProcessor), cls)

    def test_router_implements_the_abc(self):
        from ..refinement.base import TextProcessor
        from ..refinement.router import ModeProcessorRouter

        self.assertTrue(issubclass(ModeProcessorRouter, TextProcessor))

    def test_pipeline_only_depends_on_the_abcs(self):
        from ..app.pipeline import Pipeline

        source = (PACKAGE_ROOT / "app" / "pipeline.py").read_text()
        for concrete in ("LocalQwen", "Passthrough", "SoundDevice", "Whisper", "Linux"):
            self.assertNotIn(concrete, source, f"pipeline mentions {concrete}")

    def test_abc_surface_is_small_and_stable(self):
        from ..audio.base import AudioRecorder
        from ..output.base import OutputAdapter
        from ..platform.base import HotkeyManager
        from ..refinement.base import TextProcessor
        from ..transcription.base import SpeechRecognizer

        self.assertEqual(
            {m for m in TextProcessor.__abstractmethods__}, {"name", "process"}
        )
        self.assertEqual(
            {m for m in SpeechRecognizer.__abstractmethods__}, {"transcribe"}
        )
        self.assertEqual(
            {m for m in OutputAdapter.__abstractmethods__}, {"name", "deliver"}
        )
        self.assertEqual(
            {m for m in AudioRecorder.__abstractmethods__},
            {"sample_rate", "is_recording", "start", "stop"},
        )
        # stop() has a default no-op implementation, so it is optional.
        self.assertEqual({m for m in HotkeyManager.__abstractmethods__}, {"register", "start"})
        self.assertTrue(hasattr(HotkeyManager, "stop"))


class TestNoWindowsOrMacosImplementationYet(unittest.TestCase):
    """Task 13: the architecture should be ready, not half-built."""

    def test_no_platform_windows_or_macos_packages(self):
        for name in ("windows", "macos", "darwin"):
            self.assertFalse(
                (PACKAGE_ROOT / "platform" / name).exists(),
                f"platform/{name} should not exist yet",
            )

    def test_no_windows_or_macos_specific_imports(self):
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "tests" in path.parts:
                continue
            for imported in _imports_in(path):
                self.assertNotIn(
                    imported,
                    {"win32api", "win32con", "pynput", "AppKit", "Quartz"},
                    f"{path.name} imports {imported}",
                )


if __name__ == "__main__":
    unittest.main()
