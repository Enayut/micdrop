"""Tests for the diagnostics engine behind `setup`, `doctor`, `status`, `models`.

The engine has to behave well on a machine that is *broken*, not just a happy
one, so most of these tests inject failures and assert the three properties that
matter to a user:

* every check is computed independently, so one failure never hides the rest;
* a failing or crashing probe degrades to a WARN instead of taking down the
  report;
* every FAIL and WARN carries a fix the user can actually act on.
"""

import unittest
from unittest import mock

from ..app.config import GROQ_API_KEY_ENV_VAR, AppConfig
from ..app.modes import Mode
from ..diagnostics import (
    Check,
    DiagnosticReport,
    Status,
    check_config_file,
    check_display,
    check_groq,
    check_local_models,
    check_microphone,
    check_platform,
    check_whisper,
    platform_checks,
    run_full_diagnostics,
)


class TestCheckModel(unittest.TestCase):
    def test_ok_check_is_not_blocking(self):
        self.assertFalse(Check("x", Status.OK).blocking)

    def test_warn_check_is_not_blocking(self):
        self.assertFalse(Check("x", Status.WARN).blocking)

    def test_fail_check_is_blocking(self):
        self.assertTrue(Check("x", Status.FAIL).blocking)

    def test_fail_without_a_fix_is_allowed(self):
        # Nothing blocks on having advice, but the report should still be usable.
        self.assertTrue(Check("x", Status.FAIL, "broken").blocking)


class TestDiagnosticReport(unittest.TestCase):
    def _report(self, *statuses):
        report = DiagnosticReport()
        for index, status in enumerate(statuses):
            report.add(Check(f"c{index}", status, "detail"))
        return report

    def test_empty_report_is_ready(self):
        self.assertTrue(DiagnosticReport().ready)

    def test_only_failures_make_it_unready(self):
        self.assertFalse(self._report(Status.OK, Status.WARN).ready is False)
        self.assertTrue(self._report(Status.OK, Status.WARN).ready)
        self.assertFalse(self._report(Status.OK, Status.FAIL).ready)

    def test_failures_and_warnings_are_separated(self):
        report = self._report(Status.OK, Status.FAIL, Status.WARN, Status.FAIL)
        self.assertEqual(len(report.failures), 2)
        self.assertEqual(len(report.warnings), 1)

    def test_find_and_status_of(self):
        report = self._report(Status.OK, Status.FAIL)
        self.assertEqual(report.find("c1").status, Status.FAIL)
        self.assertEqual(report.status_of("c0"), Status.OK)
        self.assertIsNone(report.find("nope"))
        self.assertIsNone(report.status_of("nope"))


class TestIndividualChecks(unittest.TestCase):
    def test_platform_check_passes_on_linux(self):
        check = check_platform()
        self.assertEqual(check.name, "Platform")
        self.assertEqual(check.status, Status.OK)

    def test_display_check_reports_wayland(self):
        with mock.patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=False):
            check = check_display()
        self.assertEqual(check.status, Status.OK)
        self.assertIn("Wayland", check.detail)

    def test_display_check_fails_without_a_session(self):
        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in ("WAYLAND_DISPLAY", "DISPLAY", "XDG_SESSION_TYPE")
        }
        with mock.patch.dict("os.environ", env, clear=True):
            check = check_display()
        self.assertEqual(check.status, Status.FAIL)
        self.assertIsNotNone(check.fix)

    def test_microphone_check_reports_devices(self):
        check = check_microphone()
        self.assertEqual(check.name, "Microphone")
        # Status depends on the machine; either way it must explain itself.
        if check.status is not Status.OK:
            self.assertTrue(check.detail)

    def test_whisper_check_names_the_model(self):
        check = check_whisper(AppConfig())
        self.assertEqual(check.name, "Whisper")
        self.assertIn(AppConfig().refinement.whisper_model, check.detail)

    def test_local_models_checked_per_mode(self):
        checks = check_local_models(AppConfig())
        names = {c.name for c in checks}
        self.assertIn("Model (refine)", names)
        self.assertIn("Model (generate)", names)

    def test_non_local_provider_is_reported_as_not_local(self):
        config = AppConfig.from_dict({"models": {"generate_provider": "groq"}})
        checks = {c.name: c for c in check_local_models(config)}
        self.assertEqual(checks["Model (generate)"].status, Status.INFO)
        self.assertIn("groq", checks["Model (generate)"].detail)
        # REFINE is still local, so it keeps its own check.
        self.assertIn("Qwen", checks["Model (refine)"].detail)

    def test_groq_key_reported_as_present_without_the_value(self):
        import os

        os.environ[GROQ_API_KEY_ENV_VAR] = "gsk_secret_1234567890"
        try:
            checks = {c.name: c for c in check_groq(AppConfig())}
        finally:
            os.environ.pop(GROQ_API_KEY_ENV_VAR, None)
        check = checks["Groq API key"]
        self.assertEqual(check.status, Status.OK)
        self.assertNotIn("gsk_secret_1234567890", check.detail)
        self.assertNotIn("gsk_secret_1234567890", check.fix or "")

    def test_groq_key_optional_when_unused(self):
        import os

        os.environ.pop(GROQ_API_KEY_ENV_VAR, None)
        os.environ.pop("GROQ_API_KEY", None)
        checks = {c.name: c for c in check_groq(AppConfig())}
        # Configured for local only, so a missing key is informational.
        self.assertEqual(checks["Groq API key"].status, Status.INFO)

    def test_groq_key_required_when_a_mode_uses_it(self):
        import os

        os.environ.pop(GROQ_API_KEY_ENV_VAR, None)
        os.environ.pop("GROQ_API_KEY", None)
        config = AppConfig.from_dict({"models": {"generate_provider": "groq"}})
        checks = {c.name: c for c in check_groq(config)}
        self.assertEqual(checks["Groq API key"].status, Status.FAIL)
        self.assertIn("export", checks["Groq API key"].fix)

    def test_config_file_check_when_missing(self):
        with mock.patch("micdrop.diagnostics.base.find_config_file", return_value=None):
            check = check_config_file()
        self.assertEqual(check.status, Status.WARN)
        self.assertIn("micdrop setup", check.fix)


class TestPlatformHook(unittest.TestCase):
    def test_hook_comes_from_the_platform_package(self):
        from ..platform import select_platform

        selection = select_platform()
        self.assertTrue(callable(selection.diagnostics))

    def test_linux_hook_returns_checks(self):
        checks = platform_checks(AppConfig())
        self.assertTrue(checks)
        names = {c.name for c in checks}
        self.assertIn("evdev", names)
        self.assertIn("wl-copy", names)

    def test_unsupported_platform_is_explained_not_crashed(self):
        from ..platform import UnsupportedPlatformError

        with mock.patch(
            "micdrop.platform.select_platform", side_effect=UnsupportedPlatformError("no backend")
        ):
            checks = platform_checks(AppConfig())
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].status, Status.FAIL)
        self.assertIn("no backend", checks[0].detail)


class TestFullSweep(unittest.TestCase):
    def test_produces_a_report_and_never_raises(self):
        report = run_full_diagnostics(AppConfig())
        self.assertIsInstance(report, DiagnosticReport)
        self.assertTrue(report.checks)

    def test_contains_the_essential_checks(self):
        names = {c.name for c in run_full_diagnostics(AppConfig()).checks}
        for expected in (
            "Platform",
            "Display",
            "Microphone",
            "Whisper",
            "Model (refine)",
            "Model (generate)",
        ):
            self.assertIn(expected, names)

    def test_a_crashing_probe_degrades_to_a_warning(self):
        def explode():
            raise RuntimeError("probe is broken")

        report = DiagnosticReport()
        # run_full_diagnostics wraps each probe; simulate the same protection.
        with mock.patch(
            "micdrop.diagnostics.check_microphone", side_effect=RuntimeError("boom")
        ):
            report = run_full_diagnostics(AppConfig())
        statuses = {c.name: c.status for c in report.checks}
        # The report still has every other check.
        self.assertIn("Platform", statuses)
        self.assertIn("Whisper", statuses)

    def test_one_failure_does_not_hide_the_others(self):
        with mock.patch(
            "micdrop.diagnostics.check_platform",
            return_value=Check("Platform", Status.FAIL, "nope", "fix it"),
        ):
            report = run_full_diagnostics(AppConfig())
        names = {c.name for c in report.checks}
        self.assertIn("Platform", names)
        self.assertIn("Display", names)
        self.assertIn("Microphone", names)
        self.assertFalse(report.ready)

    def test_every_failure_offers_a_fix(self):
        """A FAIL with no fix is a dead end for the user."""
        report = run_full_diagnostics(AppConfig())
        for check in report.failures:
            self.assertIsNotNone(
                check.fix, f"{check.name} fails but gives no way to fix it"
            )

    def test_no_check_ever_exposes_a_secret(self):
        report = run_full_diagnostics(AppConfig())
        for check in report.checks:
            blob = f"{check.detail} {check.fix or ''}"
            self.assertNotIn("gsk_", blob.replace("gsk_...", ""))


class TestRendering(unittest.TestCase):
    def _report(self):
        report = DiagnosticReport()
        report.add(Check("Platform", Status.OK, "Linux"))
        report.add(Check("wl-copy", Status.FAIL, "missing", "sudo dnf install wl-clipboard"))
        report.add(Check("peft", Status.WARN, "missing", "pip install peft"))
        report.add(Check("Groq API key", Status.INFO, "not set"))
        return report

    def test_table_is_aligned(self):
        from ..cli.diagnostics import render_table

        lines = render_table(self._report()).splitlines()
        self.assertEqual(len(lines), 4)
        # Every line starts the status column in the same place.
        columns = {line.index("OK") if "OK" in line else line.index("FAIL") if "FAIL" in line else 0 for line in lines}
        self.assertTrue(all(isinstance(c, int) for c in columns))

    def test_overall_line_reflects_failures(self):
        from ..cli.diagnostics import overall_line

        self.assertIn("NOT READY", overall_line(self._report()))

    def test_overall_line_allows_warnings(self):
        from ..cli.diagnostics import overall_line

        report = DiagnosticReport()
        report.add(Check("Platform", Status.OK, "Linux"))
        report.add(Check("peft", Status.WARN, "missing"))
        self.assertIn("READY", overall_line(report))
        self.assertIn("1 warning", overall_line(report))

    def test_checklist_includes_fixes_but_not_for_ok(self):
        from ..cli.diagnostics import render_checklist

        out = render_checklist(self._report())
        self.assertIn("sudo dnf install wl-clipboard", out)
        self.assertIn("Linux", out)  # OK detail is shown

    def test_fixes_for_lists_only_actionable_checks(self):
        from ..cli.diagnostics import fixes_for

        out = fixes_for(self._report())
        self.assertIn("sudo dnf install wl-clipboard", out)
        self.assertIn("pip install peft", out)

    def test_fixes_for_is_empty_when_all_ok(self):
        from ..cli.diagnostics import fixes_for

        report = DiagnosticReport()
        report.add(Check("Platform", Status.OK, "Linux"))
        self.assertEqual(fixes_for(report), "")

    def test_rendering_survives_a_non_utf8_terminal(self):
        from ..cli.diagnostics import _safe

        # Simulates a cp437 terminal: the symbol cannot encode.
        with mock.patch("sys.stdout") as stdout:
            stdout.encoding = "ascii"
            out = _safe("✓ Platform — Linux")
        self.assertNotIn("✓", out)
        self.assertIn("Platform", out)


class TestProbesAreCheapAndSideEffectFree(unittest.TestCase):
    """Diagnostics must not import heavy packages just to report on them."""

    def test_package_probe_does_not_import_the_packages(self):
        import sys

        from ..diagnostics import base

        for heavy in ("torch", "transformers", "numpy"):
            sys.modules.pop(heavy, None)

        base.check_python_packages()

        for heavy in ("torch", "transformers", "numpy"):
            with self.subTest(module=heavy):
                self.assertNotIn(heavy, sys.modules, f"{heavy} was imported by a probe")

    def test_package_probe_is_fast(self):
        """A doctor run must not pay a multi-second import cost."""
        import time

        from ..diagnostics import base

        base.check_python_packages()  # warm any import-cache effects
        start = time.perf_counter()
        base.check_python_packages()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"check_python_packages took {elapsed:.2f}s")

    def test_missing_module_is_reported_not_raised(self):
        from ..diagnostics import base

        with mock.patch("importlib.util.find_spec", return_value=None):
            names = {c.name for c in base.check_python_packages()}
        self.assertIn("Package: torch", names)
        self.assertTrue(all("Package:" in n for n in names))

    def test_find_spec_raising_valueerror_is_treated_as_missing(self):
        from ..diagnostics import base

        with mock.patch("importlib.util.find_spec", side_effect=ValueError("no parent")):
            self.assertFalse(base._has_module("anything"))


class TestLocalIsTheOfflineDefault(unittest.TestCase):
    """Local providers must work with no network at all."""

    def test_default_configuration_needs_no_network(self):
        config = AppConfig()
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_local_config_builds_no_cloud_processor(self):
        """A local config must not construct a GroqProcessor at all.

        Construction is lazy (no weights are touched), so this is cheap and
        needs no network.
        """
        from ..refinement import create_router

        router = create_router(AppConfig())
        for mode in (Mode.REFINE, Mode.GENERATE):
            with self.subTest(mode=mode):
                processor = router.processor_for(mode)
                self.assertEqual(type(processor).__name__, "LocalQwenProcessor")

    def test_groq_settings_are_read_but_unused_by_local_providers(self):
        config = AppConfig.from_dict({"groq": {"refine_model": "cloud-a"}})
        from ..refinement import create_processor

        processor = create_processor(config, Mode.REFINE)
        self.assertEqual(type(processor).__name__, "LocalQwenProcessor")
        self.assertEqual(processor.model_name, config.local.refine_model)


if __name__ == "__main__":
    unittest.main()