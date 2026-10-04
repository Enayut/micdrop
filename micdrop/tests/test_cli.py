"""Tests for the CLI commands.

Every command is invoked through :func:`micdrop.cli.main.main` with its
argument list, exactly as a user would type it, and the exit code and output
are asserted. That keeps the tests honest about argparse wiring and dispatch,
not just about the helper functions.

Commands that touch hardware (audio, evdev, clipboard) are exercised with
mocks so the suite stays runnable anywhere; the real hardware paths are covered
by the Fedora smoke tests.
"""

import io
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from unittest import mock

from ..app.config import GROQ_API_KEY_ENV_VAR, AppConfig
from ..app.modes import Mode
from ..cli import main as cli
from ..diagnostics import Check, DiagnosticReport, Status


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def run_cli(argv):
    """Invoke the CLI, returning (stdout+stderr, exit code)."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code or 0
    return out.getvalue() + err.getvalue(), code


@contextmanager
def in_temp_dir():
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(previous)


def clean_report():
    """A report where everything passes, for predictable output."""
    report = DiagnosticReport()
    for name in ("Platform", "Display", "Microphone", "evdev", "wl-copy"):
        report.add(Check(name, Status.OK, "fine"))
    return report


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #


class TestHelpAndVersion(unittest.TestCase):
    def test_help_lists_the_core_commands(self):
        out, _ = run_cli(["--help"])
        for command in ("setup", "run", "doctor", "status", "config", "test", "models"):
            self.assertIn(command, out)

    def test_help_is_user_facing(self):
        out, _ = run_cli(["--help"])
        self.assertIn("Right Ctrl", out)
        self.assertIn("micdrop setup", out)

    def test_version(self):
        out, code = run_cli(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("micdrop", out)

    def test_bad_command_exits_nonzero(self):
        _, code = run_cli(["definitely-not-a-command"])
        self.assertNotEqual(code, 0)


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


class TestDoctor(unittest.TestCase):
    def test_doctor_prints_a_table_and_says_ready(self):
        with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
            out, code = run_cli(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("Platform", out)
        self.assertIn("Overall: READY", out)

    def test_doctor_is_non_interactive(self):
        """doctor must not prompt for anything."""
        with mock.patch("builtins.input", side_effect=AssertionError("prompted!")):
            with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
                _, code = run_cli(["doctor"])
        self.assertEqual(code, 0)

    def test_doctor_exits_nonzero_on_failure(self):
        report = clean_report()
        report.add(
            Check("wl-copy", Status.FAIL, "not installed", "sudo dnf install wl-clipboard")
        )
        with mock.patch.object(cli, "run_full_diagnostics", return_value=report):
            out, code = run_cli(["doctor"])
        self.assertEqual(code, 1)
        self.assertIn("NOT READY", out)

    def test_doctor_failure_includes_the_fix(self):
        report = clean_report()
        report.add(
            Check("wl-copy", Status.FAIL, "not installed", "sudo dnf install wl-clipboard")
        )
        with mock.patch.object(cli, "run_full_diagnostics", return_value=report):
            out, _ = run_cli(["doctor"])
        self.assertIn("sudo dnf install wl-clipboard", out)

    def test_doctor_reports_each_check_independently(self):
        """One failing check must not hide the others."""
        report = clean_report()
        report.add(Check("evdev", Status.FAIL, "not installed", "pip install evdev"))
        with mock.patch.object(cli, "run_full_diagnostics", return_value=report):
            out, code = run_cli(["doctor"])
        self.assertEqual(code, 1)
        for name in ("Platform", "Microphone", "evdev", "wl-copy"):
            self.assertIn(name, out)

    def test_doctor_tolerates_warnings(self):
        report = clean_report()
        report.add(Check("Group: input", Status.WARN, "not a member", "sudo usermod -aG input $USER"))
        with mock.patch.object(cli, "run_full_diagnostics", return_value=report):
            _, code = run_cli(["doctor"])
        self.assertEqual(code, 0)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


class TestSetup(unittest.TestCase):
    def test_setup_is_non_interactive(self):
        with mock.patch("builtins.input", side_effect=AssertionError("prompted!")):
            with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
                with in_temp_dir():
                    _, code = run_cli(["setup"])
        self.assertEqual(code, 0)

    def test_setup_creates_a_config_file_with_the_new_schema(self):
        import tomllib

        with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
            with in_temp_dir():
                run_cli(["setup"])
                with open("micdrop.toml", "rb") as fh:
                    data = tomllib.load(fh)

        self.assertEqual(data["hotkeys"]["refine"], "RIGHTCTRL")
        self.assertEqual(data["hotkeys"]["generate"], "RIGHTALT")
        self.assertEqual(data["models"]["refine_provider"], "local_qwen")
        self.assertEqual(data["models"]["generate_provider"], "local_qwen")
        self.assertEqual(data["audio"]["max_duration"], 60.0)
        self.assertIs(data["debug"]["enabled"], False)

    def test_setup_does_not_overwrite_an_existing_config(self):
        with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
            with in_temp_dir():
                with open("micdrop.toml", "w") as fh:
                    fh.write('[hotkeys]\nrefine = "KEY_F9"\n')
                run_cli(["setup"])
                with open("micdrop.toml") as fh:
                    self.assertIn("KEY_F9", fh.read())

    def test_setup_reports_success(self):
        with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
            with in_temp_dir():
                out, code = run_cli(["setup"])
        self.assertEqual(code, 0)
        self.assertIn("ready", out.lower())

    def test_setup_explains_a_failure(self):
        report = clean_report()
        report.add(
            Check(
                "ydotoold",
                Status.FAIL,
                "not running",
                "systemctl --user enable --now ydotool.service",
            )
        )
        with mock.patch.object(cli, "run_full_diagnostics", return_value=report):
            with in_temp_dir():
                out, code = run_cli(["setup"])
        self.assertEqual(code, 1)
        self.assertIn("systemctl --user", out)

    def test_setup_never_runs_a_privileged_command(self):
        """MicDrop reports problems; it does not sudo on the user's behalf."""
        with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
            with mock.patch("subprocess.run") as run:
                with in_temp_dir():
                    run_cli(["setup"])
        for call in run.call_args_list:
            args = " ".join(str(a) for a in (call.args[0] if call.args else []))
            self.assertNotIn("sudo", args)

    def test_setup_creates_a_usable_config_that_loads(self):
        from ..app.config import load_config

        with mock.patch.object(cli, "run_full_diagnostics", return_value=clean_report()):
            with in_temp_dir():
                run_cli(["setup"])
                config = load_config()
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.audio.max_duration, 60.0)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


class TestStatus(unittest.TestCase):
    def test_status_shows_both_modes(self):
        out, code = run_cli(["status"])
        self.assertEqual(code, 0)
        self.assertIn("REFINE", out)
        self.assertIn("GENERATE", out)

    def test_status_shows_hotkeys_and_providers(self):
        out, _ = run_cli(["status"])
        self.assertIn("RIGHTCTRL", out)
        self.assertIn("local_qwen", out)

    def test_status_never_prints_an_api_key(self):
        os.environ[GROQ_API_KEY_ENV_VAR] = "gsk_secretvalue_9876543210"
        try:
            out, _ = run_cli(["status"])
        finally:
            os.environ.pop(GROQ_API_KEY_ENV_VAR, None)
        self.assertNotIn("gsk_secretvalue_9876543210", out)
        self.assertIn("set", out)

    def test_status_shows_max_recording_duration(self):
        out, _ = run_cli(["status"])
        self.assertIn("max recording", out.lower())

    def test_status_reflects_a_custom_config(self):
        import tomllib

        with in_temp_dir():
            with open("micdrop.toml", "w") as fh:
                fh.write(
                    '[hotkeys]\nrefine = "KEY_F9"\n'
                    '[models]\ngenerate_provider = "groq"\n'
                )
            out, _ = run_cli(["status"])
        self.assertIn("KEY_F9", out)
        self.assertIn("groq", out)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


class TestConfig(unittest.TestCase):
    def test_config_prints_the_summary(self):
        out, code = run_cli(["config"])
        self.assertEqual(code, 0)
        self.assertIn("REFINE", out)

    def test_config_example_is_valid_toml_with_the_new_schema(self):
        import tomllib

        out, _ = run_cli(["config", "--example"])
        data = tomllib.loads(out)
        self.assertEqual(data["hotkeys"]["refine"], "RIGHTCTRL")
        self.assertEqual(data["hotkeys"]["generate"], "RIGHTALT")
        self.assertEqual(data["models"]["refine_provider"], "local_qwen")
        self.assertEqual(data["models"]["generate_provider"], "local_qwen")
        self.assertEqual(data["models"]["local"]["refine_model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(data["models"]["local"]["generate_model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(data["audio"]["max_duration"], 60.0)
        self.assertIs(data["debug"]["enabled"], False)

    def test_config_example_documents_the_groq_env_var(self):
        out, _ = run_cli(["config", "--example"])
        self.assertIn(GROQ_API_KEY_ENV_VAR, out)

    def test_config_example_never_contains_a_real_key(self):
        out, _ = run_cli(["config", "--example"])
        self.assertNotIn("gsk_", out.replace("gsk_...", ""))


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


class TestModels(unittest.TestCase):
    def test_models_lists_providers(self):
        out, code = run_cli(["models"])
        self.assertEqual(code, 0)
        for provider in ("local_qwen", "groq", "none"):
            self.assertIn(provider, out)

    def test_models_lists_both_modes(self):
        out, _ = run_cli(["models"])
        self.assertIn("REFINE", out)
        self.assertIn("GENERATE", out)

    def test_models_loads_no_model(self):
        """Listing models must be instant and offline."""
        with mock.patch("transformers.AutoModelForCausalLM") as model:
            with mock.patch("transformers.pipeline") as pipeline:
                run_cli(["models"])
        model.assert_not_called()
        pipeline.assert_not_called()

    def test_models_reports_the_configured_cloud_model(self):
        with in_temp_dir():
            with open("micdrop.toml", "w") as fh:
                fh.write(
                    '[models]\nrefine_provider = "local_qwen"\n'
                    'generate_provider = "groq"\n'
                    '[models.groq]\ngenerate_model = "my-cloud-model"\n'
                )
            out, _ = run_cli(["models"])
        self.assertIn("my-cloud-model", out)


# --------------------------------------------------------------------------- #
# test
# --------------------------------------------------------------------------- #


class TestTestCommand(unittest.TestCase):
    def test_test_reports_per_subsystem(self):
        out, code = run_cli(["test"])
        self.assertIn("self-test", out.lower())
        self.assertIn(code, (0, 1))

    def test_test_names_its_subsystems(self):
        out, _ = run_cli(["test"])
        self.assertIn("Clipboard", out)
        self.assertIn("Microphone", out)
        self.assertIn("Hotkeys", out)

    def test_test_is_non_interactive(self):
        with mock.patch("builtins.input", side_effect=AssertionError("prompted!")):
            run_cli(["test"])

    def test_each_probe_returns_a_bool_and_a_note(self):
        """Regression: a probe returning a bare bool used to crash `micdrop test`.

        `record()` unpacks (ok, note); a helper returning only a bool made the
        whole command die with "cannot unpack non-iterable bool object".
        """
        for probe in (cli._clipboard_check, cli._record_check, cli._hotkey_check):
            with self.subTest(probe=probe.__name__):
                result = probe()
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], bool)
                self.assertIsInstance(result[1], str)

    def test_probe_notes_are_meaningful_on_this_machine(self):
        for probe in (cli._clipboard_check, cli._record_check, cli._hotkey_check):
            ok, note = probe()
            self.assertTrue(note, f"{probe.__name__} returned an empty note")


# --------------------------------------------------------------------------- #
# run overrides
# --------------------------------------------------------------------------- #


def _args(**overrides):
    defaults = dict(
        provider=None,
        refine_provider=None,
        generate_provider=None,
        device=None,
        debug_audio=False,
        config=None,
    )
    defaults.update(overrides)
    return mock.Mock(**defaults)


class TestRunStartsWithOneUnavailableMode(unittest.TestCase):
    """`micdrop run` must start even when one mode's provider is broken.

    REFINE=local_qwen with GENERATE=groq and no API key is the realistic case:
    the app should still come up and keep REFINE usable, and say plainly that
    GENERATE will not work.
    """

    def _run_with_failures(self, failures, captured):
        class FakeHotkeys:
            def __init__(self):
                self.started = False

            def register(self, mode, on_press=None, on_release=None):
                pass

            def start(self):
                self.started = True

        # A Mock pipeline satisfies cmd_run's handle_press/handle_release
        # wiring without any model or hardware.
        FakePipeline = mock.Mock()

        def fake_warmup(_pipeline):
            return failures

        with mock.patch.object(cli, "build_app",
                               return_value=(FakePipeline(), FakeHotkeys(), mock.Mock())), \
             mock.patch.object(cli, "load_config", return_value=AppConfig()), \
             mock.patch.object(cli, "apply_run_overrides", side_effect=lambda c, a: c), \
             mock.patch("micdrop.refinement.warmup_application",
                        side_effect=fake_warmup), \
             mock.patch("sys.stderr", new=captured):
            rc = cli.cmd_run(_args())
        return rc

    def test_startup_succeeds_and_lists_the_broken_mode(self):
        import io

        captured = io.StringIO()
        rc = self._run_with_failures(
            {Mode.GENERATE: "no API key is set. Export MICRODROP_GROQ_API_KEY"}, captured
        )
        self.assertEqual(rc, 0)
        message = captured.getvalue()
        self.assertIn("GENERATE is unavailable", message)
        self.assertIn("no API key", message)
        self.assertIn("Still working: REFINE", message)

    def test_a_broken_stt_recognizer_is_still_fatal(self):
        import io

        from ..app.errors import TranscriptionError

        class FakeHotkeys:
            def register(self, mode, on_press=None, on_release=None):
                pass

            def start(self):
                raise AssertionError("must not listen without a recognizer")

        FakePipeline = mock.Mock()

        def boom(_pipeline):
            raise TranscriptionError("whisper model missing")

        captured = io.StringIO()
        with mock.patch.object(cli, "build_app",
                               return_value=(FakePipeline(), FakeHotkeys(), mock.Mock())), \
             mock.patch.object(cli, "load_config", return_value=AppConfig()), \
             mock.patch.object(cli, "apply_run_overrides", side_effect=lambda c, a: c), \
             mock.patch("micdrop.refinement.warmup_application",
                        side_effect=boom), \
             mock.patch("sys.stderr", new=captured):
            rc = cli.cmd_run(_args())
        self.assertEqual(rc, 1)
        self.assertIn("could not prepare MicDrop", captured.getvalue())


class TestRunOverrides(unittest.TestCase):
    """`micdrop run --refine-provider ...` applies for this run only."""

    def test_refine_provider_override(self):
        config = cli.apply_run_overrides(AppConfig(), _args(refine_provider="none"))
        self.assertEqual(config.provider_for(Mode.REFINE), "none")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_generate_provider_override(self):
        config = cli.apply_run_overrides(AppConfig(), _args(generate_provider="groq"))
        self.assertEqual(config.provider_for(Mode.GENERATE), "groq")

    def test_broad_provider_override_applies_to_both(self):
        config = cli.apply_run_overrides(AppConfig(), _args(provider="none"))
        self.assertEqual(config.provider_for(Mode.REFINE), "none")
        self.assertEqual(config.provider_for(Mode.GENERATE), "none")

    def test_specific_override_beats_the_broad_one(self):
        config = cli.apply_run_overrides(
            AppConfig(), _args(provider="none", refine_provider="local_qwen")
        )
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.provider_for(Mode.GENERATE), "none")

    def test_device_and_debug_overrides(self):
        config = cli.apply_run_overrides(
            AppConfig(), _args(device="/dev/input/event9", debug_audio=True)
        )
        self.assertEqual(config.hotkeys.device, "/dev/input/event9")
        self.assertTrue(config.debug.enabled)

    def test_overrides_do_not_mutate_the_config_file(self):
        with in_temp_dir():
            with open("micdrop.toml", "w") as fh:
                fh.write('[models]\nrefine_provider = "local_qwen"\n')
            cli.apply_run_overrides(
                AppConfig(), _args(refine_provider="groq")
            )
            with open("micdrop.toml") as fh:
                self.assertIn("local_qwen", fh.read())


class TestBadProviderAtStartup(unittest.TestCase):
    def test_unknown_provider_fails_at_startup(self):
        with in_temp_dir():
            _, code = run_cli(["run", "--refine-provider=not-a-provider"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()