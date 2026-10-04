"""Tests for configuration loading."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from ..app.config import (
    CONFIG_ENV_VAR,
    DEBUG_ENV_VAR,
    AppConfig,
    default_config_example,
    find_config_file,
    load_config,
)
from ..app.errors import UnknownModeError
from ..app.modes import Mode


class TestConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        config = AppConfig()
        # Right Ctrl = REFINE, Right Alt = GENERATE is the shipped default.
        self.assertEqual(config.hotkeys.key_for(Mode.REFINE), "RIGHTCTRL")
        self.assertEqual(config.hotkeys.key_for(Mode.GENERATE), "RIGHTALT")
        self.assertEqual(config.local.refine_model, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.local.generate_model, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertFalse(config.debug.enabled)

    def test_default_max_recording_duration_is_60s(self):
        self.assertEqual(AppConfig().audio.max_duration, 60.0)

    def test_no_device_is_hardcoded(self):
        # Empty means auto-detect; a Linux path in core config would be a
        # platform assumption.
        self.assertEqual(AppConfig().hotkeys.device, "")

    def test_default_providers_are_per_mode(self):
        config = AppConfig()
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_missing_file_falls_back_to_defaults(self):
        config = load_config("/nonexistent/path/micdrop.toml")
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.hotkeys.key_for(Mode.REFINE), "RIGHTCTRL")

    def test_unknown_mode_lookup_raises(self):
        with self.assertRaises(UnknownModeError):
            AppConfig().provider_for("refine")


class TestConfigFileLoading(unittest.TestCase):
    def _load(self, text: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text(text.strip())
            return load_config(str(path))

    def test_loads_models_section_per_mode(self):
        config = self._load(
            """
[models]
refine_provider = "none"
generate_provider = "local_qwen"
refine_max_new_tokens = 64
generate_max_new_tokens = 1024
"""
        )
        self.assertEqual(config.provider_for(Mode.REFINE), "none")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")
        self.assertEqual(config.max_new_tokens_for(Mode.REFINE), 64)
        self.assertEqual(config.max_new_tokens_for(Mode.GENERATE), 1024)

    def test_loads_every_other_section(self):
        config = self._load(
            """
[hotkeys]
refine = "RIGHTCTRL"
generate = "RIGHTALT"
device = "/dev/input/event9"

[refinement]
whisper_model = "openai/whisper-tiny"
model_path = "Qwen/Qwen3-0.6B"
adapter_path = "/tmp/my-adapter"

[audio]
sample_rate = 48000

[debug]
enabled = true
debug_dir = "/tmp/micdrop-debug"
"""
        )
        # Hotkey names may omit the KEY_ prefix.
        self.assertEqual(config.hotkeys.key_for(Mode.REFINE), "RIGHTCTRL")
        self.assertEqual(config.hotkeys.key_for(Mode.GENERATE), "RIGHTALT")
        self.assertEqual(config.hotkeys.device, "/dev/input/event9")
        self.assertEqual(config.refinement.whisper_model, "openai/whisper-tiny")
        self.assertEqual(config.refinement.adapter_path, "/tmp/my-adapter")
        self.assertEqual(config.audio.sample_rate, 48000)
        self.assertTrue(config.debug.enabled)
        self.assertEqual(config.debug.debug_dir, "/tmp/micdrop-debug")

    def test_unknown_keys_are_ignored(self):
        config = self._load(
            """
[hotkeys]
refine = "KEY_F1"
future_option = "whatever"

[unknown_section]
nonsense = 42
"""
        )
        self.assertEqual(config.hotkeys.key_for(Mode.REFINE), "KEY_F1")

    def test_invalid_toml_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text("not [valid toml {{{")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_find_config_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text('[hotkeys]\nrefine = "KEY_F1"\n')
            self.assertEqual(find_config_file(str(path)), path)


class TestExampleConfig(unittest.TestCase):
    """The shipped example must parse and match the requested schema."""

    def setUp(self):
        import tomllib

        self.data = tomllib.loads(default_config_example())

    def test_has_hotkeys_section(self):
        self.assertEqual(self.data["hotkeys"]["refine"], "RIGHTCTRL")
        self.assertEqual(self.data["hotkeys"]["generate"], "RIGHTALT")

    def test_has_models_section_with_both_providers(self):
        self.assertEqual(self.data["models"]["refine_provider"], "local_qwen")
        self.assertEqual(self.data["models"]["generate_provider"], "local_qwen")

    def test_debug_is_disabled_by_default(self):
        self.assertIs(self.data["debug"]["enabled"], False)

    def test_example_round_trips_through_the_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text(default_config_example())
            config = load_config(str(path))
        self.assertEqual(config.provider_for(Mode.REFINE), "local_qwen")
        self.assertEqual(config.provider_for(Mode.GENERATE), "local_qwen")

    def test_example_matches_the_file_on_disk(self):
        disk = Path(__file__).resolve().parents[2] / "micdrop.toml.example"
        if not disk.is_file():
            self.skipTest("micdrop.toml.example not present")
        self.assertEqual(disk.read_text().strip(), default_config_example().strip())


class TestEnvOverrides(unittest.TestCase):
    def setUp(self):
        self._saved = {
            DEBUG_ENV_VAR: os.environ.get(DEBUG_ENV_VAR),
            CONFIG_ENV_VAR: os.environ.get(CONFIG_ENV_VAR),
            "MICRODROP_ADAPTER": os.environ.get("MICRODROP_ADAPTER"),
        }

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_debug_env_override(self):
        os.environ[DEBUG_ENV_VAR] = "1"
        config = load_config("/nonexistent")
        self.assertTrue(config.debug.enabled)

    def test_debug_env_off(self):
        os.environ[DEBUG_ENV_VAR] = "0"
        config = load_config("/nonexistent")
        self.assertFalse(config.debug.enabled)

    def test_adapter_env_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["MICRODROP_ADAPTER"] = tmp
            config = AppConfig()
            self.assertEqual(config.resolved_adapter_path(), tmp)

    def test_adapter_env_nonexistent_ignored(self):
        os.environ["MICRODROP_ADAPTER"] = "/nonexistent/adapter"
        config = AppConfig()
        path = config.resolved_adapter_path()
        self.assertTrue(path is None or Path(path).is_dir())

    def test_groq_key_from_env_not_config(self):
        os.environ["GROQ_API_KEY"] = "sk-test-123"
        try:
            config = AppConfig()
            self.assertEqual(config.groq_api_key(), "sk-test-123")
            # The key must never be part of the persisted config object.
            self.assertNotIn("sk-test-123", repr(config))
        finally:
            del os.environ["GROQ_API_KEY"]

    def test_summary_contains_no_secrets(self):
        config = AppConfig()
        self.assertNotIn("sk-", config.summary())
        self.assertIn("REFINE", config.summary())
        self.assertIn("GENERATE", config.summary())


if __name__ == "__main__":
    unittest.main()


class ConfigEnvVarNames(unittest.TestCase):
    """The documented env var names must be the ones the code actually reads.

    Regression: CONFIG_ENV_VAR was misspelled "MICRODROP_CONFIG" (MICD, not
    MICRO), so a config file pointed at by the documented
    $MICRODROP_CONFIG was silently ignored and MicDrop fell back to defaults
    with no error at all.
    """

    def test_names_are_microdrop_prefixed(self):
        from micdrop.app import config as cfg

        for name in (
            cfg.CONFIG_ENV_VAR,
            cfg.DEBUG_ENV_VAR,
            cfg.ADAPTER_ENV_VAR,
            cfg.GROQ_API_KEY_ENV_VAR,
        ):
            with self.subTest(name=name):
                self.assertTrue(
                    name.startswith("MICRODROP_"),
                    f"{name} does not start with MICRODROP_",
                )

    def test_config_path_env_var_is_honoured(self):
        import os
        import tempfile

        from micdrop.app import config as cfg

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "micdrop.toml")
            with open(path, "w") as fh:
                fh.write('[hotkeys]\nrefine = "F5"\n')
            with unittest.mock.patch.dict(os.environ, {cfg.CONFIG_ENV_VAR: path}):
                self.assertEqual(cfg.find_config_file(), Path(path))
                self.assertEqual(cfg.load_config().hotkeys.refine, "F5")

    def test_debug_env_var_enables_debug(self):
        import os

        from micdrop.app import config as cfg

        with unittest.mock.patch.dict(os.environ, {cfg.DEBUG_ENV_VAR: "1"}):
            self.assertTrue(cfg.load_config().debug.enabled)
        with unittest.mock.patch.dict(os.environ, {cfg.DEBUG_ENV_VAR: ""}):
            self.assertFalse(cfg.load_config().debug.enabled)

    def test_readme_documents_the_real_names(self):
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
        from micdrop.app import config as cfg

        for name in (cfg.CONFIG_ENV_VAR, cfg.DEBUG_ENV_VAR, cfg.GROQ_API_KEY_ENV_VAR):
            with self.subTest(name=name):
                self.assertIn(name, readme)
