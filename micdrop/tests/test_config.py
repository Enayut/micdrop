"""Tests for configuration loading."""

import os
import tempfile
import unittest
from pathlib import Path

from ..app.config import (
    CONFIG_ENV_VAR,
    DEBUG_ENV_VAR,
    AppConfig,
    find_config_file,
    load_config,
)


class TestConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        config = AppConfig()
        self.assertEqual(config.hotkeys.refine, "KEY_RIGHTCTRL")
        self.assertEqual(config.hotkeys.generate, "KEY_RIGHTALT")
        self.assertEqual(config.refinement.provider, "local_qwen")
        self.assertEqual(config.refinement.model_path, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertFalse(config.debug.enabled)

    def test_missing_file_falls_back_to_defaults(self):
        config = load_config("/nonexistent/path/micdrop.toml")
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.hotkeys.refine, "KEY_RIGHTCTRL")


class TestConfigFileLoading(unittest.TestCase):
    def test_loads_toml_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text(
                """
[hotkeys]
refine = "KEY_F1"
generate = "KEY_F2"
device = "/dev/input/event9"

[refinement]
provider = "none"
whisper_model = "openai/whisper-tiny"
model_path = "Qwen/Qwen3-0.6B"
adapter_path = "/tmp/my-adapter"

[audio]
sample_rate = 48000

[debug]
enabled = true
debug_dir = "/tmp/micdrop-debug"
""".strip()
            )
            config = load_config(str(path))

        self.assertEqual(config.hotkeys.refine, "KEY_F1")
        self.assertEqual(config.hotkeys.generate, "KEY_F2")
        self.assertEqual(config.hotkeys.device, "/dev/input/event9")
        self.assertEqual(config.refinement.provider, "none")
        self.assertEqual(config.refinement.whisper_model, "openai/whisper-tiny")
        self.assertEqual(config.refinement.adapter_path, "/tmp/my-adapter")
        self.assertEqual(config.audio.sample_rate, 48000)
        self.assertTrue(config.debug.enabled)
        self.assertEqual(config.debug.debug_dir, "/tmp/micdrop-debug")

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text(
                """
[hotkeys]
refine = "KEY_F1"
future_option = "whatever"

[unknown_section]
nonsense = 42
""".strip()
            )
            config = load_config(str(path))
        self.assertEqual(config.hotkeys.refine, "KEY_F1")

    def test_invalid_toml_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "micdrop.toml"
            path.write_text("not [valid toml {{{")
            with self.assertRaises(ValueError):
                load_config(str(path))

    def test_find_config_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.toml"
            path.write_text("[hotkeys]\nrefine = \"KEY_F1\"\n")
            found = find_config_file(str(path))
            self.assertEqual(found, path)


class TestEnvOverrides(unittest.TestCase):
    def setUp(self):
        self._saved = {
            DEBUG_ENV_VAR: os.environ.get(DEBUG_ENV_VAR),
            CONFIG_ENV_VAR: os.environ.get(CONFIG_ENV_VAR),
            "MICDROP_ADAPTER": os.environ.get("MICDROP_ADAPTER"),
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
            os.environ["MICDROP_ADAPTER"] = tmp
            config = AppConfig()
            self.assertEqual(config.resolved_adapter_path(), tmp)

    def test_adapter_env_nonexistent_ignored(self):
        os.environ["MICDROP_ADAPTER"] = "/nonexistent/adapter"
        config = AppConfig()
        # Falls through to the repo default dir (which may or may not exist).
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
        self.assertNotIn("GROQ", config.summary().split("provider")[0])


if __name__ == "__main__":
    unittest.main()
