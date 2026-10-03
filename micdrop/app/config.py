"""Configuration system.

Configuration is loaded from (first match wins):

  1. an explicit path passed on the command line / to :func:`load_config`
  2. ``$MICDROP_CONFIG``
  3. ``./micdrop.toml`` (working directory)
  4. ``~/.config/micdrop/config.toml``

Missing files fall back to defaults. Unknown keys in the file are ignored.

Secrets (API keys) are NEVER stored in source or config files; they are read
from the environment at runtime (e.g. ``$GROQ_API_KEY``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .modes import Mode

CONFIG_ENV_VAR = "MICDROP_CONFIG"
DEBUG_ENV_VAR = "MICDROP_DEBUG"
ADAPTER_ENV_VAR = "MICDROP_ADAPTER"  # kept for compatibility with old setup

CONFIG_FILENAME = "micdrop.toml"

# Candidate config file locations, in priority order.
CONFIG_SEARCH_PATHS = (
    Path.cwd() / CONFIG_FILENAME,
    Path.home() / ".config" / "micdrop" / CONFIG_FILENAME,
)


@dataclass
class AudioSettings:
    """Microphone capture settings."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 0  # 0 = sounddevice default


@dataclass
class HotkeySettings:
    """Hotkeys and the keyboard device they are read from.

    Values are evdev key names (see linux/input-event-codes.h), so they stay
    portable across keyboards; the platform layer maps them to codes.
    """

    refine: str = "KEY_RIGHTCTRL"
    generate: str = "KEY_RIGHTALT"
    device: str = "/dev/input/event3"  # auto-detected if missing


@dataclass
class RefinementSettings:
    """Speech-to-text and text-processing settings."""

    # Which TextProcessor implementation to use.
    # Known providers: local_qwen, none. (groq planned; see groq section.)
    provider: str = "local_qwen"

    # Speech-to-text.
    whisper_model: str = "openai/whisper-base"
    language: str = "english"

    # Local Qwen3 + LoRA refinement model.
    model_path: str = "Qwen/Qwen3-0.6B"
    adapter_path: str | None = None  # None = env/default-dir auto-detection

    max_new_tokens: int = 128


@dataclass
class GroqSettings:
    """Placeholder for a future Groq/API TextProcessor.

    The API key is only ever read from the environment (api_key_env); it is
    never written to disk or source code.
    """

    enabled: bool = False
    api_key_env: str = "GROQ_API_KEY"
    model: str = "llama-3.1-8b-instant"


@dataclass
class OutputSettings:
    """Clipboard/paste behaviour (Linux implementation for now)."""

    paste_delay: float = 0.15
    tool_timeout: float = 5.0


@dataclass
class DebugSettings:
    """Debug/audio-persistence behaviour.

    Recordings live in memory only. When debug is enabled they are also
    written to debug_dir and kept for inspection; otherwise nothing is ever
    written to disk.
    """

    enabled: bool = False
    debug_dir: str = "debug"


@dataclass
class AppConfig:
    """Top-level MicDrop configuration."""

    audio: AudioSettings = field(default_factory=AudioSettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    groq: GroqSettings = field(default_factory=GroqSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Build a config from parsed TOML data. Unknown keys are ignored."""
        config = cls()
        section_map = {
            "audio": config.audio,
            "hotkeys": config.hotkeys,
            "refinement": config.refinement,
            "groq": config.groq,
            "output": config.output,
            "debug": config.debug,
        }

        for key, value in data.items():
            if key in section_map and isinstance(value, dict):
                _apply_kwargs(section_map[key], value)

        return config

    def resolved_adapter_path(self) -> str | None:
        """Resolve the LoRA adapter path.

        Priority: explicit config value, then $MICDROP_ADAPTER, then the
        repo's adapters/prompt-refiner-lora directory.
        """
        candidates = [
            self.refinement.adapter_path,
            os.environ.get(ADAPTER_ENV_VAR),
        ]
        default = Path(__file__).resolve().parents[2] / "adapters" / "prompt-refiner-lora"
        candidates.append(str(default))

        for candidate in candidates:
            if candidate and Path(candidate).is_dir():
                return candidate
        return None

    def groq_api_key(self) -> str | None:
        """Read the Groq API key from the environment (never from disk)."""
        return os.environ.get(self.groq.api_key_env) or None

    def summary(self) -> str:
        """Human-readable summary for the startup banner (no secrets)."""
        lines = [
            "Configuration:",
            f"  audio:             {self.audio.sample_rate} Hz, {self.audio.channels} ch",
            f"  hotkeys:           REFINE={self.hotkeys.refine}  GENERATE={self.hotkeys.generate}",
            f"  keyboard device:   {self.hotkeys.device}",
            f"  stt model:         {self.refinement.whisper_model} ({self.refinement.language})",
            f"  provider:          {self.refinement.provider}",
            f"  local model:       {self.refinement.model_path}",
            f"  debug recordings:  {self.debug.debug_dir if self.debug.enabled else 'disabled (in-memory only)'}",
        ]
        return "\n".join(lines)


def _apply_kwargs(obj: Any, values: dict[str, Any]) -> None:
    """Set only known dataclass fields from a dict, ignoring the rest."""
    known = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key in known:
            setattr(obj, key, value)


def find_config_file(explicit: str | Path | None = None) -> Path | None:
    """Locate the configuration file to load, if any."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None

    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return None


def load_config(explicit: str | Path | None = None) -> AppConfig:
    """Load configuration; falls back to defaults when nothing is found."""
    config = AppConfig()

    path = find_config_file(explicit)
    if path is None:
        return _apply_env_overrides(config)

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Failed to read config file {path}: {exc}") from exc

    config = AppConfig.from_dict(data)
    config = _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply environment-variable overrides on top of the file config."""
    overrides: dict[str, Any] = {}
    if os.environ.get(DEBUG_ENV_VAR, "").lower() in ("1", "true", "yes"):
        overrides["debug"] = replace(config.debug, enabled=True)
    if overrides:
        config = replace(config, **overrides)
    return config


def default_config_example() -> str:
    """A ready-to-copy example config file."""
    return """\
# MicDrop configuration (TOML). Copy to ./micdrop.toml or
# ~/.config/micdrop/config.toml. All values shown are the defaults.

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"

[hotkeys]
refine = "KEY_RIGHTCTRL"    # hold to record, release to REFINE
generate = "KEY_RIGHTALT"   # hold to record, release to GENERATE
device = "/dev/input/event3"  # auto-detected if missing

[refinement]
provider = "local_qwen"     # local_qwen | none (groq planned)
whisper_model = "openai/whisper-base"
language = "english"
model_path = "Qwen/Qwen3-0.6B"
# adapter_path = "adapters/prompt-refiner-lora"  # default: $MICDROP_ADAPTER or repo adapters/
max_new_tokens = 128

[groq]
enabled = false
api_key_env = "GROQ_API_KEY"  # key read from env, never stored here
model = "llama-3.1-8b-instant"

[output]
paste_delay = 0.15
tool_timeout = 5.0

[debug]
enabled = false             # true: keep recordings in ./debug/ for inspection
debug_dir = "debug"
"""


MODES = tuple(Mode)  # re-export convenience for callers that iterate modes
