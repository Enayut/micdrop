"""Configuration system.

Configuration is loaded from (first match wins):

  1. an explicit path passed on the command line / to :func:`load_config`
  2. ``$MICRODROP_CONFIG``
  3. ``./micdrop.toml`` (working directory)
  4. ``~/.config/micdrop/micdrop.toml``

Missing files fall back to defaults. Unknown keys in the file are ignored.

Everything the app can be configured with lives here; no other module hardcodes
a sample rate, a hotkey, a model name or a token budget. The one thing that
cannot live in a file is an API key, which is read from the environment at
runtime (``$MICRODROP_GROQ_API_KEY``) and never written to disk.

Provider selection is per mode (``[models] refine_provider`` /
``generate_provider``), so REFINE and GENERATE can run on different backends —
a local model for one, a cloud API for the other — with no change to the
pipeline. See :meth:`AppConfig.provider_for`.

TOML note: ``[models.local]`` and ``[models.groq]`` are nested *tables*, so
``tomllib`` hands them back inside the ``models`` dict. They may also be written
as flat ``[local]`` / ``[groq]`` tables; both spellings are accepted.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .modes import HOTKEY_KEYS, MAX_NEW_TOKENS_KEYS, Mode, PROVIDER_KEYS, require_mode

CONFIG_ENV_VAR = "MICRODROP_CONFIG"
DEBUG_ENV_VAR = "MICRODROP_DEBUG"
ADAPTER_ENV_VAR = "MICRODROP_ADAPTER"  # kept for compatibility with old setup

#: Environment variable holding the Groq API key. Deliberately prefixed so it
#: cannot collide with a bare GROQ_API_KEY set by some other tool.
GROQ_API_KEY_ENV_VAR = "MICRODROP_GROQ_API_KEY"
#: Accepted for convenience when someone has already exported GROQ_API_KEY.
GROQ_API_KEY_FALLBACK_ENV_VAR = "GROQ_API_KEY"

CONFIG_FILENAME = "micdrop.toml"


def config_search_paths() -> tuple[Path, ...]:
    """Candidate config file locations, in priority order.

    Computed on each call rather than at import time: the working directory can
    change during a process (and does inside tests), and a frozen tuple would
    keep pointing at wherever the process happened to start.
    """
    return (
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / ".config" / "micdrop" / CONFIG_FILENAME,
    )


@dataclass
class AudioSettings:
    """Microphone capture settings.

    ``max_duration`` is a safety limit: holding a hotkey forever must not record
    forever. When the limit is reached the recorder stops itself and the pending
    cycle completes normally, so the app can never get stuck in "recording".
    """

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 0  # 0 = sounddevice default

    #: Hard cap on a single recording, in seconds. 0 disables the limit.
    max_duration: float = 60.0


@dataclass
class HotkeySettings:
    """Hotkeys and the keyboard device they are read from.

    Values are evdev key names (see linux/input-event-codes.h), so they stay
    portable across keyboards; the platform layer maps them to codes. Both
    ``"RIGHTCTRL"`` and ``"KEY_RIGHTCTRL"`` are accepted — see
    ``micdrop.platform.linux.keyboard.key_name_candidates``.

    These defaults are the product's shipped defaults: Right Ctrl = REFINE,
    Right Alt = GENERATE. They are only defaults; the whole point is that they
    are configurable.
    """

    refine: str = "RIGHTCTRL"
    generate: str = "RIGHTALT"
    # Empty means "auto-detect", which is also what happens if the configured
    # path cannot be opened. Hard-coding a device path here would put a Linux
    # assumption in the core config, so there is no default.
    device: str = ""

    def key_for(self, mode: Mode) -> str:
        """The configured hotkey name for `mode`."""
        return getattr(self, HOTKEY_KEYS[require_mode(mode)])


@dataclass
class LocalModelSettings:
    """Model identifiers for the ``local_qwen`` provider.

    Kept separate from provider *selection* so that changing which model backs
    REFINE is a one-line config edit and does not interact with the provider
    name.
    """

    refine_model: str = "Qwen/Qwen3-0.6B"
    generate_model: str = "Qwen/Qwen3-0.6B"

    #: Optional LoRA adapter directory. Empty/None means auto-detect.
    adapter_path: str | None = None

    def model_for(self, mode: Mode) -> str:
        """The local model configured for `mode`. Raises for an unknown mode."""
        mode = require_mode(mode)
        return self.refine_model if mode is Mode.REFINE else self.generate_model


@dataclass
class GroqSettings:
    """Settings for the Groq cloud provider.

    The API key is only ever read from the environment
    (:data:`GROQ_API_KEY_ENV_VAR`); it is never stored in source or config
    files, and never logged. ``refine_model``/``generate_model`` are
    configurable so the app is not pinned to one hosted model.
    """

    api_key_env: str = GROQ_API_KEY_ENV_VAR
    refine_model: str = "llama-3.3-70b-versatile"
    generate_model: str = "llama-3.3-70b-versatile"

    #: Chat completion sampling. 0.0 keeps the model from wandering.
    temperature: float = 0.0
    #: Seconds before an API call is abandoned. A cloud provider must not hang
    #: the hotkey listener indefinitely.
    timeout: float = 30.0

    def model_for(self, mode: Mode) -> str:
        """The Groq model configured for `mode`. Raises for an unknown mode."""
        mode = require_mode(mode)
        return self.refine_model if mode is Mode.REFINE else self.generate_model


@dataclass
class ModelSettings:
    """Which text processor serves each mode, and how much room it has.

    These two values are independent on purpose. A cloud provider can back
    GENERATE while REFINE stays local, or vice versa, or either can be pointed
    at a different model without touching anything else.

    The token budgets differ because the two modes produce different amounts of
    text: REFINE rewrites a sentence and must stay close to its length, while
    GENERATE writes an artifact (an email, a function, an explanation) that has
    to fit in the user's paste buffer.
    """

    #: Known providers: local_qwen, groq, none.
    refine_provider: str = "local_qwen"
    generate_provider: str = "local_qwen"

    refine_max_new_tokens: int = 128
    generate_max_new_tokens: int = 512

    def provider_for(self, mode: Mode) -> str:
        """The provider configured for `mode`. Raises for an unknown mode."""
        return getattr(self, PROVIDER_KEYS[require_mode(mode)])

    def max_new_tokens_for(self, mode: Mode) -> int:
        """The generation budget for `mode`. Raises for an unknown mode."""
        return getattr(self, MAX_NEW_TOKENS_KEYS[require_mode(mode)])


@dataclass
class RefinementSettings:
    """Speech-to-text settings, shared by both modes."""

    # Legacy single-provider setting. Only consulted when the file does not
    # define a [models] section (see AppConfig.from_dict).
    provider: str | None = None

    whisper_model: str = "openai/whisper-base"
    language: str = "english"

    # Legacy shared local-model id, superseded by [models.local]. Still read so
    # older config files keep working.
    model_path: str = "Qwen/Qwen3-0.6B"
    adapter_path: str | None = None

    # Legacy shared token budget; see ModelSettings.refine_max_new_tokens.
    max_new_tokens: int = 128


@dataclass
class OutputSettings:
    """Clipboard/paste behaviour (Linux implementation for now)."""

    paste_delay: float = 0.15
    tool_timeout: float = 5.0


@dataclass
class DebugSettings:
    """Debug and text-persistence behaviour.

    Raw microphone audio is *never* logged. Recordings live in memory only, and
    are written to ``debug_dir`` solely when ``enabled`` is true.

    Transcriptions are also kept out of the logs by default: without debug mode
    the pipeline logs only lengths and mode, never the words. Turning debug on
    makes both the recording and the transcript text visible for inspection.
    """

    enabled: bool = False
    debug_dir: str = "debug"


@dataclass
class AppConfig:
    """Top-level MicDrop configuration."""

    audio: AudioSettings = field(default_factory=AudioSettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    local: LocalModelSettings = field(default_factory=LocalModelSettings)
    groq: GroqSettings = field(default_factory=GroqSettings)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)

    # ------------------------------------------------------------------ #
    # Mode-keyed lookups
    # ------------------------------------------------------------------ #

    def provider_for(self, mode: Mode) -> str:
        """Return the configured text-processor provider for `mode`.

        This is the only place mode -> provider resolution happens, so REFINE
        and GENERATE can never be wired to each other's provider by accident.
        """
        return self.models.provider_for(mode)

    def max_new_tokens_for(self, mode: Mode) -> int:
        """Return the generation budget for `mode`."""
        return self.models.max_new_tokens_for(mode)

    def providers(self) -> dict[Mode, str]:
        """Both mode -> provider mappings, for logging and diagnostics."""
        return {mode: self.provider_for(mode) for mode in Mode}

    def groq_api_key(self) -> str | None:
        """Read the Groq API key from the environment (never from disk).

        Prefers the documented ``$MICRODROP_GROQ_API_KEY`` and falls back to
        ``$GROQ_API_KEY`` for convenience. Never logged, never persisted.
        """
        return (
            os.environ.get(self.groq.api_key_env)
            or os.environ.get(GROQ_API_KEY_FALLBACK_ENV_VAR)
            or None
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Build a config from parsed TOML data. Unknown keys are ignored."""
        config = cls()

        flat_sections = {
            "audio": config.audio,
            "hotkeys": config.hotkeys,
            "refinement": config.refinement,
            "output": config.output,
            "debug": config.debug,
        }

        for key, value in data.items():
            target = flat_sections.get(key)
            if target is not None and isinstance(value, dict):
                _apply_kwargs(target, value)

        # [models] may hold nested [models.local] / [models.groq] tables.
        models_section = data.get("models")
        if isinstance(models_section, dict):
            _apply_kwargs(config.models, models_section)
            for nested_name, nested_target in (
                ("local", config.local),
                ("groq", config.groq),
            ):
                nested = models_section.get(nested_name)
                if isinstance(nested, dict):
                    _apply_kwargs(nested_target, nested)

        # Flat [local] / [groq] tables are also accepted.
        for name, target in (("local", config.local), ("groq", config.groq)):
            flat = data.get(name)
            if isinstance(flat, dict):
                _apply_kwargs(target, flat)

        _migrate_legacy_keys(config, data)
        return config

    def resolved_adapter_path(self) -> str | None:
        """Resolve the LoRA adapter path.

        Priority: [models.local] adapter_path, then the legacy
        [refinement] adapter_path, then $MICRODROP_ADAPTER, then the repo's
        adapters/prompt-refiner-lora directory.
        """
        candidates = [
            self.local.adapter_path,
            self.refinement.adapter_path,
            os.environ.get(ADAPTER_ENV_VAR),
        ]
        default = Path(__file__).resolve().parents[2] / "adapters" / "prompt-refiner-lora"
        candidates.append(str(default))

        for candidate in candidates:
            if candidate and Path(candidate).is_dir():
                return candidate
        return None

    def summary(self) -> str:
        """Human-readable summary for the startup banner (no secrets)."""
        text_persistence = (
            self.debug.debug_dir if self.debug.enabled else "off (text not logged or saved)"
        )
        key_state = "set" if self.groq_api_key() else "not set"
        lines = [
            "Configuration:",
            f"  audio:             {self.audio.sample_rate} Hz, {self.audio.channels} ch, "
            f"max {self.audio.max_duration:.0f}s",
            f"  hotkeys:           REFINE={self.hotkeys.refine}  GENERATE={self.hotkeys.generate}",
            f"  keyboard device:   {self.hotkeys.device or '(auto-detect)'}",
            f"  stt model:         {self.refinement.whisper_model} ({self.refinement.language})",
            f"  REFINE:            provider={self.provider_for(Mode.REFINE)} "
            f"max_new_tokens={self.max_new_tokens_for(Mode.REFINE)}",
            f"  GENERATE:          provider={self.provider_for(Mode.GENERATE)} "
            f"max_new_tokens={self.max_new_tokens_for(Mode.GENERATE)}",
            f"  local models:      refine={self.local.refine_model}  "
            f"generate={self.local.generate_model}",
            f"  groq models:       refine={self.groq.refine_model}  "
            f"generate={self.groq.generate_model}  (api key {key_state})",
            f"  debug:             {'enabled' if self.debug.enabled else 'disabled (in-memory only)'}",
            f"  debug text/audio:  {text_persistence}",
        ]
        return "\n".join(lines)


def _migrate_legacy_keys(config: AppConfig, data: dict[str, Any]) -> None:
    """Honour old single-mode keys when [models] is absent.

    Older config files set one provider (and one token budget) for both modes,
    and named the local model in [refinement] model_path. Rather than breaking
    them, those values are copied into the per-mode slots — but only when the
    file does not already define a [models] section, so an explicit per-mode
    setting always wins.
    """
    if not isinstance(data.get("models"), dict):
        legacy_provider = config.refinement.provider
        if legacy_provider:
            config.models.refine_provider = legacy_provider
            config.models.generate_provider = legacy_provider

        refinement_section = data.get("refinement")
        if isinstance(refinement_section, dict) and "max_new_tokens" in refinement_section:
            legacy_tokens = int(refinement_section["max_new_tokens"])
            config.models.refine_max_new_tokens = legacy_tokens
            config.models.generate_max_new_tokens = legacy_tokens

    # [refinement] model_path predates [models.local]; honour it unless the new
    # section set the models explicitly.
    if not isinstance(data.get("local"), dict) and not isinstance(
        (data.get("models") or {}).get("local"), dict
    ):
        if config.local.refine_model == LocalModelSettings.refine_model:
            config.local.refine_model = config.refinement.model_path
        if config.local.generate_model == LocalModelSettings.generate_model:
            config.local.generate_model = config.refinement.model_path


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
        if path.is_file():
            return path
        return None

    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path
        return None

    for candidate in config_search_paths():
        if candidate.is_file():
            return candidate
    return None


def config_path_for_write() -> Path:
    """Where `micdrop setup` should create a config file.

    A file in the working directory wins over the user-level one, so a new
    config lands where the rest of the search order finds it first.
    """
    return Path.cwd() / CONFIG_FILENAME


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
    """A ready-to-copy example config file.

    Must stay in sync with ``micdrop.toml.example`` on disk.
    """
    return _CONFIG_EXAMPLE.strip() + "\n"


_CONFIG_EXAMPLE = """\
# MicDrop configuration (TOML).
#
# Copy to ./micdrop.toml (or ~/.config/micdrop/micdrop.toml), or just run:
#     micdrop setup
#
# Every value shown is the default.

[hotkeys]
# Hold to record, release to process. Names may be written with or without the
# "KEY_" prefix ("RIGHTCTRL" and "KEY_RIGHTCTRL" are the same key).
refine = "RIGHTCTRL"        # Right Ctrl  -> REFINE
generate = "RIGHTALT"       # Right Alt  -> GENERATE
# device = "/dev/input/event3"  # omit to auto-detect the keyboard

[models]
# Text processor per mode. Known values: local_qwen, groq, none.
# They are independent, so either mode can use a cloud provider while the other
# stays local. Run `micdrop models` to see what is available.
refine_provider = "local_qwen"
generate_provider = "local_qwen"

# Generation budget per mode. REFINE rewrites a sentence and stays close to its
# length; GENERATE writes a whole artifact, so it needs more room.
refine_max_new_tokens = 128
generate_max_new_tokens = 512

[models.local]
# Model ids for the local_qwen provider.
refine_model = "Qwen/Qwen3-0.6B"
generate_model = "Qwen/Qwen3-0.6B"
# adapter_path = "adapters/prompt-refiner-lora"   # default: $MICRODROP_ADAPTER or repo adapters/

[models.groq]
# Cloud provider settings. The API key is NOT stored here: it is read from the
# MICRODROP_GROQ_API_KEY environment variable at runtime.
api_key_env = "MICRODROP_GROQ_API_KEY"
refine_model = "llama-3.3-70b-versatile"
generate_model = "llama-3.3-70b-versatile"
temperature = 0.0
timeout = 30.0

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_duration = 60.0    # safety cap; recording stops by itself after this

[refinement]
# Speech-to-text.
whisper_model = "openai/whisper-base"
language = "english"

[output]
paste_delay = 0.15
tool_timeout = 5.0

[debug]
# false: raw audio stays in memory only, and transcripts are neither logged
#        nor written to disk. Microphone audio is never logged either way.
# true:  recordings and transcripts are saved to debug_dir for inspection.
enabled = false
debug_dir = "debug"
"""


MODES = tuple(Mode)  # re-export convenience for callers that iterate modes