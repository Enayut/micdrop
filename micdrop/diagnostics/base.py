"""Platform-neutral diagnostic checks.

These are the checks that make sense anywhere: what OS this is, whether there is
a display, whether a microphone exists, which Python packages are importable,
whether the configured models are cached, whether a cloud API key is present.

Anything platform-specific lives in the platform package instead — see
``micdrop/platform/linux/diagnostics.py`` — and is reached through the hook in
``micdrop/diagnostics/__init__.py``. That split is what keeps this module free
of evdev / wl-copy / ydotool knowledge, and is enforced by
``micdrop/tests/test_architecture.py``.

Design constraints:

* **Independent checks.** Each is computed on its own, so one failure never
  hides the rest.
* **Never raises.** A diagnostic that crashes is worse than no diagnostic.
* **Never fixes anything silently.** No sudo, no systemctl, no writing to
  /etc. MicDrop reports; the user decides.
* **No secrets in output.** API keys are reported as present/absent, never shown.
"""

from __future__ import annotations

import importlib.util

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..app.config import AppConfig, find_config_file
from ..app.modes import Mode


class Status(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"


#: Statuses that mean "MicDrop cannot work as configured".
BLOCKING = (Status.FAIL,)


@dataclass
class Check:
    """One independent diagnostic result."""

    name: str
    status: Status
    detail: str = ""
    #: What the user should actually do about a WARN/FAIL. Plain text, no sudo
    #: performed — only suggested.
    fix: Optional[str] = None

    @property
    def blocking(self) -> bool:
        return self.status in BLOCKING

    @property
    def ok(self) -> bool:
        return self.status is not Status.OK


@dataclass
class DiagnosticReport:
    """The result of a full sweep."""

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARN]

    @property
    def ready(self) -> bool:
        """True when nothing is FAILed (warnings are tolerated)."""
        return not self.failures

    def find(self, name: str) -> Optional[Check]:
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def status_of(self, name: str) -> Optional[Status]:
        check = self.find(name)
        return check.status if check else None


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run a command, returning (returncode, combined output). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    except Exception:  # noqa: BLE001
        return 1, ""


def _has_module(name: str) -> bool:
    """Whether `name` can be imported — without actually importing it.

    ``importlib.util.find_spec`` only locates the module, so this stays fast
    and side-effect free. A previous version called ``__import__``, which made
    a single `micdrop doctor` spend ~13s inside torch's import and, worse,
    executed arbitrary third-party module code just to answer "is it there?".
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        # ValueError: a module whose parent package is missing.
        return False
    except Exception:  # noqa: BLE001 - never let a probe raise
        return False


def _huggingface_cached(repo_id: str) -> bool:
    """Whether a Hugging Face repo is already in the local cache."""
    try:
        slug = repo_id.replace("/", "--")
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        return (hub / f"models--{slug}").is_dir()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_platform() -> Check:
    system = platform.system()
    if system != "Linux":
        return Check(
            "Platform",
            Status.FAIL,
            f"{system} detected",
            "MicDrop currently supports Linux only.",
        )
    return Check("Platform", Status.OK, f"Linux {platform.release()}")


def check_display() -> Check:
    """Wayland vs X11. MicDrop's paste path is Wayland-oriented."""
    wayland = os.environ.get("WAYLAND_DISPLAY")
    display = os.environ.get("DISPLAY")
    xdg = os.environ.get("XDG_SESSION_TYPE", "")

    if wayland:
        return Check("Display", Status.OK, f"Wayland ({wayland})")
    if xdg.lower() == "wayland" or display:
        return Check(
            "Display",
            Status.WARN,
            "X11 session detected" if display else "XDG_SESSION_TYPE=wayland but WAYLAND_DISPLAY is unset",
            "wl-copy needs a Wayland session. Log in to GNOME on Wayland "
            "(Settings -> About -> Windowing/Display).",
        )
    return Check(
        "Display",
        Status.FAIL,
        "no display session detected",
        "Run MicDrop from inside your desktop session, not over a bare SSH shell.",
    )


def check_desktop() -> Check:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "") or os.environ.get("DESKTOP_NAME", "")
    if not desktop:
        return Check("Desktop", Status.INFO, "unknown (XDG_CURRENT_DESKTOP unset)")
    if "gnome" in desktop.lower():
        return Check("Desktop", Status.OK, desktop)
    return Check(
        "Desktop",
        Status.WARN,
        desktop,
        "MicDrop is developed and tested on GNOME/Wayland. Other desktops may "
        "work but are untested.",
    )


def check_microphone() -> Check:
    if not _has_module("sounddevice"):
        return Check(
            "Microphone",
            Status.FAIL,
            "sounddevice not installed",
            "Install it with: pip install sounddevice",
        )
    try:
        import sounddevice as sd

        inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    except Exception as exc:  # noqa: BLE001
        return Check("Microphone", Status.FAIL, f"could not query audio: {exc}")

    if not inputs:
        return Check(
            "Microphone",
            Status.FAIL,
            "no input device found",
            "Check that a microphone is connected and not muted in Settings -> Sound.",
        )
    default = ""
    try:
        default_index = sd.default.device[0]
        if default_index is not None:
            default = sd.query_devices(default_index, "input")["name"]
    except Exception:  # noqa: BLE001
        default = ""
    return Check("Microphone", Status.OK, f"{len(inputs)} input device(s)" + (f", default: {default}" if default else ""))


def check_python_packages() -> list[Check]:
    """Required third-party modules."""
    checks = []
    for module, install, needed_for in (
        ("numpy", "pip install numpy", "audio"),
        ("sounddevice", "pip install sounddevice", "microphone"),
        ("soundfile", "pip install soundfile", "debug audio"),
        ("transformers", "pip install transformers", "Whisper + local models"),
        ("torch", "pip install torch", "local models"),
        ("peft", "pip install peft", "optional: LoRA adapters"),
    ):
        if _has_module(module):
            checks.append(Check(f"Package: {module}", Status.OK, needed_for))
        else:
            checks.append(
                Check(
                    f"Package: {module}",
                    Status.WARN,
                    f"missing ({needed_for})",
                    install,
                )
            )
    return checks


def check_whisper(config: AppConfig) -> Check:
    if not _has_module("transformers"):
        return Check(
            "Whisper",
            Status.FAIL,
            "transformers not installed",
            "Install it with: pip install transformers torch",
        )
    model = config.refinement.whisper_model
    if _huggingface_cached(model):
        return Check("Whisper", Status.OK, f"{model} (cached)")
    return Check(
        "Whisper",
        Status.OK,
        f"{model} (will download on first run)",
        "The first `micdrop run` needs network access to fetch it, then it is cached.",
    )


def check_local_models(config: AppConfig) -> list[Check]:
    """Refinement and generation models, reported per mode."""
    checks = []
    for mode in Mode:
        provider = config.provider_for(mode)
        if provider != "local_qwen":
            checks.append(
                Check(
                    f"Model ({mode.value})",
                    Status.INFO,
                    f"not local — provider is {provider!r}",
                )
            )
            continue

        model = config.local.model_for(mode)
        if _huggingface_cached(model):
            checks.append(Check(f"Model ({mode.value})", Status.OK, f"{model} (cached)"))
        else:
            checks.append(
                Check(
                    f"Model ({mode.value})",
                    Status.OK,
                    f"{model} (will download on first run)",
                    "The first `micdrop run` needs network access to fetch it.",
                )
            )
    return checks


def check_groq(config: AppConfig) -> list[Check]:
    """Groq availability. Never prints the key."""
    uses_groq = any(config.provider_for(mode) == "groq" for mode in Mode)
    checks: list[Check] = []

    if not _has_module("groq"):
        checks.append(
            Check(
                "Package: groq",
                Status.WARN if not uses_groq else Status.FAIL,
                "not installed",
                "Install it with: pip install groq",
            )
        )
    else:
        checks.append(Check("Package: groq", Status.OK, "installed"))

    key = config.groq_api_key()
    if key:
        checks.append(Check("Groq API key", Status.OK, f"set (${config.groq.api_key_env})"))
    else:
        status = Status.FAIL if uses_groq else Status.INFO
        checks.append(
            Check(
                "Groq API key",
                status,
                f"not set (${config.groq.api_key_env})",
                f"Export it with: export {config.groq.api_key_env}=gsk_...\n"
                f"    Only needed if a mode uses the 'groq' provider.",
            )
        )
    return checks


def check_config_file() -> Check:
    path = find_config_file()
    if path is None:
        return Check(
            "Config file",
            Status.WARN,
            "none found (using defaults)",
            "Run 'micdrop setup' to create one, or copy micdrop.toml.example "
            "to micdrop.toml and edit it.",
        )
    return Check("Config file", Status.OK, str(path))


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
