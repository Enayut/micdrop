"""Platform selection — the one place that decides which backend to use.

Everything else in MicDrop depends on the ABCs and never asks what platform it
is running on. This module is the single boundary where that question gets
answered::

    platform/
      base.py      shared HotkeyManager interface
      linux/       Fedora/GNOME/Wayland implementation
      macos/       (not implemented yet)
      windows/     (not implemented yet)

The output interface lives in ``micdrop.output.base`` — it is not platform code,
just an interface with implementations that are.

To add a platform later: drop in ``platform/<name>/`` with a ``HotkeyManager``
and an ``OutputAdapter``, then register it in :data:`PLATFORM_REGISTRY`. No
other file changes, and no ``if linux:`` branches get added anywhere.

Currently supported: **Linux only**. macOS and Windows are deliberately not
implemented; ``micdrop doctor`` reports that clearly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional

from ..app.config import AppConfig, HotkeySettings, OutputSettings
from ..output.base import OutputAdapter
from .base import HotkeyManager

__all__ = [
    "HotkeyManager",
    "OutputAdapter",
    "PlatformSelection",
    "UnsupportedPlatformError",
    "current_platform",
    "select_platform",
    "PLATFORM_REGISTRY",
    "PLANNED_PLATFORMS",
]


class UnsupportedPlatformError(RuntimeError):
    """Raised when no platform backend is available for this system."""


@dataclass
class PlatformSelection:
    """A resolved platform backend and its factories.

    The factories are callables so that importing this module does not import a
    platform backend: the Linux ``evdev`` import only happens if Linux is
    actually selected.
    """

    name: str
    display_name: str
    #: Each factory takes the *settings* it needs, not the whole AppConfig, so
    #: a backend never has to know about sections it does not use.
    hotkey_factory: Callable[[HotkeySettings], HotkeyManager]
    output_factory: Callable[[OutputSettings], OutputAdapter]
    #: Module paths, so `micdrop doctor` can run platform-specific probes without
    #: this file importing them.
    hotkey_module: str
    output_module: str
    #: Platform-specific diagnostic probes, called by `micdrop doctor`/`setup`.
    #: This is how Linux knows what /dev/uinput is while the core stays clean.
    diagnostics: Optional[Callable[[AppConfig], list]] = None

    def create_hotkeys(self, config: AppConfig) -> HotkeyManager:
        """Build the platform HotkeyManager from [hotkeys]."""
        return self.hotkey_factory(config.hotkeys)

    def create_output(self, config: AppConfig) -> OutputAdapter:
        """Build the platform OutputAdapter from [output]."""
        return self.output_factory(config.output)


def _linux_selection() -> PlatformSelection:
    # Imported lazily: this is the only place the Linux backend is pulled in.
    from .linux import diagnostics as linux_diagnostics
    from .linux.clipboard import LinuxOutputAdapter
    from .linux.keyboard import LinuxHotkeyManager

    return PlatformSelection(
        name="linux",
        display_name="Linux",
        hotkey_factory=LinuxHotkeyManager,
        output_factory=LinuxOutputAdapter,
        hotkey_module="micdrop.platform.linux.keyboard",
        output_module="micdrop.platform.linux.clipboard",
        diagnostics=linux_diagnostics.checks,
    )


#: Platform name -> loader. Add a new platform here and nowhere else.
PLATFORM_REGISTRY: dict[str, Callable[[], PlatformSelection]] = {
    "linux": _linux_selection,
}

#: Platforms that are planned but not implemented, so the error message can
#: say "not yet" instead of "unknown".
PLANNED_PLATFORMS = ("macos", "windows")


def current_platform() -> str:
    """The platform key for this system ("linux", "macos", "windows", ...)."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return sys.platform


def select_platform(name: Optional[str] = None) -> PlatformSelection:
    """Resolve the backend to use, or raise UnsupportedPlatformError.

    Call this once at startup. Everything downstream takes the resolved
    selection, so no other module branches on the operating system.
    """
    key = (name or current_platform()).strip().lower()

    loader = PLATFORM_REGISTRY.get(key)
    if loader is None:
        if key in PLANNED_PLATFORMS:
            raise UnsupportedPlatformError(
                f"MicDrop does not support {key} yet. "
                f"Supported platforms: {', '.join(sorted(PLATFORM_REGISTRY))}."
            )
        raise UnsupportedPlatformError(
            f"No MicDrop backend for this platform ({key!r}). "
            f"Supported platforms: {', '.join(sorted(PLATFORM_REGISTRY))}."
        )
    return loader()