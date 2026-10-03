"""HotkeyManager interface.

A HotkeyManager watches global input devices and turns hotkey press/release
events into mode-tagged callbacks. The core pipeline never talks to evdev,
uinput or any other platform API — only to this interface, so Windows and
macOS implementations can be added later without touching the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..app.modes import Mode

# on_press(mode) / on_release(mode)
PressCallback = Callable[[Mode], None]
ReleaseCallback = Callable[[Mode], None]


class HotkeyManager(ABC):
    """Maps configured hotkeys to pipeline events."""

    @abstractmethod
    def register(self, mode: Mode, on_press: PressCallback, on_release: ReleaseCallback) -> None:
        """Bind press/release callbacks to the hotkey configured for mode."""

    @abstractmethod
    def start(self) -> None:
        """Run the event loop. Blocks until stop() or a fatal platform error."""

    def stop(self) -> None:
        """Request shutdown (from another thread). Default: unsupported."""
