"""Linux hotkey handling via evdev.

This is the keyboard listener from the old main.py, refactored behind the
HotkeyManager interface:

* configurable hotkeys (evdev key names from config, e.g. KEY_RIGHTCTRL)
* two modes (REFINE / GENERATE) bound to two hotkeys
* auto-detects the keyboard device when the configured path is missing
* callback errors are contained: a pipeline failure can never kill the
  listener loop; device errors reconnect instead of crashing
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from ...app.config import HotkeySettings
from ...app.modes import Mode
from ..base import HotkeyManager, PressCallback, ReleaseCallback

logger = logging.getLogger("micdrop")

try:
    import evdev
    from evdev import KeyEvent
except ImportError:  # pragma: no cover - only on non-Linux dev machines
    evdev = None
    KeyEvent = None


def resolve_key_code(name: str) -> int:
    """Map an evdev key name (e.g. 'KEY_RIGHTCTRL') to its code."""
    if evdev is None:
        raise RuntimeError("evdev is not available on this platform")
    code = getattr(evdev.ecodes, name, None)
    if code is None or not isinstance(code, int):
        raise ValueError(
            f"Unknown key name {name!r}. Use evdev key names like KEY_RIGHTCTRL."
        )
    return code


def find_keyboard_device(required_keys: list[int]) -> Optional[str]:
    """Find an input device that can produce all required keys.

    Skips virtual devices (ydotoold's own uinput device also emits the
    keys we press with it, which would self-trigger). Returns None if no
    match is found.
    """
    if evdev is None:
        return None

    for path in sorted(evdev.list_devices()):
        try:
            device = evdev.InputDevice(path)
            if "virtual" in device.name.lower():
                continue
            caps = device.capabilities(verbose=False)
            key_codes = caps.get(evdev.ecodes.EV_KEY, ())
            if all(code in key_codes for code in required_keys):
                return path
        except (OSError, PermissionError):
            continue
    return None


def find_virtual_device(required_keys: list[int]) -> Optional[str]:
    """Find a virtual uinput device that can produce the required keys.

    Used by `micdrop hotkey-test` to round-trip hotkeys through ydotool
    without touching the physical keyboard.
    """
    devices = find_all_virtual_devices(required_keys)
    return devices[0] if devices else None


def find_all_virtual_devices(required_keys: list[int]) -> list[str]:
    """Find every virtual uinput device that can produce the required keys.

    ydotoold can expose several virtual devices; injected events surface on
    one of them, so tests listen on all matches.
    """
    if evdev is None:
        return []

    matches = []
    for path in sorted(evdev.list_devices()):
        try:
            device = evdev.InputDevice(path)
            if "virtual" not in device.name.lower():
                continue
            caps = device.capabilities(verbose=False)
            key_codes = caps.get(evdev.ecodes.EV_KEY, ())
            if all(code in key_codes for code in required_keys):
                matches.append(path)
        except (OSError, PermissionError):
            continue
    return matches


class LinuxHotkeyManager(HotkeyManager):
    """Reads keyboard events with evdev and dispatches mode callbacks."""

    def __init__(self, settings: HotkeySettings | None = None) -> None:
        if evdev is None:
            raise RuntimeError(
                "evdev is required for the Linux hotkey manager "
                "(pip install evdev; Linux only)."
            )
        self.settings = settings or HotkeySettings()
        self._bindings: dict[int, tuple[Mode, PressCallback, ReleaseCallback]] = {}
        self._device = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- HotkeyManager interface ----------------------------------------- #

    def register(self, mode: Mode, on_press: PressCallback, on_release: ReleaseCallback) -> None:
        key_name = {
            Mode.REFINE: self.settings.refine,
            Mode.GENERATE: self.settings.generate,
        }[mode]
        code = resolve_key_code(key_name)
        if code in self._bindings:
            raise ValueError(
                f"Hotkey conflict: {key_name} is bound to more than one mode."
            )
        self._bindings[code] = (mode, on_press, on_release)

    def start(self) -> None:
        """Run the listener loop on the calling thread (blocks)."""
        self._loop()

    def start_in_background(self) -> threading.Thread:
        """Run the listener on a daemon thread; returns the thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True, name="micdrop-hotkeys")
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop_event.set()

    # -- internals -------------------------------------------------------- #

    def _open_device(self):
        """Open the configured keyboard, falling back to auto-detection."""
        required = list(self._bindings.keys())
        path = self.settings.device

        try:
            device = evdev.InputDevice(path)
            logger.info("Keyboard: %s (%s)", device.name, path)
            return device
        except (OSError, PermissionError) as exc:
            logger.warning("Cannot open %s (%s); trying auto-detection.", path, exc)

        detected = find_keyboard_device(required)
        if detected is None:
            raise RuntimeError(
                f"Could not open {path} and no keyboard with the configured "
                "hotkey keys was auto-detected. Check the [hotkeys] device "
                "setting and that your user can read /dev/input/*."
            )
        device = evdev.InputDevice(detected)
        logger.info("Keyboard (auto-detected): %s (%s)", device.name, detected)
        return device

    def _loop(self) -> None:
        codes = {code: binding for code, binding in self._bindings.items()}
        reconnect_delay = 2.0

        while not self._stop_event.is_set():
            try:
                device = self._open_device()
                for event in device.read_loop():
                    if self._stop_event.is_set():
                        return
                    if event.type != evdev.ecodes.EV_KEY:
                        continue

                    key_event = evdev.categorize(event)
                    binding = codes.get(key_event.scancode)
                    if binding is None:
                        continue

                    mode, on_press, on_release = binding

                    if key_event.keystate == KeyEvent.key_down:
                        self._dispatch(f"{mode.value} press", on_press, mode)
                    elif key_event.keystate == KeyEvent.key_up:
                        self._dispatch(f"{mode.value} release", on_release, mode)
                    # keystate == 2 is auto-repeat: deliberately ignored.

            except (OSError, RuntimeError) as exc:
                if self._stop_event.is_set():
                    return
                logger.error(
                    "Keyboard listener lost (%s). Reconnecting in %.0fs...", exc, reconnect_delay
                )
                self._stop_event.wait(reconnect_delay)
            except Exception as exc:  # noqa: BLE001 - never die permanently
                if self._stop_event.is_set():
                    return
                logger.error("Keyboard listener error (%s). Retrying...", exc)
                self._stop_event.wait(reconnect_delay)

    @staticmethod
    def _dispatch(what: str, callback, mode: Mode) -> None:
        try:
            callback(mode)
        except Exception as exc:  # noqa: BLE001 - a callback must not kill the loop
            logger.error("Hotkey callback failed (%s): %s", what, exc, exc_info=True)
