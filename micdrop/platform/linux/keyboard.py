"""Linux hotkey handling via evdev.

This is the keyboard listener from the old main.py, refactored behind the
HotkeyManager interface:

* configurable hotkeys (evdev key names from config, e.g. KEY_RIGHTCTRL)
* two modes (REFINE / GENERATE) bound to two hotkeys
* auto-detects the keyboard device when the configured path is missing
* callback errors are contained: a pipeline failure can never kill the
  listener loop; device errors reconnect instead of crashing

Default binding: Right Ctrl = REFINE, Right Alt = GENERATE.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from ...app.config import HotkeySettings
from ...app.modes import DISPLAY_NAMES, HOTKEY_KEYS, Mode, require_mode
from ..base import HotkeyManager, PressCallback, ReleaseCallback

logger = logging.getLogger("micdrop")

try:
    import evdev
    from evdev import KeyEvent
except ImportError:  # pragma: no cover - only on non-Linux dev machines
    evdev = None
    KeyEvent = None


def key_name_candidates(name: str) -> list[str]:
    """Spellings of a configured hotkey name, most specific first.

    A config may write a key the way a human says it ("Right Ctrl"), the way
    evdev spells it ("KEY_RIGHTCTRL"), or anywhere in between. Rather than
    guessing a single normalization, return the candidates and let the caller
    pick the one that exists.
    """
    raw = (name or "").strip().upper()
    if not raw:
        raise ValueError("Hotkey name is empty. Use a key name like RIGHTCTRL or KEY_F5.")

    underscored = raw.replace("-", "_").replace(" ", "_")
    squashed = raw.replace("-", "").replace(" ", "")

    forms = []
    for form in (underscored, squashed):
        prefixed = form if form.startswith("KEY_") else f"KEY_{form}"
        if prefixed not in forms:
            forms.append(prefixed)
    return forms


def normalize_key_name(name: str) -> str:
    """Normalize a configured hotkey name to the evdev ``KEY_*`` spelling.

    Both forms are accepted so a config can say what a human would write:

        "RIGHTCTRL"     -> "KEY_RIGHTCTRL"
        "KEY_RIGHTCTRL" -> "KEY_RIGHTCTRL"
        "F5"            -> "KEY_F5"

    Use :func:`key_name_candidates` when the exact spelling matters, since a
    name like "Right Ctrl" is ambiguous without knowing the evdev table.
    """
    return key_name_candidates(name)[0]


def resolve_key_code(name: str) -> int:
    """Map a hotkey key name ('RIGHTCTRL', 'Right Ctrl' or 'KEY_RIGHTCTRL') to its code."""
    if evdev is None:
        raise RuntimeError("evdev is not available on this platform")

    candidates = key_name_candidates(name)
    for candidate in candidates:
        code = getattr(evdev.ecodes, candidate, None)
        if isinstance(code, int):
            return code

    raise ValueError(
        f"Unknown key name {name!r} (tried {', '.join(candidates)}). "
        "Use evdev key names like RIGHTCTRL or KEY_RIGHTCTRL."
    )


#: Side words that should render as their own word in the banner, so that
#: "RIGHTCTRL" shows as "Right Ctrl" rather than "Rightctrl".
_SIDE_WORDS = ("LEFT", "RIGHT")


def pretty_key_name(name: str) -> str:
    """'KEY_RIGHTCTRL' -> 'Right Ctrl', 'KEY_F5' -> 'F5'."""
    core = (name or "").strip().upper().removeprefix("KEY_")
    if not core:
        return "(unset)"
    for side in _SIDE_WORDS:
        if core.startswith(side) and len(core) > len(side):
            return f"{side.title()} {core[len(side):].title()}"
    return core.title()


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
        """Bind one key to one mode.

        Raises ValueError if the same key is bound to both modes — that would
        make the two modes indistinguishable, which defeats the whole design.
        """
        require_mode(mode)

        key_name = self.settings.key_for(mode)
        code = resolve_key_code(key_name)
        if code in self._bindings:
            existing = self._bindings[code][0]
            raise ValueError(
                f"Hotkey conflict: {key_name} is bound to both "
                f"{DISPLAY_NAMES[existing]} and {DISPLAY_NAMES[mode]}."
            )
        self._bindings[code] = (mode, on_press, on_release)
        logger.debug(
            "Bound %s to %s (code %d)", HOTKEY_KEYS[mode], DISPLAY_NAMES[mode], code
        )

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
        path = (self.settings.device or "").strip()

        if not path:
            logger.info("No keyboard device configured; auto-detecting...")
        else:
            try:
                device = evdev.InputDevice(path)
                logger.info("Keyboard: %s (%s)", device.name, path)
                return device
            except (OSError, PermissionError) as exc:
                logger.warning("Cannot open %s (%s); trying auto-detection.", path, exc)

        detected = find_keyboard_device(required)
        if detected is None:
            raise RuntimeError(
                "No keyboard with the configured hotkey keys was found. Set "
                "[hotkeys] device in micdrop.toml, and check that your user can "
                "read the input devices."
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
