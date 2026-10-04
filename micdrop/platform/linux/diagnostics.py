"""Linux-specific diagnostic checks: evdev, uinput, groups, hotkeys, clipboard.

These live in the platform package, not in the core, because knowing what
/dev/uinput is or that ydotoold needs a socket is Linux knowledge. The core
diagnostics engine calls :func:`checks` through the platform boundary
(``PlatformSelection.diagnostics``) and stays platform-neutral.

The split means adding a macOS or Windows backend also means adding its own
diagnostics module, with no change to the core or to the CLI.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from ...app.config import AppConfig
from ...app.modes import Mode
from ...diagnostics.base import Check, Status


def checks(config: AppConfig) -> list[Check]:
    """All Linux-specific checks, each computed independently."""
    results: list[Check] = []
    results.extend(_check_evdev())
    results.extend(_check_uinput())
    results.extend(_check_input_group())
    results.extend(_check_hotkey_keys(config))
    results.extend(_check_clipboard_tools())
    return results


def _check_evdev() -> list[Check]:
    try:
        import evdev
    except ImportError:
        return [
            Check(
                "evdev",
                Status.FAIL,
                "not installed",
                "Install it with: pip install evdev  (Linux only)",
            )
        ]
    version = getattr(evdev, "__version__", None) or "installed"
    return [Check("evdev", Status.OK, str(version))]


def _check_uinput() -> list[Check]:
    """/dev/uinput must be writable for ydotool to make a virtual keyboard."""
    if os.access("/dev/uinput", os.W_OK):
        return [Check("/dev/uinput", Status.OK, "writable")]

    return [
        Check(
            "/dev/uinput",
            Status.FAIL,
            "not writable by this user",
            "MicDrop itself never needs root, but uinput access does. Run this "
            "once as an administrator:\n"
            "    echo 'KERNEL==\"uinput\", MODE=\"0660\", GROUP=\"input\"' | \\\n"
            "      sudo tee /etc/udev/rules.d/80-uinput.rules\n"
            "    sudo udevadm control --reload-rules && sudo udevadm trigger /dev/uinput\n"
            "    sudo usermod -aG input $USER   # then log out and back in",
        )
    ]


def _check_input_group() -> list[Check]:
    try:
        import grp

        groups = {g.gr_name for g in grp.getgrall()}
        in_group = "input" in groups
    except Exception:  # noqa: BLE001
        in_group = False
    in_group = in_group or "input" in os.environ.get("GROUPS", "").split()

    if in_group:
        return [Check("Group: input", Status.OK, "member")]
    return [
        Check(
            "Group: input",
            Status.WARN,
            "not a member",
            "You may not be able to open input devices. Run:\n"
            "    sudo usermod -aG input $USER\n"
            "then log out and back in.",
        )
    ]


def _check_hotkey_keys(config: AppConfig) -> list[Check]:
    """Resolve the configured hotkeys and find a keyboard that has them."""
    from .keyboard import find_keyboard_device, resolve_key_code

    try:
        codes = {mode: resolve_key_code(config.hotkeys.key_for(mode)) for mode in Mode}
    except ValueError as exc:
        return [
            Check(
                "Hotkeys",
                Status.FAIL,
                str(exc),
                'Use evdev key names such as "RIGHTCTRL" or "KEY_F5" under [hotkeys].',
            )
        ]

    if codes[Mode.REFINE] == codes[Mode.GENERATE]:
        return [
            Check(
                "Hotkeys",
                Status.FAIL,
                "both modes are bound to the same key",
                "Give each mode a different key under [hotkeys].",
            )
        ]

    summary = ", ".join(
        f"{mode.value}={config.hotkeys.key_for(mode)}({codes[mode]})" for mode in Mode
    )
    found = find_keyboard_device(list(codes.values()))
    if found:
        return [Check("Hotkeys", Status.OK, f"{summary} on {found}")]

    return [
        Check(
            "Hotkeys",
            Status.FAIL,
            f"{summary} — no keyboard found with both keys",
            "MicDrop reads key events from the kernel, so it does not depend on "
            "GNOME shortcut settings. Check that your user can read the input "
            "devices (see the 'Group: input' check), and that the keyboard you "
            "type on actually has both keys.",
        )
    ]


def _check_clipboard_tools() -> list[Check]:
    """wl-copy, ydotool, and the ydotoold daemon."""
    results: list[Check] = []
    for tool, package in (("wl-copy", "wl-clipboard"), ("ydotool", "ydotool")):
        path = shutil.which(tool)
        if path:
            results.append(Check(tool, Status.OK, path))
        else:
            results.append(
                Check(
                    tool,
                    Status.FAIL,
                    "not installed",
                    f"Install it with: sudo dnf install {package}",
                )
            )

    from .clipboard import check_dependencies

    problems = check_dependencies()
    if not problems:
        results.append(Check("ydotoold", Status.OK, "daemon running"))
        return results

    daemon_problems = [p for p in problems if "daemon" in p or "socket" in p]
    if daemon_problems:
        results.append(
            Check(
                "ydotoold",
                Status.FAIL,
                "daemon not reachable",
                "Start it as your own user (no root):\n"
                "    sudo systemctl disable --now ydotool.service\n"
                "    systemctl --user enable --now ydotool.service\n"
                "Check it with: systemctl --user status ydotool.service",
            )
        )
    else:
        results.append(Check("ydotoold", Status.WARN, "; ".join(problems)))
    return results


__all__ = ["checks"]