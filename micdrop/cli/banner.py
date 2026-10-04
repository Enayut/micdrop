"""Startup banner for `micdrop run`.

Kept separate from the command logic so the wiring of hotkey -> mode ->
provider is printed from one place and can be asserted in tests.
"""

from __future__ import annotations

from ..app.config import AppConfig
from ..app.modes import DESCRIPTIONS, DISPLAY_NAMES, Mode


def hotkey_lines(config: AppConfig) -> dict[Mode, str]:
    """Human-readable hotkey label per mode, e.g. 'Right Ctrl'."""
    from ..platform.linux.keyboard import pretty_key_name

    return {mode: pretty_key_name(config.hotkeys.key_for(mode)) for mode in Mode}


def banner_lines(config: AppConfig) -> list[str]:
    """The startup banner as a list of lines."""
    keys = hotkey_lines(config)
    lines = [
        "MicDrop — voice-to-prompt",
        "",
        f"  Hold {keys[Mode.REFINE]} and speak  ->  REFINE   (clean up your words)",
        f"  Hold {keys[Mode.GENERATE]} and speak  ->  GENERATE (produce the artifact)",
        "",
    ]
    for mode in Mode:
        lines.append(
            f"  {DISPLAY_NAMES[mode]:8s} {keys[mode]:10s} "
            f"provider={config.provider_for(mode):12s} {DESCRIPTIONS[mode]}"
        )
    lines.append("")
    return lines


def print_banner(config: AppConfig) -> None:
    print("\n".join(banner_lines(config)))
    print(config.summary())
    print("\nPress Ctrl+C in the terminal to quit.")
