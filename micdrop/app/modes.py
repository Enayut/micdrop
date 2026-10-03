"""User modes.

MicDrop has exactly two modes:

* REFINE   — clean up the transcription (fix errors, grammar, filler) while
             strictly preserving meaning. Never invents information.
* GENERATE — interpret the spoken request and turn it into a useful final
             prompt/output. May restructure and moderately expand.

The mode is an explicit part of the application state, never inferred from
the text: the hotkey selects the mode, the mode is passed through the whole
pipeline, and each TextProcessor decides how to behave per mode.
"""

import enum


class Mode(enum.Enum):
    """The two user-facing processing modes."""

    REFINE = "refine"
    GENERATE = "generate"


#: Human-readable display names, for logs and CLI output.
DISPLAY_NAMES = {
    Mode.REFINE: "REFINE",
    Mode.GENERATE: "GENERATE",
}

#: Short description of each mode, for `--help` and startup banners.
DESCRIPTIONS = {
    Mode.REFINE: "Clean up the transcription, preserve exact meaning",
    Mode.GENERATE: "Interpret the request and produce a useful final prompt",
}


def parse_mode(name: str) -> Mode:
    """Parse a mode name from CLI/config. Raises ValueError if unknown."""
    try:
        return Mode(name.strip().lower())
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        raise ValueError(f"Unknown mode {name!r}. Valid modes: {valid}") from None
