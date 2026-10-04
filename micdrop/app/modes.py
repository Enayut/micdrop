"""User modes.

MicDrop has exactly two modes, and they are different products:

* REFINE   — clean up what the user said. Remove disfluency and repetition,
             correct obvious Whisper mistakes, fix grammar and punctuation,
             keep technical terminology. Strictly preserves the meaning and the
             scope of the request: it never answers the request, never invents
             requirements, never expands a short request into a large prompt.
             The output is what the user meant to type, tidied.

* GENERATE — carry out the user's instruction. Follow it and produce the
             artifact that was asked for (the email, the code, the
             explanation), with the user's explicit constraints preserved and
             nothing unrelated added. The output is directly pasteable into
             whatever application has focus.

The mode is an explicit, external choice made before any processing starts:
the hotkey selects it, the platform layer passes it with every press/release
event, and the pipeline carries it through transcription and processing. It is
never inferred from the words that were spoken.

There is deliberately no third mode and no "auto" mode. When a mode is missing
or a processor is asked to handle a mode it was not built for, MicDrop raises
:class:`~micdrop.app.errors.ModeError` instead of guessing — see
``micdrop/refinement/router.py`` for the enforcement point.
"""

import enum

from .errors import UnknownModeError


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
    Mode.REFINE: "Clean up the words you said, preserving the exact meaning",
    Mode.GENERATE: "Carry out your instruction and produce the requested result",
}

#: One-line statement of what each mode must and must not do. Kept here, next
#: to the enum, so the contract is visible in a single place and can be asserted
#: on by tests instead of being duplicated in prompt text and comments.
MODE_CONTRACTS = {
    Mode.REFINE: {
        "preserve": "the user's meaning, scope and wording",
        "forbid": (
            "answering the request, adding information, inferring missing "
            "requirements, expanding a short request, or explaining the edits"
        ),
        "output": "only the refined text, with no reasoning",
    },
    Mode.GENERATE: {
        "preserve": "every explicit constraint the user stated",
        "forbid": (
            "discussing the transformation, exposing reasoning, or adding "
            "content the user did not ask for"
        ),
        "output": "only the requested artifact, ready to paste",
    },
}

#: Config key under `[hotkeys]` that belongs to each mode. Keeping this explicit
#: stops a future edit from wiring one mode's hotkey to the other.
HOTKEY_KEYS = {
    Mode.REFINE: "refine",
    Mode.GENERATE: "generate",
}

#: Config key under `[models]` selecting the provider for each mode.
PROVIDER_KEYS = {
    Mode.REFINE: "refine_provider",
    Mode.GENERATE: "generate_provider",
}

#: Config key under `[models]` setting the token budget for each mode.
MAX_NEW_TOKENS_KEYS = {
    Mode.REFINE: "refine_max_new_tokens",
    Mode.GENERATE: "generate_max_new_tokens",
}


def parse_mode(name: str) -> Mode:
    """Parse a mode name from CLI/config. Raises UnknownModeError if unknown."""
    if isinstance(name, Mode):
        return name
    try:
        return Mode(str(name).strip().lower())
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        raise UnknownModeError(
            f"Unknown mode {name!r}. Valid modes: {valid}"
        ) from None


def require_mode(value: object) -> Mode:
    """Validate that `value` is a real Mode, or raise UnknownModeError.

    Used at every boundary where a mode enters the system, so a bad value fails
    loudly instead of being treated as "whatever mode we happen to be in".
    """
    if isinstance(value, Mode):
        return value
    raise UnknownModeError(f"Expected a Mode, got {value!r}.")
