"""Exceptions shared across the MicDrop pipeline.

Every pipeline stage raises a subclass of MicDropError. The pipeline catches
these (and any unexpected exception) per stage so that a failure in
transcription, refinement, or output can never crash the hotkey listener.

The mode-routing errors are separate on purpose. A MicDropError means "this
attempt failed, move on"; a ModeError means "the request itself was
incoherent — a mode that does not exist, or a processor being asked to handle
the wrong mode". Both are contained by the pipeline so the listener stays alive,
but a ModeError is never recovered from by substituting a different mode.
"""


class MicDropError(Exception):
    """Base class for all MicDrop errors."""


class RecordingError(MicDropError):
    """Microphone recording failed."""


class TranscriptionError(MicDropError):
    """Speech-to-text failed."""


class ProcessingError(MicDropError):
    """Text refinement/generation failed."""


class UnknownProviderError(MicDropError):
    """The configured text provider does not exist or is not implemented yet."""


class ModeError(MicDropError):
    """Base class for mode-routing violations.

    Raised whenever something is asked to handle a mode it was not built for:
    an unknown mode name, a REFINE processor handed a GENERATE request, or a
    mode with no processor configured. Mode errors are never recovered from by
    silently switching modes — that is the whole point of them.
    """


class UnknownModeError(ModeError):
    """A mode value is not one of the supported user modes."""


class ModeMismatchError(ModeError):
    """A processor was asked to handle a mode it is not bound to.

    This is the guard behind the safety invariant: a REFINE processor refuses
    GENERATE work instead of quietly doing something else.
    """


class ModeProcessorUnavailableError(ModeError):
    """No processor is configured for the requested mode.

    The pipeline then does nothing for that hotkey cycle. It never borrows the
    other mode's processor.
    """


class OutputError(MicDropError):
    """Delivering the result to the output adapter failed."""
