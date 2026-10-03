"""Exceptions shared across the MicDrop pipeline.

Every pipeline stage raises a subclass of MicDropError. The pipeline catches
these (and any unexpected exception) per stage so that a failure in
transcription, refinement, or output can never crash the hotkey listener.
"""


class MicDropError(Exception):
    """Base class for all MicDrop errors."""


class RecordingError(MicDropError):
    """Microphone recording failed."""


class TranscriptionError(MicDropError):
    """Speech-to-text failed."""


class ProcessingError(MicDropError):
    """Text refinement/generation failed."""


class OutputError(MicDropError):
    """Delivering the result to the output adapter failed."""
