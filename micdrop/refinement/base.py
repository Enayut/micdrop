"""TextProcessor interface.

A TextProcessor takes a raw transcription and returns the text that should be
delivered, for exactly one :class:`~micdrop.app.modes.Mode`.

Implementations are *mode-bound*. A processor is constructed for REFINE or for
GENERATE, never for "either", and :meth:`process` refuses a mode it was not
built for with :class:`~micdrop.app.errors.ModeMismatchError`. That is what makes
the safety invariant enforceable rather than aspirational: there is no way to
hand a GENERATE request to the REFINE processor and have it quietly comply.

Providers may be local models (Qwen3 + LoRA), cloud APIs, or passthroughs. The
pipeline only sees this interface, so adding a provider never touches the core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..app.errors import ModeMismatchError, ProcessingError
from ..app.modes import Mode, require_mode


class TextProcessor(ABC):
    """Process text for one specific mode."""

    def __init__(self, mode: Mode) -> None:
        self._mode = require_mode(mode)

    @property
    def mode(self) -> Mode:
        """The single mode this processor is bound to."""
        return self._mode

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, for logs."""

    @abstractmethod
    def process(self, text: str, mode: Mode) -> str:
        """Process raw transcribed text for the bound mode.

        `mode` is passed in by the pipeline so implementations cannot lose
        track of it; it must equal :attr:`mode` or the call raises
        ModeMismatchError. Implementations must not raise for ordinary
        model-level problems — they raise ProcessingError instead. Returning ""
        signals that nothing usable was produced.
        """

    def warmup(self) -> None:
        """Load models ahead of first use. Optional."""

    # -- helpers for implementations ------------------------------------- #

    def _check_mode(self, mode: Mode) -> None:
        """Raise ModeMismatchError unless `mode` is this processor's mode."""
        if mode is not self._mode:
            raise ModeMismatchError(
                f"{type(self).__name__} is bound to {self._mode.value.upper()} "
                f"and cannot process a {mode} request. Refusing rather than "
                f"guessing; the other mode has its own processor."
            )


__all__ = ["TextProcessor", "ProcessingError", "ModeMismatchError"]
