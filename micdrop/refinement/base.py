"""TextProcessor interface.

A TextProcessor takes the raw transcription and produces the final text for
the given Mode. Implementations may be local models (Qwen3 + LoRA) or API
providers (Groq etc.) — the pipeline only sees this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..app.errors import ProcessingError
from ..app.modes import Mode


class TextProcessor(ABC):
    """Refine or generate text from a raw transcription."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, for logs."""

    @abstractmethod
    def process(self, text: str, mode: Mode) -> str:
        """Process raw transcribed text according to the mode.

        Implementations must not raise for ordinary model-level problems —
        they raise ProcessingError instead. Returning "" signals that
        nothing usable was produced.
        """

    def warmup(self) -> None:
        """Load models ahead of first use. Optional."""
