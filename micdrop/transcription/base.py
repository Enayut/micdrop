"""SpeechRecognizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..app.errors import TranscriptionError


class SpeechRecognizer(ABC):
    """Turns recorded audio into text."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio (float samples) and return plain text.

        Implementations return "" for unrecognizable audio and raise
        TranscriptionError on failure. They must never raise anything else
        for ordinary failure modes.
        """

    def warmup(self) -> None:
        """Load models/allocate resources ahead of first use. Optional."""
