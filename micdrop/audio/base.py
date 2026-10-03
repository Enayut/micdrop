"""AudioRecorder interface.

Implementations capture microphone audio and return it as a numpy array in
memory. Nothing is ever written to disk by the recorder itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..app.errors import RecordingError


class AudioRecorder(ABC):
    """Push-to-talk microphone recorder."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of the audio returned by :meth:`stop`."""

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """Whether a recording is currently in progress."""

    @abstractmethod
    def start(self) -> None:
        """Begin capturing audio. Raises RecordingError if it cannot start."""

    @abstractmethod
    def stop(self) -> np.ndarray:
        """Stop capturing and return the audio as a float array.

        Returns an empty array when nothing was captured.
        Raises RecordingError on failure.
        """

    def close(self) -> None:
        """Release resources. Default: nothing to do."""
