"""Microphone recording via sounddevice.

This is the audio-capture implementation moved out of the old main.py.
sounddevice is cross-platform (PortAudio), so this class is usable on Linux,
Windows and macOS alike; only the hotkey/clipboard layers are platform specific.

Recording safety: a push-to-talk hotkey can be held down by accident (a stuck
key, a distracted hand, a child). :attr:`AudioSettings.max_duration` puts a
hard cap on a single recording; when it is reached the recorder stops itself
from the audio callback thread, so a busy main thread cannot miss the limit. The
captured audio is kept — hitting the cap is a normal end of recording, and the
pipeline still transcribes and delivers it. The default is 60 seconds.

The audio itself is memory-only: nothing is written to disk unless the pipeline
is explicitly in debug mode.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..app.config import AudioSettings
from ..app.errors import RecordingError
from .base import AudioRecorder

logger = logging.getLogger("micdrop")


class MaxDurationReached(Exception):
    """Raised internally when a recording hits its configured duration cap.

    Not a :class:`RecordingError`: nothing went wrong, the safety limit simply
    fired. The pipeline treats it as a normal end-of-recording.
    """


class SoundDeviceRecorder(AudioRecorder):
    """Records from the default input device while `start()` is active."""

    def __init__(self, settings: AudioSettings | None = None) -> None:
        self._settings = settings or AudioSettings()
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream = None
        self._started_at: float | None = None
        #: Set when the duration cap fired, so the next stop() can report it.
        self._hit_max_duration = False

    # -- AudioRecorder interface ---------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return self._settings.sample_rate

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def max_duration(self) -> float:
        """The configured cap in seconds; 0 means unlimited."""
        return float(self._settings.max_duration or 0.0)

    @property
    def elapsed(self) -> float:
        """Seconds since start(), or 0.0 when not recording."""
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def start(self) -> None:
        if self._recording:
            return

        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise RecordingError(f"sounddevice is unavailable: {exc}") from exc

        with self._lock:
            self._frames = []
        self._recording = True
        self._started_at = time.monotonic()
        self._hit_max_duration = False

        try:
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype=self._settings.dtype,
                blocksize=self._settings.blocksize or None,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:  # sd.PortAudioError and friends
            self._recording = False
            self._started_at = None
            self._stream = None
            raise RecordingError(f"Could not open the microphone: {exc}") from exc

    def stop(self) -> np.ndarray:
        """Stop recording and return everything captured.

        Safe to call twice, and safe to call when the recorder already stopped
        itself at the duration cap: in that case the buffered audio is still
        returned rather than thrown away, because those 60 seconds are the only
        record of what the user said.
        """
        was_recording = self._recording
        self._recording = False
        elapsed = self.elapsed
        self._started_at = None

        if was_recording and self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # stream errors must not lose the audio
                self._stream = None
                raise RecordingError(f"Stopping the recording failed: {exc}") from exc
        self._stream = None

        with self._lock:
            frames = self._frames
            self._frames = []

        if self._hit_max_duration:
            self._hit_max_duration = False
            logger.warning(
                "Recording stopped automatically after %.1fs (max_duration). "
                "Raise [audio] max_duration to speak for longer.",
                elapsed,
            )

        if not frames:
            return np.empty(0, dtype=self._settings.dtype)

        audio = np.concatenate(frames, axis=0)
        if audio.ndim > 1 and audio.shape[1] == 1:
            # Mono capture: flatten to a 1-D samples array, which is what
            # recognizers (e.g. the Whisper ASR pipeline) expect.
            audio = audio.reshape(-1)
        return audio

    # -- internals -------------------------------------------------------- #

    def _audio_callback(self, indata, frames_count, time_info, status) -> None:
        if status:
            # Non-fatal overflows etc.; keep recording.
            logger.debug("Audio stream status: %s", status)

        if not self._recording:
            return

        # Enforce the cap here, on the callback thread, so it cannot be missed
        # by a main thread that is busy and only checks on release.
        limit = self.max_duration
        if limit and self.elapsed >= limit:
            self._recording = False
            self._hit_max_duration = True
            return  # drop this block; the frames already buffered are kept

        with self._lock:
            self._frames.append(indata.copy())