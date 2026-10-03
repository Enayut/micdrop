"""Microphone recording via sounddevice.

This is the audio-capture implementation moved out of the old main.py.
sounddevice is cross-platform (PortAudio), so this class is usable on
Linux, Windows and macOS alike; only the hotkey/clipboard layers are
platform specific.
"""

from __future__ import annotations

import threading

import numpy as np

from ..app.config import AudioSettings
from ..app.errors import RecordingError
from .base import AudioRecorder


class SoundDeviceRecorder(AudioRecorder):
    """Records from the default input device while `start()` is active."""

    def __init__(self, settings: AudioSettings | None = None) -> None:
        self._settings = settings or AudioSettings()
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream = None

    # -- AudioRecorder interface ---------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return self._settings.sample_rate

    @property
    def is_recording(self) -> bool:
        return self._recording

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
            self._stream = None
            raise RecordingError(f"Could not open the microphone: {exc}") from exc

    def stop(self) -> np.ndarray:
        if not self._recording:
            return np.empty(0, dtype=self._settings.dtype)

        self._recording = False

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as exc:  # stream errors must not lose the audio
            raise RecordingError(f"Stopping the recording failed: {exc}") from exc
        finally:
            self._stream = None

        with self._lock:
            frames = self._frames
            self._frames = []

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
            print(f"Audio: {status}")
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())
