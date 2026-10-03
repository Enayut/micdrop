"""Core MicDrop pipeline.

Orchestrates the four stages:

    AudioRecorder -> SpeechRecognizer -> TextProcessor -> OutputAdapter
                                                      ^
                                                     mode

The pipeline is platform independent: it only talks to abstract interfaces
(AudioRecorder, SpeechRecognizer, TextProcessor, OutputAdapter). It never
imports evdev, ydotool, wl-copy or any other platform-specific module —
those live under micdrop/platform/ and are wired in by the CLI.

Error handling contract: a failure in any stage is logged clearly and
`handle_release()` returns None. It never raises, so the hotkey listener
survives any transcription/refinement/output failure.

Audio privacy: recordings are kept in memory only and are never written to
disk unless debug mode is explicitly enabled (then they are saved to the
debug directory and kept for inspection).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from .config import AppConfig
from .errors import (
    MicDropError,
    ProcessingError,
    RecordingError,
    TranscriptionError,
)
from .modes import DISPLAY_NAMES, Mode
from ..audio.base import AudioRecorder
from ..output.base import OutputAdapter
from ..refinement.base import TextProcessor
from ..transcription.base import SpeechRecognizer

logger = logging.getLogger("micdrop")


class Pipeline:
    """Push-to-talk pipeline driven by explicit mode events."""

    def __init__(
        self,
        recorder: AudioRecorder,
        recognizer: SpeechRecognizer,
        processor: TextProcessor,
        output: OutputAdapter,
        config: AppConfig | None = None,
        debug_dir: str | None = None,
    ) -> None:
        self.recorder = recorder
        self.recognizer = recognizer
        self.processor = processor
        self.output = output
        self.config = config or AppConfig()
        self.debug_dir = debug_dir or self.config.debug.debug_dir

        self._active_mode: Optional[Mode] = None

    # ------------------------------------------------------------------ #
    # Hotkey-facing API (called by the HotkeyManager)
    # ------------------------------------------------------------------ #

    def handle_press(self, mode: Mode) -> None:
        """A hotkey was pressed: start recording in the given mode."""
        self._active_mode = mode
        try:
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001 - listener must survive
            self._active_mode = None
            logger.error("Recording could not start: %s", exc)
            return

        logger.info("Recording... (mode: %s) — release the hotkey to finish.", DISPLAY_NAMES[mode])

    def handle_release(self, mode: Optional[Mode] = None) -> Optional[str]:
        """A hotkey was released: finish recording and run the pipeline.

        `mode` comes from the release event (the platform layer knows which
        key was released); it must match the mode of the active press.
        Returns the final text on success, None on any failure.
        Never raises.
        """
        if mode is None:
            mode = self._active_mode
        elif self._active_mode is not None and mode is not self._active_mode:
            # Release of a different key than the one that started recording
            # (e.g. both hotkeys held). Ignore it; the active press owns the
            # pipeline until it is released.
            logger.debug(
                "Ignoring release of %s while %s recording is active.",
                DISPLAY_NAMES[mode],
                DISPLAY_NAMES[self._active_mode],
            )
            return None
        self._active_mode = None

        if mode is None:
            # Release without press (e.g. started while another key was held).
            logger.debug("Release event without a matching press; ignoring.")
            return None

        try:
            audio = self.recorder.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("Recording failed: %s", exc)
            return None

        result = self.process(audio, self.recorder.sample_rate, mode)
        return result

    # ------------------------------------------------------------------ #
    # Core stages (usable without hotkeys, e.g. for tests and CLI)
    # ------------------------------------------------------------------ #

    def process(self, audio: np.ndarray, sample_rate: int, mode: Mode) -> Optional[str]:
        """Run transcribe -> process -> deliver for a complete recording.

        Returns the final text, or None if any stage failed or produced
        nothing. Never raises.
        """
        if audio is None or len(audio) == 0:
            logger.info("No audio recorded; nothing to do.")
            return None

        self._maybe_save_debug_audio(audio, sample_rate, mode)

        # ---- Stage 1: speech-to-text ---------------------------------- #
        try:
            text = self.recognizer.transcribe(audio, sample_rate)
        except MicDropError as exc:
            logger.error("Transcription failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - unexpected, still survive
            logger.error("Transcription failed unexpectedly: %s", exc, exc_info=True)
            return None

        text = (text or "").strip()
        if not text:
            logger.info("Transcription was empty; skipping refinement and output.")
            return None

        logger.info("Transcription: %s", text)

        # ---- Stage 2: text processing --------------------------------- #
        try:
            final_text = self.processor.process(text, mode)
        except MicDropError as exc:
            logger.error("Text processing failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Text processing failed unexpectedly: %s", exc, exc_info=True)
            return None

        final_text = (final_text or "").strip()
        if not final_text:
            logger.error("Text processor returned empty output; nothing to deliver.")
            return None

        logger.info("%s result: %s", DISPLAY_NAMES[mode], final_text)

        # ---- Stage 3: output ------------------------------------------- #
        try:
            delivered = self.output.deliver(final_text)
        except MicDropError as exc:
            logger.error("Output failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Output failed unexpectedly: %s", exc, exc_info=True)
            return None

        if not delivered:
            logger.error("Output adapter reported failure; text was NOT delivered.")
            return None

        logger.info("Delivered via %s.", getattr(self.output, "name", type(self.output).__name__))
        return final_text

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _maybe_save_debug_audio(
        self, audio: np.ndarray, sample_rate: int, mode: Mode
    ) -> None:
        """Persist a recording only when debug mode is explicitly enabled."""
        if not self.config.debug.enabled:
            return  # in-memory only: nothing is ever written to disk

        try:
            import soundfile as sf  # deferred: only needed for debug dumps
        except ImportError:
            logger.warning("debug.enabled but soundfile is missing; cannot save recording.")
            return

        from pathlib import Path

        debug_dir = Path(self.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = debug_dir / f"recording-{stamp}-{mode.value}.wav"
        try:
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            logger.debug("Debug recording saved: %s", path)
        except OSError as exc:
            logger.warning("Could not save debug recording: %s", exc)
