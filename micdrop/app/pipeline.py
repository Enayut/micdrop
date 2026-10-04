"""Core MicDrop pipeline.

Orchestrates the four stages:

    AudioRecorder -> SpeechRecognizer -> TextProcessor -> OutputAdapter
                                                      ^
                                                     mode

The pipeline is platform independent: it only talks to abstract interfaces
(AudioRecorder, SpeechRecognizer, TextProcessor, OutputAdapter). It never
imports evdev, ydotool, wl-copy or any other platform-specific module — those
live under micdrop/platform/ and are wired in by the CLI. Feeding it a Windows
or macOS HotkeyManager/OutputAdapter needs no change here.

Mode handling
-------------
A mode is chosen *before* processing starts (the hotkey selects it) and is
carried explicitly through every stage. :meth:`process` validates it, and the
TextProcessor it dispatches to is bound to exactly that mode. The pipeline has
no "default mode" and never guesses one from the text.

Error handling contract
-----------------------
A failure in any stage is logged clearly and `handle_release()` returns None. It
never raises, so the hotkey listener survives any transcription, refinement or
output failure, and simply waits for the next press. A hotkey cycle whose
*press* failed to start recording is abandoned at the release rather than
running the rest of the pipeline on nothing.

Privacy
-------
Raw microphone audio is never logged. It stays in memory; it is only written to
disk when debug mode is explicitly enabled. Transcriptions follow the same rule:
without debug mode the logs record only the mode and character counts, never the
words. See :meth:`_log_text`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import AppConfig
from .errors import MicDropError
from .modes import DISPLAY_NAMES, Mode, require_mode
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
        # True when a press failed to start recording: the matching release
        # must be dropped rather than run through an empty pipeline.
        self._press_failed = False

    @property
    def active_mode(self) -> Optional[Mode]:
        """The mode of the recording currently in progress, if any."""
        return self._active_mode

    @property
    def debug_enabled(self) -> bool:
        return bool(self.config.debug.enabled)

    # ------------------------------------------------------------------ #
    # Hotkey-facing API (called by the HotkeyManager)
    # ------------------------------------------------------------------ #

    def handle_press(self, mode: Mode) -> None:
        """A hotkey was pressed: start recording in the given mode."""
        try:
            mode = require_mode(mode)
        except MicDropError as exc:
            logger.error("Ignoring hotkey press with a bad mode: %s", exc)
            return

        self._active_mode = mode
        self._press_failed = False
        try:
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001 - listener must survive
            self._active_mode = None
            self._press_failed = True
            logger.error("Recording could not start: %s", exc)
            return

        logger.info(
            "Recording... (mode: %s) — release the hotkey to finish.",
            DISPLAY_NAMES[mode],
        )

    def handle_release(self, mode: Optional[Mode] = None) -> Optional[str]:
        """A hotkey was released: finish recording and run the pipeline.

        `mode` comes from the release event (the platform layer knows which
        key was released); it must match the mode of the active press.
        Returns the final text on success, None on any failure.
        Never raises.
        """
        if mode is not None:
            try:
                mode = require_mode(mode)
            except MicDropError as exc:
                logger.error("Ignoring hotkey release with a bad mode: %s", exc)
                return None

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

        if self._press_failed:
            # Recording never started, so there is nothing to process. Running
            # the pipeline here would transcribe silence and paste a
            # hallucination, so the cycle is abandoned instead.
            self._press_failed = False
            logger.debug("Dropping release of a press that failed to start.")
            return None

        if mode is None:
            # Release without press (e.g. started while another key was held).
            logger.debug("Release event without a matching press; ignoring.")
            return None

        try:
            audio = self.recorder.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("Recording failed: %s", exc)
            return None

        return self.process(audio, self.recorder.sample_rate, mode)

    # ------------------------------------------------------------------ #
    # Core stages (usable without hotkeys, e.g. for tests and CLI)
    # ------------------------------------------------------------------ #

    def process(self, audio: np.ndarray, sample_rate: int, mode: Mode) -> Optional[str]:
        """Run transcribe -> process -> deliver for a complete recording.

        Returns the final text, or None if any stage failed or produced
        nothing. Never raises.
        """
        try:
            mode = require_mode(mode)
        except MicDropError as exc:
            logger.error("Cannot process audio: %s", exc)
            return None

        if audio is None or len(audio) == 0:
            logger.info("No audio recorded; nothing to do.")
            return None

        logger.debug(
            "Processing %.2fs of audio at %d Hz in %s mode.",
            len(audio) / float(sample_rate or 1),
            sample_rate,
            mode.value.upper(),
        )
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
            logger.info("Transcription was empty; skipping processing and output.")
            return None

        self._log_text("Transcription", text, mode)

        # ---- Stage 2: text processing --------------------------------- #
        # The processor is dispatched to by mode. It is bound to that mode and
        # will refuse anything else, so REFINE text can never be generated.
        try:
            final_text = self.processor.process(text, mode)
        except MicDropError as exc:
            logger.error("%s processing failed: %s", DISPLAY_NAMES[mode], exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s processing failed unexpectedly: %s",
                DISPLAY_NAMES[mode],
                exc,
                exc_info=True,
            )
            return None

        final_text = (final_text or "").strip()
        if not final_text:
            logger.error(
                "%s processor returned empty output; nothing to deliver.",
                DISPLAY_NAMES[mode],
            )
            return None

        self._log_text(f"{DISPLAY_NAMES[mode]} result", final_text, mode)

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

        logger.info(
            "Delivered via %s.",
            getattr(self.output, "name", type(self.output).__name__),
        )
        return final_text

    # ------------------------------------------------------------------ #
    # Logging and debug persistence
    # ------------------------------------------------------------------ #

    def _log_text(self, label: str, text: str, mode: Mode) -> None:
        """Log text at INFO only in debug mode; otherwise log just its size.

        Transcripts are the user's speech. By default they appear nowhere in
        the logs — only the mode and character count, which is enough to tell
        "heard nothing" from "produced nothing" while debugging.
        """
        if not self.debug_enabled:
            logger.info(
                "%s: %d chars (hidden; set debug.enabled=true to show text).",
                label,
                len(text),
            )
            return

        logger.info("%s: %s", label, text)
        self._maybe_save_debug_text(label, text, mode)

    def _maybe_save_debug_audio(
        self, audio: np.ndarray, sample_rate: int, mode: Mode
    ) -> None:
        """Persist a recording only when debug mode is explicitly enabled."""
        if not self.debug_enabled:
            return  # in-memory only: nothing is ever written to disk

        try:
            import soundfile as sf  # deferred: only needed for debug dumps
        except ImportError:
            logger.warning("debug.enabled but soundfile is missing; cannot save recording.")
            return

        try:
            path = self._debug_path(f"recording-{mode.value}", ".wav")
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            logger.debug("Debug recording saved: %s", path)
        except OSError as exc:
            logger.warning("Could not save debug recording: %s", exc)

    def _maybe_save_debug_text(self, label: str, text: str, mode: Mode) -> None:
        """Persist transcript text only when debug mode is explicitly enabled."""
        if not self.debug_enabled:
            return
        slug = label.lower().replace(" ", "-")
        try:
            path = self._debug_path(f"{slug}-{mode.value}", ".txt")
            path.write_text(
                f"mode: {mode.value}\nlabel: {label}\n\n{text}\n", encoding="utf-8"
            )
            logger.debug("Debug text saved: %s", path)
        except OSError as exc:
            logger.warning("Could not save debug text: %s", exc)

    def _debug_path(self, stem: str, suffix: str) -> Path:
        debug_dir = Path(self.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return debug_dir / f"{stem}-{stamp}{suffix}"
