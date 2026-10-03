"""Whisper speech-to-text via the transformers pipeline.

This is the transcription code from the old main.py/transcribe.py, wrapped
in the SpeechRecognizer interface. The model is loaded lazily (not at
import time) and from the local Hugging Face cache when available.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..app.config import RefinementSettings
from ..app.errors import TranscriptionError
from .base import SpeechRecognizer

logger = logging.getLogger("micdrop")


class WhisperRecognizer(SpeechRecognizer):
    """Automatic speech recognition with openai/whisper-* models."""

    def __init__(self, settings: Optional[RefinementSettings] = None) -> None:
        settings = settings or RefinementSettings()
        self.model_name = settings.whisper_model
        self.language = settings.language
        self._pipeline = None  # lazy: loaded on first use / warmup()

    # -- SpeechRecognizer interface -------------------------------------- #

    def warmup(self) -> None:
        self._ensure_loaded()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio is None or len(audio) == 0:
            return ""

        self._ensure_loaded()

        # Accept (samples,) or (samples, channels); the ASR pipeline wants
        # 1-D mono samples.
        samples = np.asarray(audio)
        if samples.ndim > 1:
            samples = samples.reshape(-1)

        try:
            result = self._pipeline(
                {"array": samples, "sampling_rate": sample_rate},
                generate_kwargs={
                    "language": self.language,
                    "task": "transcribe",
                },
            )
        except Exception as exc:  # noqa: BLE001 - model/runtime errors
            raise TranscriptionError(f"Whisper inference failed: {exc}") from exc

        return (result.get("text") or "").strip()

    # -- helpers ---------------------------------------------------------- #

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        logger.info("Loading Whisper model %s...", self.model_name)
        try:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                f"Could not load Whisper model {self.model_name!r}: {exc}"
            ) from exc
        logger.info("Whisper loaded.")

    def transcribe_file(self, path: str) -> str:
        """Convenience for CLI tools: transcribe a WAV/FLAC file."""
        self._ensure_loaded()
        try:
            result = self._pipeline(
                path,
                generate_kwargs={
                    "language": self.language,
                    "task": "transcribe",
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Whisper inference failed: {exc}") from exc
        return (result.get("text") or "").strip()
