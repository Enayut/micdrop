"""Test doubles for the pipeline interfaces."""

from __future__ import annotations

import numpy as np

from ..app.errors import ProcessingError, RecordingError, TranscriptionError
from ..app.modes import Mode
from ..audio.base import AudioRecorder
from ..output.base import OutputAdapter
from ..refinement.base import TextProcessor
from ..transcription.base import SpeechRecognizer


class FakeRecorder(AudioRecorder):
    """Returns canned audio; can be told to fail."""

    def __init__(
        self,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        fail_start: Exception | None = None,
        fail_stop: Exception | None = None,
    ) -> None:
        self._audio = (
            audio if audio is not None else np.zeros((1600, 1), dtype="float32")
        )
        self._sample_rate = sample_rate
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.started = False
        self.stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_recording(self) -> bool:
        return self.started and not self.stopped

    def start(self) -> None:
        if self.fail_start:
            raise self.fail_start
        self.started = True
        self.stopped = False

    def stop(self) -> np.ndarray:
        if self.fail_stop:
            raise self.fail_stop
        self.stopped = True
        return self._audio


class FakeRecognizer(SpeechRecognizer):
    """Returns canned text; can fail or return empty output."""

    def __init__(
        self,
        text: str = "hello world",
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[np.ndarray, int]] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.calls.append((audio, sample_rate))
        if self.error:
            raise self.error
        return self.text


class FakeProcessor(TextProcessor):
    """Records (text, mode) calls; can fail or transform text."""

    def __init__(self, error: Exception | None = None, result: str | None = None) -> None:
        self.error = error
        self.result = result
        self.calls: list[tuple[str, Mode]] = []

    @property
    def name(self) -> str:
        return "fake"

    def process(self, text: str, mode: Mode) -> str:
        self.calls.append((text, mode))
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return f"[{mode.value}] {text}"


class FakeOutput(OutputAdapter):
    """Records delivered text; can fail."""

    def __init__(self, fail: bool = False, raise_error: Exception | None = None) -> None:
        self.fail = fail
        self.raise_error = raise_error
        self.delivered: list[str] = []

    @property
    def name(self) -> str:
        return "fake output"

    def deliver(self, text: str) -> bool:
        self.delivered.append(text)
        if self.raise_error:
            raise self.raise_error
        if self.fail:
            self.fail = False  # fail once, then recover
            return False
        return True


def make_pipeline(**kwargs):
    """Build a Pipeline wired to fakes, with per-component overrides."""
    from ..app.pipeline import Pipeline

    recorder = kwargs.get("recorder", FakeRecorder())
    recognizer = kwargs.get(
        "recognizer",
        kwargs.get("recognizer_text") is not None
        and FakeRecognizer(kwargs["recognizer_text"])
        or FakeRecognizer(),
    )
    if isinstance(recognizer, str):
        recognizer = FakeRecognizer(recognizer)
    processor = kwargs.get(
        "processor",
        kwargs.get("processor_error") is not None
        and FakeProcessor(error=kwargs["processor_error"])
        or FakeProcessor(),
    )
    output = kwargs.get(
        "output",
        FakeOutput(
            fail=kwargs.get("output_fail", False),
            raise_error=kwargs.get("output_error"),
        ),
    )
    pipeline = Pipeline(
        recorder=recorder,
        recognizer=recognizer,
        processor=processor,
        output=output,
        config=kwargs.get("config"),
    )
    return pipeline, recorder, recognizer, processor, output


# Common errors reused by failure-path tests.
RECORDER_ERROR = RecordingError("microphone vanished")
TRANSCRIBE_ERROR = TranscriptionError("whisper exploded")
PROCESSOR_ERROR = ProcessingError("qwen exploded")
OUTPUT_ERROR = RuntimeError("clipboard exploded")
