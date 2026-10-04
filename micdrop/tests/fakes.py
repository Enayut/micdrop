"""Test doubles for the pipeline interfaces.

The processors here are *deterministic stand-ins*, not models. Their job is to
make routing observable: a fake records which mode it was called with and what
it was given, and returns a canned result keyed by mode. The semantic tests
then assert on routing and on how output flows — never on whether the fake is
intelligent.
"""

from __future__ import annotations

import numpy as np

from ..app.errors import (
    ProcessingError,
    RecordingError,
    TranscriptionError,
)
from ..app.modes import Mode
from ..audio.base import AudioRecorder
from ..output.base import OutputAdapter
from ..platform.base import HotkeyManager
from ..refinement.base import TextProcessor
from ..refinement.router import ModeProcessorRouter
from ..transcription.base import SpeechRecognizer


class FakeRecorder(AudioRecorder):
    """Returns canned audio; can be told to fail."""

    def __init__(
        self,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        fail_start: Exception | None = None,
        fail_stop: Exception | None = None,
        max_duration: float = 60.0,
    ) -> None:
        self._audio = (
            audio if audio is not None else np.zeros((1600, 1), dtype="float32")
        )
        self._sample_rate = sample_rate
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.max_duration = max_duration
        self.started = False
        self.stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def elapsed(self) -> float:
        return 0.0

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
    """Returns canned text; can fail or return empty output.

    Passing `errors` yields a per-call failure schedule, which is how
    multi-cycle tests model one failing call followed by a good one.
    """

    def __init__(
        self,
        text: str = "hello world",
        error: Exception | None = None,
        errors: list[Exception | None] | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.errors = list(errors) if errors else None
        self.calls: list[tuple[np.ndarray, int]] = []

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.calls.append((audio, sample_rate))
        if self.errors is not None:
            index = len(self.calls) - 1
            if index < len(self.errors) and self.errors[index] is not None:
                raise self.errors[index]
            return self.text
        if self.error:
            raise self.error
        return self.text


class FakeProcessor(TextProcessor):
    """Mode-bound processor recording its calls; can fail or return a fixed string.

    Bound to one mode exactly like a real provider, so using it with the other
    mode raises rather than silently working.
    """

    def __init__(
        self,
        mode: Mode = Mode.REFINE,
        error: Exception | None = None,
        result: str | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(mode)
        self.error = error
        self.result = result
        self.label = label or f"fake:{mode.value}"
        self.calls: list[tuple[str, Mode]] = []

    @property
    def name(self) -> str:
        return self.label

    def process(self, text: str, mode: Mode) -> str:
        self._check_mode(mode)
        self.calls.append((text, mode))
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return f"[{mode.value}] {text}"


class FakeRouter(ModeProcessorRouter):
    """Router over two FakeProcessors, so both modes work in one pipeline."""

    def __init__(self, error: Exception | None = None, result: str | None = None):
        super().__init__(
            {
                Mode.REFINE: FakeProcessor(Mode.REFINE, error=error, result=result),
                Mode.GENERATE: FakeProcessor(Mode.GENERATE, error=error, result=result),
            }
        )

    def calls_for(self, mode: Mode) -> list[tuple[str, Mode]]:
        """Calls recorded by the fake bound to `mode`."""
        return self.processor_for(mode).calls

    @property
    def modes_called(self) -> list[Mode]:
        """The mode of every call across all modes, in call order."""
        seen = [mode for _, mode in self.processor_for(Mode.REFINE).calls]
        seen += [mode for _, mode in self.processor_for(Mode.GENERATE).calls]
        return seen


class RecordingProcessor(TextProcessor):
    """Mode-bound processor returning a canned answer per input.

    `responses` maps raw input text to output text, so a test can say exactly
    what a given utterance should produce in that mode. `default` covers inputs
    with no explicit response.
    """

    def __init__(
        self,
        mode: Mode,
        responses: dict[str, str] | None = None,
        default: str | None = None,
        label: str | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(mode)
        self.responses = dict(responses or {})
        self.default = default if default is not None else f"<{mode.value} artifact>"
        self.label = label or f"recording:{mode.value}"
        self.error = error
        self.calls: list[tuple[str, Mode]] = []

    @property
    def name(self) -> str:
        return self.label

    def process(self, text: str, mode: Mode) -> str:
        self._check_mode(mode)
        self.calls.append((text, mode))
        if self.error:
            raise self.error
        return self.responses.get(text, self.default)


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


class FakeHotkeyManager(HotkeyManager):
    """Minimal stand-in for LinuxHotkeyManager, so wiring can be tested anywhere."""

    def __init__(self) -> None:
        self.handlers: dict[Mode, tuple[object, object]] = {}
        self.running = False

    def register(self, mode, on_press, on_release) -> None:
        self.handlers[mode] = (on_press, on_release)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    # test helpers ------------------------------------------------------- #

    def press(self, mode: Mode):
        return self.handlers[mode][0](mode)

    def release(self, mode: Mode):
        return self.handlers[mode][1](mode)


def make_processor_pair(
    refine_responses: dict[str, str] | None = None,
    generate_responses: dict[str, str] | None = None,
    refine_error: Exception | None = None,
    generate_error: Exception | None = None,
) -> ModeProcessorRouter:
    """A router with one deterministic processor per mode."""
    return ModeProcessorRouter(
        {
            Mode.REFINE: RecordingProcessor(
                Mode.REFINE, responses=refine_responses, error=refine_error
            ),
            Mode.GENERATE: RecordingProcessor(
                Mode.GENERATE, responses=generate_responses, error=generate_error
            ),
        }
    )


def make_pipeline(**kwargs):
    """Build a Pipeline wired to fakes, with per-component overrides.

    Defaults to a router with one fake per mode, so both modes work without the
    test having to think about processor wiring. `processor` overrides that.
    """
    from ..app.pipeline import Pipeline

    recorder = kwargs.get("recorder", FakeRecorder())

    recognizer = kwargs.get("recognizer")
    if recognizer is None:
        recognizer = FakeRecognizer(kwargs.get("recognizer_text", "hello world"))
    elif isinstance(recognizer, str):
        recognizer = FakeRecognizer(recognizer)

    processor = kwargs.get("processor")
    if processor is None:
        if "processor_error" in kwargs:
            processor = FakeRouter(error=kwargs["processor_error"])
        elif "processor_result" in kwargs:
            processor = FakeRouter(result=kwargs["processor_result"])
        else:
            processor = FakeRouter()

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
        debug_dir=kwargs.get("debug_dir"),
    )
    return pipeline, recorder, recognizer, processor, output


# Common errors reused by failure-path tests.
RECORDER_ERROR = RecordingError("microphone vanished")
TRANSCRIBE_ERROR = TranscriptionError("whisper exploded")
PROCESSOR_ERROR = ProcessingError("qwen exploded")
OUTPUT_ERROR = RuntimeError("clipboard exploded")
