"""A do-nothing TextProcessor, useful for tests and quick pipeline trials.

Provider name 'none'. It is still mode-bound: configuring `none` for one mode
does not make that mode use the other mode's processor, it just means that mode
delivers the raw transcription unchanged.
"""

from __future__ import annotations

from ..app.modes import Mode
from .base import TextProcessor


class PassthroughProcessor(TextProcessor):
    """Returns the transcription unchanged (provider 'none'), for one mode."""

    def __init__(self, mode: Mode = Mode.REFINE) -> None:
        super().__init__(mode)

    @property
    def name(self) -> str:
        return f"passthrough (no model) [{self.mode.value}]"

    def process(self, text: str, mode: Mode) -> str:
        self._check_mode(mode)
        return (text or "").strip()
