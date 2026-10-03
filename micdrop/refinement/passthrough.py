"""A do-nothing TextProcessor, useful for tests and quick pipeline trials."""

from __future__ import annotations

from ..app.modes import Mode
from .base import TextProcessor


class PassthroughProcessor(TextProcessor):
    """Returns the transcription unchanged (provider 'none')."""

    @property
    def name(self) -> str:
        return "passthrough (no refinement)"

    def process(self, text: str, mode: Mode) -> str:
        return (text or "").strip()
