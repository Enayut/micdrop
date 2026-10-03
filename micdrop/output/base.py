"""OutputAdapter interface.

The final processed text goes through exactly one OutputAdapter. For Fedora
the implementation copies to the clipboard and pastes at the cursor; other
platforms (or entirely different targets, e.g. a text file) plug in here
without the core pipeline changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OutputAdapter(ABC):
    """Delivers the final text somewhere useful."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name, for logs."""

    @abstractmethod
    def deliver(self, text: str) -> bool:
        """Deliver text. Returns True on success, False on failure.

        Implementations must print/log a clear, actionable error and return
        False instead of raising, so an output problem can never break the
        recording loop.
        """

    def check_ready(self) -> list[str]:
        """Return a list of dependency problems; empty list means ready.

        Default: nothing to check.
        """
        return []
