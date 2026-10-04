"""ModeProcessorRouter — the single place mode -> processor resolution happens.

The pipeline holds one of these. It looks up the processor bound to the mode it
was given and calls it. It never guesses, never falls back to the other mode's
processor, and never reuses a processor bound to a different mode.

This is the enforcement point for the safety invariant:

    REFINE must never silently fall back to GENERATE, and GENERATE must never
    be handled with the REFINE prompt.

Two independent guards implement it:

1. The router only dispatches to a processor whose ``.mode`` equals the
   requested mode. A mismatch raises instead of calling it.
2. Each processor re-checks the mode itself (see ``TextProcessor._check_mode``),
   so the invariant holds even for a processor used directly.

A missing processor for a mode is a hard failure for that hotkey cycle: the
cycle is logged and skipped, and the listener stays up for the next one. That
is deliberate — a wrong paste is worse than no paste.
"""

from __future__ import annotations

import logging

from ..app.errors import ModeProcessorUnavailableError
from ..app.modes import DESCRIPTIONS, Mode, require_mode
from .base import TextProcessor

logger = logging.getLogger("micdrop")


class ModeProcessorRouter(TextProcessor):
    """Dispatches to the processor bound to each mode."""

    def __init__(self, processors: dict[Mode, TextProcessor]) -> None:
        # Bound to no mode of its own: it is a dispatcher, not a processor for
        # one mode, so it overrides `mode` rather than pretending.
        self._mode = None  # type: ignore[assignment]
        self._processors: dict[Mode, TextProcessor] = {}
        self._failures: dict[Mode, str] = {}

        for mode, processor in processors.items():
            require_mode(mode)
            bound = getattr(processor, "mode", None)
            if bound is not mode:
                raise ModeProcessorUnavailableError(
                    f"Processor {processor!r} is bound to {bound!r} but was "
                    f"registered for {mode.value.upper()}. Refusing to wire a "
                    f"processor to the wrong mode."
                )
            self._processors[mode] = processor

    # -- routing ----------------------------------------------------------- #

    def supports(self, mode: Mode) -> bool:
        """Whether a processor is configured for `mode`."""
        return mode in self._processors

    def processor_for(self, mode: Mode) -> TextProcessor:
        """The processor bound to `mode`, or raise ModeProcessorUnavailableError."""
        require_mode(mode)
        try:
            processor = self._processors[mode]
        except KeyError:
            configured = ", ".join(
                m.value.upper() for m in sorted(self._processors, key=lambda m: m.value)
            ) or "none"
            raise ModeProcessorUnavailableError(
                f"No processor is configured for {mode.value.upper()} "
                f"({DESCRIPTIONS[mode]}). Configured modes: {configured}. "
                f"Set models.{mode.value}_provider in micdrop.toml. Not "
                f"falling back to another mode's processor."
            ) from None

        reason = self._failures.get(mode)
        if reason is not None:
            raise ModeProcessorUnavailableError(
                f"{mode.value.upper()} is configured but unavailable: {reason} "
                f"Not falling back to another provider."
            )

        if processor.mode is not mode:
            raise ModeProcessorUnavailableError(
                f"Processor for {mode.value.upper()} is bound to "
                f"{processor.mode.value.upper()}; refusing to dispatch."
            )
        return processor

    # -- TextProcessor interface ------------------------------------------- #

    @property
    def mode(self) -> Mode:  # type: ignore[override]
        raise ModeProcessorUnavailableError(
            "The router is not bound to a single mode; ask processor_for(mode)."
        )

    @property
    def name(self) -> str:
        parts = [
            f"{mode.value}={self._processors[mode].name}"
            for mode in sorted(self._processors, key=lambda m: m.value)
        ]
        return "router(" + ", ".join(parts) + ")"

    def process(self, text: str, mode: Mode) -> str:
        return self.processor_for(mode).process(text, mode)

    def warmup(self) -> None:
        """Warm every mode; raise on the first failure (all-or-nothing)."""
        failures = self.warmup_individually()
        if failures:
            mode, message = sorted(failures.items(), key=lambda kv: kv[0].value)[0]
            raise ModeProcessorUnavailableError(
                f"{mode.value.upper()} could not be prepared: {message}"
            )

    def warmup_individually(self) -> dict[Mode, str]:
        """Warm each mode separately, returning mode -> failure message.

        A misconfigured mode must not take down a working one. If GENERATE
        points at a cloud provider with no API key, REFINE stays fully usable
        and GENERATE reports the same clear error on every press — which is
        also what keeps it from ever silently substituting another provider.

        The failed mode is *disabled* rather than left half-built, so a
        later press cannot reach a processor that failed to initialise.
        """
        failures: dict[Mode, str] = {}
        for mode, processor in self._processors.items():
            logger.info("Warming up %s: %s", mode.value.upper(), processor.name)
            try:
                processor.warmup()
            except Exception as exc:  # noqa: BLE001 - one mode must not kill the rest
                logger.error(
                    "%s is unavailable and will fail on every press: %s",
                    mode.value.upper(),
                    exc,
                )
                self._failures[mode] = str(exc)
                failures[mode] = str(exc)
            else:
                self._failures.pop(mode, None)
        return failures

    @property
    def failures(self) -> dict[Mode, str]:
        """Modes that failed to prepare, mapped to their reason."""
        return dict(self._failures)

    def describe(self) -> dict[Mode, str]:
        """mode -> processor name, for startup logging and diagnostics."""
        return {mode: processor.name for mode, processor in self._processors.items()}
