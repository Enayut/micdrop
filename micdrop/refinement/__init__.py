"""Text processing providers and the per-mode factory.

A "provider" is a backend that turns a raw transcription into deliverable text.
Providers are selected *per mode* from ``[models] refine_provider`` /
``generate_provider``, so REFINE and GENERATE can run on different backends::

    [models]
    refine_provider   = "local_qwen"   # local, works offline
    generate_provider = "groq"         # cloud

Adding a provider means:

1. write a class implementing :class:`~micdrop.refinement.base.TextProcessor`,
   bound to a mode, in its own module;
2. add one entry to :data:`PROVIDERS` below.

Nothing in the pipeline, the CLI or the config loader needs to change. See
``test_provider_selection.py::TestCustomProvider`` for a test that registers a
brand-new provider at runtime to prove it.

Providers available out of the box:

===============  =========================================================
local_qwen       Qwen3 (optionally + LoRA) via transformers, offline
groq             Groq cloud API via the official ``groq`` client
none             passthrough; delivers the raw transcription unchanged
===============  =========================================================
"""

from __future__ import annotations

import logging
from typing import Callable

from ..app.config import AppConfig
from ..app.errors import UnknownProviderError
from ..app.modes import Mode, require_mode
from .base import TextProcessor
from .groq import GroqProcessor
from .local_qwen import LocalQwenProcessor
from .passthrough import PassthroughProcessor
from .prompts import (
    GENERATE_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    system_prompt_for,
)
from .router import ModeProcessorRouter

logger = logging.getLogger("micdrop")

__all__ = [
    "TextProcessor",
    "LocalQwenProcessor",
    "GroqProcessor",
    "PassthroughProcessor",
    "ModeProcessorRouter",
    "UnknownProviderError",
    "create_processor",
    "create_router",
    "warmup_application",
    "PROVIDERS",
    "PLANNED_PROVIDERS",
    "SYSTEM_PROMPTS",
    "REFINE_SYSTEM_PROMPT",
    "GENERATE_SYSTEM_PROMPT",
    "system_prompt_for",
]


#: Provider name -> factory building a processor bound to `mode`.
PROVIDERS: dict[str, Callable[[AppConfig, Mode], TextProcessor]] = {
    "local_qwen": lambda config, mode: LocalQwenProcessor(config, mode),
    "groq": lambda config, mode: GroqProcessor(config, mode),
    "none": lambda config, mode: PassthroughProcessor(mode),
}

#: Names recognized but not implemented. Listed so the error message can say
#: what to use instead.
PLANNED_PROVIDERS: tuple[str, ...] = ()

#: One-line descriptions, used by `micdrop models`.
PROVIDER_DESCRIPTIONS = {
    "local_qwen": "Local Qwen3 model (transformers). Works offline.",
    "groq": "Groq cloud API. Needs $MICRODROP_GROQ_API_KEY and a network connection.",
    "none": "No processing; delivers the raw transcription unchanged.",
}


def create_processor(config: AppConfig, mode: Mode) -> TextProcessor:
    """Build the processor configured for a single mode.

    `mode` is required on purpose: there is no such thing as an
    unspecified-provider processor.
    """
    mode = require_mode(mode)
    provider = (config.provider_for(mode) or "").strip().lower()

    if not provider:
        raise UnknownProviderError(
            f"No provider configured for {mode.value.upper()}. Set "
            f"[models] {mode.value}_provider in micdrop.toml."
        )

    factory = PROVIDERS.get(provider)
    if factory is None:
        if provider in PLANNED_PROVIDERS:
            raise UnknownProviderError(
                f"Provider {provider!r} for {mode.value.upper()} is not "
                f"implemented yet. Add a TextProcessor and one PROVIDERS entry. "
                f"For now set [models] {mode.value}_provider = 'local_qwen'."
            )
        raise UnknownProviderError(
            f"Unknown provider {provider!r} for {mode.value.upper()}. "
            f"Known providers: {', '.join(sorted(PROVIDERS))}. "
            f"Run 'micdrop models' for details."
        )

    processor = factory(config, mode)
    if processor.mode is not mode:
        # Defensive: a factory that misbinds would defeat the safety invariant.
        raise UnknownProviderError(
            f"Provider {provider!r} returned a processor bound to "
            f"{processor.mode.value.upper()} for a {mode.value.upper()} request."
        )
    return processor


def create_router(config: AppConfig, modes: list[Mode] | None = None) -> ModeProcessorRouter:
    """Build one processor per mode and return the router that dispatches to them.

    `modes` defaults to both user modes. Building a router without a processor
    for a mode is allowed — that mode's hotkey then reports an error and does
    nothing, rather than borrowing the other mode's processor.
    """
    modes = modes if modes is not None else list(Mode)
    processors: dict[Mode, TextProcessor] = {}
    for mode in modes:
        processors[mode] = create_processor(config, mode)

    router = ModeProcessorRouter(processors)
    for mode, processor in router.describe().items():
        logger.info("Mode %s -> %s", mode.value.upper(), processor)
    return router


def warmup_application(pipeline) -> dict[Mode, str]:
    """Load everything once, at startup, before the listener starts.

    The lifecycle this guarantees::

        micdrop run
          -> load Whisper
          -> load the local model(s) each mode needs
          -> initialise cloud providers (validate key / build client)
          -> start the hotkey listener
          -> wait

    Models are never loaded inside a hotkey cycle, so the first press is as fast
    as every later one.

    Returns a mapping of mode -> failure message for the modes that could not
    be prepared. A mode whose provider is unavailable (say GENERATE points at
    groq with no API key) does **not** stop the app: the other mode keeps
    working, and the broken mode reports the same clear error on every press.
    Only the recognizer is fatal, because nothing works without it.
    """
    pipeline.recognizer.warmup()
    warm = getattr(pipeline.processor, "warmup_individually", None)
    if callable(warm):
        return warm()
    pipeline.processor.warmup()
    return {}
