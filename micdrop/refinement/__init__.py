"""Text processing (prompt refinement / generation) providers."""

from __future__ import annotations

from ..app.config import AppConfig
from ..app.errors import MicDropError
from .base import TextProcessor
from .local_qwen import LocalQwenProcessor

__all__ = ["TextProcessor", "LocalQwenProcessor", "create_processor"]


class UnknownProviderError(MicDropError):
    """The configured refinement provider does not exist or is not enabled."""


def create_processor(config: AppConfig) -> TextProcessor:
    """Instantiate the configured TextProcessor implementation.

    New providers (e.g. Groq) plug in here with one elif branch; the core
    pipeline never needs to change.
    """
    provider = (config.refinement.provider or "").strip().lower()

    if provider == "local_qwen":
        return LocalQwenProcessor(config)

    if provider == "none":
        from .passthrough import PassthroughProcessor

        return PassthroughProcessor()

    if provider == "groq":
        raise UnknownProviderError(
            "Provider 'groq' is not implemented yet. Its configuration "
            "placeholder lives in the [groq] section; set "
            "refinement.provider = 'local_qwen' to use the local model."
        )

    raise UnknownProviderError(
        f"Unknown refinement provider {provider!r}. "
        "Known providers: local_qwen, none (groq planned)."
    )
