"""Groq cloud TextProcessor.

Implements the same :class:`~micdrop.refinement.base.TextProcessor` interface as
the local model, so it can back either mode:

    [models]
    refine_provider   = "local_qwen"   # local, offline
    generate_provider = "groq"          # cloud

Nothing in the pipeline, the CLI or the config loader knows this module exists;
they only see ``TextProcessor``. That is the whole point of the provider
architecture, and it is why a cloud backend is a config edit rather than a code
change.

Design rules this file follows:

* **The API key is never in code, never in a config file, never logged.** It is
  read from ``$MICRODROP_GROQ_API_KEY`` at construction time and passed to the
  client; the string itself is never included in any log message or exception
  text.
* **A missing key is a clear, actionable failure**, not a silent fallback to a
  local model. If you configure Groq, you get Groq or you get an error.
* **Mode-bound like every provider.** Constructed for REFINE or GENERATE and
  refuses the other one.
* **No network at import time.** The `groq` client is imported lazily so
  importing MicDrop stays cheap and works offline.
* **Offline-safe.** When both modes use local providers this module is never
  touched and nothing reaches the network.

Uses the official `groq` Python client rather than hand-rolled HTTP, so retries,
auth and error typing come from the library.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..app.config import GROQ_API_KEY_ENV_VAR, AppConfig
from ..app.errors import ProcessingError
from ..app.modes import Mode
from .base import TextProcessor
from .local_qwen import strip_reasoning
from .prompts import system_prompt_for

logger = logging.getLogger("micdrop")


class GroqProcessor(TextProcessor):
    """Text processing via the Groq chat completions API, for one mode."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        mode: Mode = Mode.GENERATE,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        super().__init__(mode)
        config = config or AppConfig()
        self.settings = config.groq
        self.model_name = self.settings.model_for(self.mode)
        self.max_new_tokens = config.max_new_tokens_for(self.mode)
        self.system_prompt = system_prompt_for(self.mode)

        # Store only whether a key exists, never the key itself, so an
        # accidental repr()/traceback cannot leak it.
        self._has_key = bool(api_key) or bool(config.groq_api_key())
        self._api_key = api_key or config.groq_api_key()
        self._client = client  # injected in tests; None means build lazily

    @property
    def name(self) -> str:
        return f"groq ({self.model_name}) [{self.mode.value}]"

    def warmup(self) -> None:
        """Validate credentials and build the client.

        Called once at startup so a missing key or missing package is reported
        before the user ever presses a hotkey, rather than on first use.
        """
        self._ensure_client()

    def process(self, text: str, mode: Mode) -> str:
        self._check_mode(mode)
        if not text or not text.strip():
            return ""

        client = self._ensure_client()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.settings.temperature,
                max_completion_tokens=self.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as ProcessingError
            raise ProcessingError(self._describe_error(exc)) from exc

        return strip_reasoning(_extract_text(completion))

    # -- internals -------------------------------------------------------- #

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        if not self._has_key:
            raise ProcessingError(
                f"The groq provider is configured for {self.mode.value.upper()} "
                f"but no API key is set. Export it before starting MicDrop:\n"
                f"    export {self.settings.api_key_env}=gsk_...\n"
                f"(the key is never read from the config file, and never logged). "
                f"To go back to a local model, set "
                f"[models] {self.mode.value}_provider = \"local_qwen\"."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ProcessingError(
                "The 'groq' package is required for the groq provider. "
                "Install it with: pip install groq  "
                "(or use the local provider: "
                f'[models] {self.mode.value}_provider = "local_qwen").'
            ) from exc

        try:
            self._client = Groq(
                api_key=self._api_key,
                timeout=self.settings.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(f"Could not create the Groq client: {exc}") from exc

        logger.info("Groq client ready (mode: %s, model: %s)", self.mode.value, self.model_name)
        return self._client

    def _describe_error(self, exc: Exception) -> str:
        """Turn a client exception into an actionable message.

        The API key is never part of these messages. Auth and connectivity
        errors are called out separately because they have completely different
        fixes.
        """
        name = type(exc).__name__

        if name == "AuthenticationError":
            return (
                f"Groq rejected the API key ({self.settings.api_key_env}). "
                "Check that the key is correct and still active."
            )
        if name in ("APIConnectionError", "APITimeoutError"):
            return (
                f"Could not reach the Groq API (network unavailable or timed out "
                f"after {self.settings.timeout:.0f}s). MicDrop is not switching "
                f"providers; use a local model if you need to work offline."
            )
        if name == "RateLimitError":
            return "Groq rate limit reached. Try again shortly."
        if name == "NotFoundError":
            return (
                f"Groq does not know the model {self.model_name!r}. "
                f"Set a valid id under [models.groq] "
                f"{self.mode.value}_model."
            )

        # Strip anything that looks like a key out of third-party message text
        # before it can reach a log line.
        detail = _redact(str(exc), self._api_key)
        return f"Groq request failed ({name}): {detail}"


def _extract_text(completion: Any) -> str:
    """Pull the assistant text out of a chat completion response."""
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ProcessingError(f"Unexpected Groq response shape: {exc}") from exc
    return (content or "").strip()


def _redact(text: str, api_key: Optional[str]) -> str:
    """Remove the API key from a message, plus anything that looks like one."""
    if api_key:
        text = text.replace(api_key, "***redacted***")
    # Belt and braces: a truncated key can appear in an upstream error string.
    import re

    return re.sub(r"gsk_[A-Za-z0-9_\-]{8,}", "***redacted***", text)


__all__ = ["GroqProcessor", "GROQ_API_KEY_ENV_VAR"]