"""Local Qwen3-0.6B (+ optional LoRA) text processor, bound to one mode.

This is the existing prompt_refiner.py implementation, refactored behind the
TextProcessor interface. Three changes matter beyond the original:

* It is constructed for a specific mode and refuses the other one, so the
  REFINE prompt and the GENERATE prompt can never be swapped by accident.
* The system prompt is looked up once at construction from
  :mod:`micdrop.refinement.prompts`; there is no runtime prompt selection.
* The model is loaded lazily and *once per model+adapter pair*, then shared, so
  REFINE and GENERATE in one process do not hold two copies of the weights.

The generation budget also comes from the mode's config entry, since GENERATE
has to emit a whole artifact and REFINE must stay near the input length.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading

from ..app.config import ADAPTER_ENV_VAR, AppConfig
from ..app.errors import ProcessingError
from ..app.modes import Mode
from .base import TextProcessor
from .prompts import system_prompt_for

logger = logging.getLogger("micdrop")

# Shared model cache: (model_name, adapter_path) -> (model, tokenizer). Two
# mode-bound processors pointing at the same weights share one load.
_MODEL_CACHE: dict[tuple[str, str | None], tuple[object, object]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Drop cached models. For tests and for freeing memory on shutdown."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def find_adapter(explicit: str | None = None) -> str | None:
    """Locate a fine-tuned LoRA adapter, if one is installed.

    Priority: explicit config value, then $MICRODROP_ADAPTER, then
    adapters/prompt-refiner-lora in the repo. See training/COLAB.md for how
    to produce and install one.
    """
    env = os.environ.get(ADAPTER_ENV_VAR)
    if env and os.path.isdir(env):
        return env
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "adapters", "prompt-refiner-lora",
    )
    default = pathlib.Path(default).resolve()
    if os.path.isdir(default):
        return str(default)
    return None


class LocalQwenProcessor(TextProcessor):
    """Qwen3-0.6B (optionally + LoRA adapter) running locally, for one mode."""

    def __init__(self, config: AppConfig | None = None, mode: Mode = Mode.REFINE) -> None:
        super().__init__(mode)
        config = config or AppConfig()
        # Per-mode model id from [models.local]; no model name is hardcoded here.
        self.model_name = config.local.model_for(self.mode)
        self.max_new_tokens = config.max_new_tokens_for(self.mode)
        self.adapter_path = find_adapter(config.local.adapter_path or config.refinement.adapter_path)
        # Bound at construction: this instance will only ever use this prompt.
        self.system_prompt = system_prompt_for(self.mode)
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        base = f"Qwen3 local ({self.model_name})"
        adapter = " + LoRA" if self.adapter_path else ""
        return f"{base}{adapter} [{self.mode.value}]"

    def warmup(self) -> None:
        self._ensure_loaded()

    def process(self, text: str, mode: Mode) -> str:
        # Mode check happens before any work, so a cross-mode call is a cheap,
        # loud error rather than a slow, wrong one.
        self._check_mode(mode)
        if not text or not text.strip():
            return ""

        self._ensure_loaded()

        try:
            return self._generate(text, self.system_prompt)
        except ProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - model/runtime errors
            raise ProcessingError(f"Qwen3 inference failed: {exc}") from exc

    # -- internals -------------------------------------------------------- #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        cache_key = (self.model_name, self.adapter_path)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is None:
                cached = self._load_model()
                _MODEL_CACHE[cache_key] = cached
        self._model, self._tokenizer = cached

    def _load_model(self):
        """Load base model + optional adapter. Raises ProcessingError on failure."""
        logger.info("Loading %s (mode: %s)...", self.model_name, self.mode.value)
        try:
            import torch  # noqa: F401 - fail fast with a clear message
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                f"Could not load model {self.model_name!r}: {exc}. "
                "Is the model downloaded and are torch/transformers installed?"
            ) from exc

        if self.adapter_path:
            try:
                from peft import PeftModel

                logger.info("Loading fine-tuned adapter: %s", self.adapter_path)
                model = PeftModel.from_pretrained(model, self.adapter_path)
                model.eval()
            except ImportError:
                logger.warning(
                    "Adapter found but peft is not installed; using the stock "
                    "base model. Install with: ./.venv/bin/python -m pip install peft"
                )
                self.adapter_path = None
        else:
            logger.info(
                "No fine-tuned adapter found; using stock %s (see training/COLAB.md).",
                self.model_name,
            )

        logger.info("Model loaded.")
        return (model, tokenizer)

    def _generate(self, text: str, system_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        result = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return strip_reasoning(result)


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks and a whole-output code fence from model output.

    Both prompts forbid reasoning, but small models emit it anyway. Stripping it
    here means the failure mode is "slightly worse text", never "the user's
    text plus a paragraph of model introspection pasted into their editor".
    """
    result = (text or "").strip()
    if "</think>" in result:
        # Keep whatever came after the closing tag: that is the answer.
        result = result.split("</think>", 1)[-1].strip()
    if "<think>" in result:
        result = result.split("<think>", 1)[0].strip()
    if result.startswith("```") and result.endswith("```") and result.count("```") == 2:
        inner = result[3:-3].strip()
        lines = inner.split("\n")
        if len(lines) > 1 and not lines[0].strip():
            inner = "\n".join(lines[1:]).strip()
        result = inner
    return result
