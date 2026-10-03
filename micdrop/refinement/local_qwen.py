"""Local Qwen3-0.6B + LoRA text processor.

This is the existing prompt_refiner.py implementation, refactored behind the
TextProcessor interface. The model is loaded lazily (not at import time) so
that importing this module is cheap and testable.

Two modes:

* REFINE   — the original refinement system prompt: correct transcription
  errors, fix grammar/punctuation, remove filler, strictly preserve meaning,
  never invent information, output only the cleaned text.
* GENERATE — interpret the spoken request and turn it into a useful final
  prompt; may restructure and moderately expand, but must not answer or
  execute the request itself.
"""

from __future__ import annotations

import logging
import os
import pathlib

import torch

from ..app.config import ADAPTER_ENV_VAR, AppConfig
from ..app.errors import ProcessingError
from ..app.modes import Mode
from .base import TextProcessor

logger = logging.getLogger("micdrop")

REFINE_SYSTEM_PROMPT = """You are a prompt refinement engine.

You receive raw speech transcribed by Whisper.

Your task:
1. Correct obvious transcription errors.
2. Fix grammar and punctuation.
3. Preserve the user's exact intent.
4. Make the request clear and concise.
5. Do not answer the user's request.
6. Do not add requirements that the user did not imply.
7. Do not explain your changes.
8. Do not include reasoning.
9. Do not include <think> tags.
10. Output ONLY the final refined prompt.

Your entire response must contain only the refined prompt."""

GENERATE_SYSTEM_PROMPT = """You are a prompt generation engine.

You receive raw speech transcribed by Whisper. The speaker is dictating a
request, task, or question out loud.

Your task:
1. Interpret what the speaker wants.
2. Transform the raw speech into one clear, well-structured,
   self-contained prompt.
3. You may reorganize the wording, clarify the request, and moderately
   expand it with standard, obviously-implied detail.
4. Correct obvious transcription errors.
5. Remove filler words and repetitions.
6. Do not answer the request yourself. Do not execute it.
7. Do not add requirements or constraints the speaker did not imply.
8. Do not explain your changes. Do not include reasoning.
9. Do not include <think> tags.
10. Output ONLY the final generated prompt.

Your entire response must contain only the generated prompt."""

SYSTEM_PROMPTS = {
    Mode.REFINE: REFINE_SYSTEM_PROMPT,
    Mode.GENERATE: GENERATE_SYSTEM_PROMPT,
}


def find_adapter(explicit: str | None = None) -> str | None:
    """Locate a fine-tuned LoRA adapter, if one is installed.

    Priority: explicit config value, then $MICDROP_ADAPTER, then
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
    """Qwen3-0.6B (optionally + LoRA adapter) running locally."""

    def __init__(self, config: AppConfig | None = None) -> None:
        config = config or AppConfig()
        self.model_name = config.refinement.model_path
        self.max_new_tokens = config.refinement.max_new_tokens
        self.adapter_path = find_adapter(config.refinement.adapter_path)
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        base = f"Qwen3 local ({self.model_name})"
        return f"{base} + LoRA" if self.adapter_path else base

    def warmup(self) -> None:
        self._ensure_loaded()

    def process(self, text: str, mode: Mode) -> str:
        if not text or not text.strip():
            return ""

        self._ensure_loaded()
        system_prompt = SYSTEM_PROMPTS[mode]

        try:
            return self._generate(text, system_prompt)
        except ProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - model/runtime errors
            raise ProcessingError(f"Qwen3 inference failed: {exc}") from exc

    # -- internals -------------------------------------------------------- #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading %s...", self.model_name)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception as exc:  # noqa: BLE001
            raise ProcessingError(
                f"Could not load model {self.model_name!r}: {exc}"
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

        self._model = model
        logger.info("Model loaded.")

    def _generate(self, text: str, system_prompt: str) -> str:
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

        # Safety cleanup in case the model still emits reasoning.
        if "<think>" in result:
            result = result.split("<think>", 1)[0].strip()
        if "</think>" in result:
            result = result.split("</think>", 1)[-1].strip()

        return result
