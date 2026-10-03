import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen3-0.6B"


def _find_adapter() -> str | None:
    """Locate a fine-tuned LoRA adapter, if one is installed.

    Priority: $MICDROP_ADAPTER, then adapters/prompt-refiner-lora in the
    repo. See training/COLAB.md for how to produce and install one.
    """
    env = os.environ.get("MICDROP_ADAPTER")
    if env and os.path.isdir(env):
        return env
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "adapters", "prompt-refiner-lora")
    if os.path.isdir(default):
        return default
    return None


print("Loading Qwen3 0.6B...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

ADAPTER_PATH = _find_adapter()
if ADAPTER_PATH:
    try:
        from peft import PeftModel

        print(f"Loading fine-tuned adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model.eval()
        print("Adapter loaded.")
    except ImportError:
        print(
            "WARNING: adapter found but peft is not installed; "
            "using the stock base model.\n"
            "Install with: ./.venv/bin/python -m pip install peft"
        )
        ADAPTER_PATH = None
else:
    print("No fine-tuned adapter found; using stock Qwen3-0.6B "
          "(see training/COLAB.md).")

print("Qwen3 loaded.\n")


SYSTEM_PROMPT = """You are a prompt refinement engine.

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


def refine_prompt(raw_text: str) -> str:
    if not raw_text.strip():
        return ""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": raw_text,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # Safety cleanup in case the model still emits reasoning.
    if "<think>" in result:
        result = result.split("<think>", 1)[0].strip()

    if "</think>" in result:
        result = result.split("</think>", 1)[-1].strip()

    return result