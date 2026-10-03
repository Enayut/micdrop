"""Evaluate a MicDrop LoRA adapter on dataset/test.jsonl.

Reports exact-match, token-F1, and category breakdowns. Designed to run on
Colab right after training (adapter stays quantized; eval on GPU is fast).

Usage:
    python evaluate_lora.py --adapter prompt-refiner-lora --test data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"

# !! Keep byte-identical with prompt_refiner.py and train_lora.py !!
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


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def token_f1(pred: str, gold: str) -> float:
    p = pred.lower().split()
    g = gold.lower().split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate(model, tok, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                      skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="prompt-refiner-lora")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="0 = all")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("Loading base model (4-bit) + adapter...")
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = load_jsonl(args.test)
    if args.max_examples:
        rows = rows[:args.max_examples]

    per_cat: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])
    shown = 0
    for i, row in enumerate(rows):
        pred = generate(model, tok, row["input"])
        gold = row["output"].strip()
        em = float(pred == gold)
        f1 = token_f1(pred, gold)
        agg = per_cat[row.get("category", "unknown")]
        agg[0] += 1
        agg[1] += em
        agg[2] += f1
        if shown < args.show and (em == 0 or i % 20 == 0):
            print(f"\n--- [{row.get('category')}] input: {row['input'][:90]}")
            print(f"    gold: {gold[:110]}")
            print(f"    pred: {pred[:110]}  (EM={em}, F1={f1:.2f})")
            shown += 1

    n_all = sum(v[0] for v in per_cat.values())
    em_all = sum(v[1] for v in per_cat.values()) / max(1, n_all)
    f1_all = sum(v[2] for v in per_cat.values()) / max(1, n_all)
    print(f"\n===== RESULTS ({n_all} examples) =====")
    print(f"exact match : {em_all:.3f}")
    print(f"token F1    : {f1_all:.3f}")
    print(f"\nper category (n, EM, F1):")
    for cat, (n, em, f1) in sorted(per_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"  {cat:<30} n={n:<4} EM={em / n:.3f}  F1={f1 / n:.3f}")


if __name__ == "__main__":
    main()
