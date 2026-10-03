"""QLoRA fine-tuning of Qwen3-0.6B for the MicDrop prompt-refinement task.

Self-contained: no imports from the micdrop repo, so it runs on Google Colab
or any GPU machine. Uses the MicDrop dataset splits (train/validation.jsonl)
and trains ONLY on the assistant response (the cleaned prompt) via completion
masking.

Design choices
--------------
* QLoRA 4-bit (NF4) base + LoRA r=16 on all linear projections of attention
  and MLP -> fits comfortably on a free Colab T4 (16 GB).
* The system prompt is byte-identical to the runtime system prompt in
  prompt_refiner.py -- keep the two in sync or the adapter trains/deploys on
  mismatched prompts.
* Chat template with enable_thinking=False, exactly like runtime inference.
* fp16 on GPUs without bfloat16 (T4), bf16 elsewhere.

Usage:
    python train_lora.py --train data/train.jsonl --val data/validation.jsonl \
        --out prompt-refiner-lora
    # defaults: --epochs 3 --batch-size 8 --grad-accum 2 --lr 2e-4 --r 16
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_MODEL = "Qwen/Qwen3-0.6B"

# !! Keep byte-identical with prompt_refiner.py in the micdrop repo !!
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


def encode_examples(rows: list[dict], tok, max_len: int) -> list[dict]:
    """Tokenize with loss masked to the assistant response only."""
    encoded: list[dict] = []
    skipped = 0
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
        ]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        full_text = prompt_text + row["output"].strip() + "<|im_end|>\n"

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_len:
            skipped += 1
            continue

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        encoded.append({
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        })
    if skipped:
        print(f"  skipped {skipped} examples longer than {max_len} tokens")
    return encoded


def make_collate(tok):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/validation.jsonl")
    ap.add_argument("--out", default="prompt-refiner-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)          # LoRA rank
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Loading base model in 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.r,
        lora_alpha=2 * args.r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print(f"Encoding {args.train} ...")
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    train_ds = Dataset.from_list(
        encode_examples(train_rows, tok, args.max_len))
    val_ds = Dataset.from_list(
        encode_examples(val_rows, tok, args.max_len))
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=args.out + "-ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16_ok,
        fp16=(torch.cuda.is_available() and not bf16_ok),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate(tok),
    )
    trainer.train()

    print(f"\nSaving adapter to {args.out}/")
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Done. Package it (see training/COLAB.md):")
    print(f"  zip -r {args.out}.zip {args.out}")


if __name__ == "__main__":
    main()
