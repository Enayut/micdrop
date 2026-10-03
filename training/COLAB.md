# Fine-tuning Qwen3-0.6B on Google Colab (QLoRA)

End-to-end: upload the MicDrop dataset, train a QLoRA adapter, evaluate it,
download it, and drop it into the micdrop repo. A free Colab T4 is
sufficient (~30–60 min).

The two scripts referenced here live in `training/` in this repo:
`train_lora.py` and `evaluate_lora.py`.

## 1. Open Colab

Create a notebook, then **Runtime → Change runtime type → T4 GPU**.
(Qwen3-0.6B QLoRA needs ~6–8 GB VRAM; the T4's 16 GB is plenty.)

## 2. Install dependencies

```python
!pip -q install "transformers>=4.51" datasets peft bitsandbytes accelerate
```

## 3. Upload the dataset + scripts

Option A — small upload (recommended, ~1 MB):

```python
from google.colab import files
import zipfile, os, glob

# zip the splits locally first (see local commands at the bottom),
# then upload micdrop_dataset.zip here:
up = files.upload()                      # choose micdrop_dataset.zip
with zipfile.ZipFile("micdrop_dataset.zip") as z:
    z.extractall("data")

for f in ["train_lora.py", "evaluate_lora.py"]:
    files.upload()                       # upload each from training/
```

Option B — pull straight from GitHub if you push the repo:

```python
!git clone https://github.com/<you>/micdrop.git
%cd micdrop
```

## 4. Sanity-check the data

```python
import json
rows = [json.loads(l) for l in open("data/train.jsonl")]
print(len(rows), rows[0])
```

Expect ~1,095 rows and records shaped like
`{"input", "output", "category", "source", "synthetic", "topic"}`.

## 5. Train

Defaults: 3 epochs, effective batch 16 (8 × 2 accum), LoRA r=16, lr 2e-4,
completion-only loss. Takes roughly 30–60 min on the T4.

```python
!python train_lora.py --train data/train.jsonl --val data/validation.jsonl \
    --out prompt-refiner-lora
```

Useful knobs:
- Slower but better: `--epochs 4`
- Overfitting (eval loss rising after epoch 2)? Keep the epoch-2 checkpoint
  or rerun with `--epochs 2`.
- OOM (unlikely on T4): add `--batch-size 4 --grad-accum 4`.

## 6. Evaluate on the held-out test set

```python
!python evaluate_lora.py --adapter prompt-refiner-lora \
    --test data/test.jsonl --show 15
```

Read the per-category table: `technical_correction` and
`speech_style_normalization` rows are the ones that matter most for
MicDrop. Spot-check the printed predictions — they must clean, not answer.

## 7. Download the adapter

```python
import shutil
shutil.make_archive("prompt-refiner-lora", "zip", "prompt-refiner-lora")
from google.colab import files
files.download("prompt-refiner-lora.zip")
```

## 8. Install it into the micdrop repo

Unzip on your machine, then:

```bash
mkdir -p ~/micdrop/adapters
unzip prompt-refiner-lora.zip -d ~/micdrop/adapters/
# results in ~/micdrop/adapters/prompt-refiner-lora/
```

`prompt_refiner.py` in this repo already auto-detects an adapter there —
no code changes needed. If the adapter lives elsewhere, set:

```bash
export MICDROP_ADAPTER=/path/to/prompt-refiner-lora
```

## 9. Verify locally

```bash
./.venv/bin/python -c "from prompt_refiner import refine_prompt; print(refine_prompt('what is um the difference between malok and kallak in c'))"
# -> What is the difference between malloc and calloc in C?
```

Then run the real pipeline: `python main.py`, hold Right Ctrl, speak.

## Local packaging commands (step 3, Option A)

```bash
cd /path/to/micdrop
zip -r micdrop_dataset.zip dataset/train.jsonl dataset/validation.jsonl dataset/test.jsonl
```

## Notes

- `test.jsonl` is never used in training; evaluate only at the end.
- If you add real recordings to `dataset/raw/micdrop_real.jsonl` and rebuild
  the dataset, retrain the adapter on the new build and re-evaluate.
- Keep the SYSTEM_PROMPT byte-identical across `prompt_refiner.py`,
  `train_lora.py`, and `evaluate_lora.py`.
