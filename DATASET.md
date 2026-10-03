# MicDrop dataset

Training/evaluation data for the MicDrop prompt-refinement stage:

```text
messy spoken / Whisper-like transcription  ->  clean, intent-preserving prompt
```

The data trains **Qwen3-0.6B + LoRA/QLoRA** offline. Runtime never trains; it
only loads the saved adapter between Whisper and clipboard/paste.

Record format (metadata is for analysis/filtering only — not part of the
model prompt):

```json
{
  "input": "what is um the difference between malok and kallak in c",
  "output": "What is the difference between malloc and calloc in C?",
  "category": "technical_correction",
  "source": "micdrop_seed",
  "synthetic": true,
  "topic": "coding"
}
```

## Current build (2026-09-25)

* **1,400 examples**: 1,096 train / 152 validation / 152 test
* **2,322 pre-dedup records** → exact-dup and near-dup filtered
* **Zero output overlap** between splits (group-wise split; verified by assertion)
* Sources: micdrop_seed 600, micdrop_real 2, dowis 92, dolly 450,
  codealpaca 500, magicoder 350, oasst1 500 (before dedup/trimming)
* Categories: filler_removal 434, speech_correction 272, repetition 253,
  technical_correction 158, speech_style_normalization 78, programming 66,
  clean_simple 67, numbers_versions 43, multi_part 12, ... (full table in
  `eval_report.md`)

## Layout

```text
micdrop/
├── dataset/
│   ├── raw/                      # downloaded text extracts (no audio, ever)
│   │   ├── dowis_en_prompts.jsonl
│   │   ├── dolly_15k.jsonl
│   │   ├── code_alpaca_20k.jsonl
│   │   ├── magicoder_40k.jsonl
│   │   ├── oasst1_en_prompts.jsonl
│   │   ├── _download.log
│   │   └── micdrop_real.jsonl    # YOUR real Whisper pairs (see below)
│   ├── processed/
│   │   └── all_records.jsonl     # pre-split records for analysis
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── preview.jsonl             # 120 random examples for manual inspection
│   ├── eval_report.json
│   └── eval_report.md
├── data_pipeline/
│   ├── dataset_sources.py        # verified registry: licenses, roles, contracts
│   ├── download_datasets.py      # bandwidth-aware raw extraction
│   ├── inspect_datasets.py       # schema/count/length inspection
│   ├── transform_datasets.py     # unify to input/output, filter, categorize
│   ├── noisy_inputs.py           # realistic Whisper-like noise engine
│   ├── filter_dataset.py         # dedup, quality checks (CLI)
│   ├── build_dataset.py          # splits, leakage assertions, reports
│   ├── preview_dataset.py        # regenerate preview with filters
│   └── seed_data.py              # hand-written MicDrop seed prompts
├── DATASET.md                    # this file
└── DATASET_SOURCES.md            # per-dataset license/size/usage table
```

## Exact commands

All commands run from the repo root using the project venv.

```bash
# 1. install the tooling (once)
./.venv/bin/python -m pip install datasets huggingface_hub pyarrow pandas requests

# 2. download/prepare the public datasets (~35 MB text; no audio)
./.venv/bin/python data_pipeline/download_datasets.py

#    ...or individually:
./.venv/bin/python data_pipeline/download_datasets.py dowis
./.venv/bin/python data_pipeline/download_datasets.py dolly codealpaca
./.venv/bin/python data_pipeline/download_datasets.py oasst1 magicoder

# 3. build the MicDrop dataset (splits + reports + preview)
./.venv/bin/python data_pipeline/build_dataset.py
#    larger build:
./.venv/bin/python data_pipeline/build_dataset.py --target-total 4000

# 4. inspect the dataset / raw extracts
./.venv/bin/python data_pipeline/inspect_datasets.py          # raw extracts
./.venv/bin/python data_pipeline/inspect_datasets.py --live   # + live HF license check

# 5. generate the preview (random 120, or filtered)
./.venv/bin/python data_pipeline/preview_dataset.py
./.venv/bin/python data_pipeline/preview_dataset.py --category technical_correction --n 40
./.venv/bin/python data_pipeline/preview_dataset.py --source dowis

# 6. run dataset quality checks (exit 1 on failure)
./.venv/bin/python data_pipeline/filter_dataset.py
```

## How the data is built

### Sources (see DATASET_SOURCES.md for the full license table)

1. **micdrop_seed** (original, ~120 prompts) — defines the target distribution:
   malloc/calloc, PyTorch, PostgreSQL, Kubernetes, CUDA, LoRA, file paths,
   versions, multi-part requests. Each seed becomes 1 clean pair + up to 3
   noisy variants.
2. **micdrop_real** (you) — real Whisper transcriptions, used verbatim.
   Highest-value data; see "Adding real recordings" below.
3. **DOWIS** (CC-BY-4.0, 92 used) — human-recorded spoken↔written prompts for
   SLLM tasks; contributes genuinely spoken instruction phrasings
   (informal style included). Text-only: the 1.8 GB audio is never downloaded
   (datasets-server /rows API).
4. **dolly** (CC-BY-SA-3.0, ~450) — human-written free-form prompts
   (open_qa/brainstorming/generation only; context-dependent rows filtered).
5. **codealpaca** (CC-BY-4.0, ~500) — short technical prompts.
6. **magicoder** (Apache-2.0, ~350) — deeper technical prompts, 20–200 chars.
7. **oasst1** (Apache-2.0, ~500) — real crowd-sourced user prompts, informal
   phrasing, question-heavy.

From every external dataset only user-side instruction/question **text** is
consumed — never answers — so "output is a prompt, not an answer" is
structural. Each clean text becomes the `output`; the `input` is a synthetic
Whisper-like rendering of it.

### Synthetic noise generation (`noisy_inputs.py`)

Noise operators, applied with per-category probabilities (never randomly
corrupting every word):

| Operator | Example |
|---|---|
| fillers | "uh", "um", "like", "you know", "i mean" inserted at plausible spots |
| repetition | "create a ... create a ...", triple stutters on function words, one "no wait" restart max |
| false starts | "fix the bug uh actually fix the login bug" (self-correction then real intent) |
| conversational frames | "hey can you ...", "... thanks" (frame dropped in output) |
| technical confusion | malloc→malok, calloc→kallak, PyTorch→pie torch, PostgreSQL→post grass sql, Kubernetes→cooper netties, Hugging Face→hugging phase, Qwen→gwen, CUDA→cooda, JSON→dot jason, /→slash |
| spoken numbers | "3.9" → "three point nine", "512 MB" → "five hundred twelve megabytes" |
| merged words | "alot", "toit" (plausible merges only) |
| informal grammar | contractions, lowered first word |

Profiles (`light`, `medium`, `heavy`, `tech_focus`, `repetition_focus`,
`false_start_focus`, `spoken_literal`, `clean_tech`) mix these per category;
`spoken_literal` produces pure ASR-misrecognition utterances ("malok and
kallak in c"). ~10% of examples are **clean or near-clean** (input≈output)
so the model does not learn to always rewrite aggressively.

Noise is word-level and context-driven, NOT a hardcoded dictionary of whole
utterances: the model must use surrounding technical context to recover the
intended term, which is exactly the runtime skill required.

### Categories

`clean_simple`, `filler_removal`, `repetition`, `speech_correction`,
`technical_correction`, `programming`, `research`, `creative`, `questions`,
`multi_part`, `numbers_versions`, `file_paths`, `commands`, `long_requests`,
`mixed_noise`, `speech_style_normalization`. Assigned from the noise
operators that actually fired + content heuristics; `topic` (coding /
systems / ai_ml / data / general) is kept separately for analysis.

### Quality controls

1. exact-duplicate removal (normalized input+output)
2. near-duplicate removal: within an output group (same target prompt),
   variants with Jaccard ≥ 0.85 vs a kept variant are dropped (clean variants
   preferred); across different outputs, pairs ≥ 0.95 are dropped
3. max 4 records per target output (augmentation bound)
4. answer-like outputs rejected ("the answer...", "you should...", ...)
5. malformed records dropped (empty sides, control chars, overlong)
6. follow-ups rejected ("Make it simpler", "Now explain it to a dog" — context-dependent)
7. DOWIS task-code rows ("S2ST English") and markup rows rejected
8. group-wise split: all variants of one target land in ONE split; zero
   normalized-output overlap train/val/test (asserted at build time)
9. questions stay questions (leading wh-word or auxiliary+pronoun → "?");
   imperatives keep "."
10. outputs ≤ 300 chars, inputs ≤ 500 chars (spoken-utterance bound)

Run `filter_dataset.py` any time — it re-validates all three splits and
overlap-freedom, exiting nonzero on failure.

### Evaluation report

`dataset/eval_report.md` / `.json`: totals, per-category, per-source,
synthetic vs real, duplicates removed, length stats (mean/p95), longest and
shortest inputs. Regenerated on every build.

## Adding your own real Whisper transcriptions

This is the highest-leverage improvement available: real errors from YOUR
voice + YOUR microphone + YOUR Whisper model.

1. Use MicDrop normally; keep the raw transcription (the `input`) and write
   the prompt you actually meant (the `output`).
2. Append one JSON object per line to `dataset/raw/micdrop_real.jsonl`:

   ```json
   {"input": "so um create a fast api endpoint that that streams responses from a local qwen model", "output": "Create a FastAPI endpoint that streams responses from a local Qwen model.", "category": "real_recording"}
   ```

   (`category` is optional and defaults to `real_recording`.)
3. Rebuild:

   ```bash
   ./.venv/bin/python data_pipeline/build_dataset.py
   ```

Real records bypass the noise engine (used verbatim), get deduplicated with
everything else, and are split group-wise like all sources. A starter file
with two examples already exists; delete the examples once you add real ones.

## Scaling to 3,000–5,000

```bash
./.venv/bin/python data_pipeline/build_dataset.py --target-total 4000
```

If the raw extracts run dry before reaching the target, raise the quotas in
`transform_datasets.EXTERNAL_QUOTA` (the sources hold far more: dolly ~7.5k
eligible free-form, codealpaca ~20k, magicoder ~25k, oasst1 ~15k rows) and
re-run download+build. Prefer quality: do not raise quotas beyond what the
filters keep.

## Training (separate, offline)

This repo only builds data. For LoRA/QLoRA fine-tuning of Qwen3-0.6B, train
elsewhere (e.g. a machine with a GPU) on `dataset/train.jsonl`, validate on
`dataset/validation.jsonl`, and evaluate on `dataset/test.jsonl` — held out
from training by construction. Format each example as a single-turn task
prompt asking for the cleaned prompt, train on the `output` completion only,
and ship the saved adapter for runtime use.
