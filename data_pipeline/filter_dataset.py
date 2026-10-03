"""Filter, deduplicate, and quality-check MicDrop records.

Two modes:
  * Library mode (used by build_dataset.py): clean_records() applies
      1. exact-duplicate removal (identical input+output)
      2. malformed-record removal (empty sides, too long, control chars)
      3. answer-like-output removal (outputs must be prompts, not answers)
      4. near-duplicate removal (token-set Jaccard on normalized text)
      5. target-collision collapsing (same normalized output -> keep one)
    and returns (kept_records, stats_dict).

  * CLI mode: run quality checks over built jsonl files and print a report,
    exit code 1 if any hard check fails.

Quality rules (project requirement #1-#11):
  - no duplicate / near-duplicate examples
  - no answer-like outputs (the model must not learn to answer)
  - technical terms, numbers, file paths must survive into the output
    (checked by build-time assertions, not by filtering)
  - outputs stay concise (<= MAX_OUTPUT_CHARS)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

MAX_INPUT_CHARS = 500    # spoken inputs: long dictation + fillers still fit
MAX_OUTPUT_CHARS = 300
NEAR_DUP_SAME_TARGET = 0.85   # same output group: inputs this similar = redundant variant
NEAR_DUP_CROSS_TARGET = 0.95  # different outputs: pair this similar = true duplicate
MAX_PER_OUTPUT_GROUP = 4      # max records sharing one normalized output
ANSWER_STARTS = (
    "the answer", "you should", "you can", "you need", "here is", "here's",
    "sure", "certainly", "as an ai", "sorry", "unfortunately", "i think",
    "yes", "no,", "first,", "second,", "in summary", "to summarize",
)


def normalize_for_dedup(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_set(text: str) -> set[str]:
    return set(normalize_for_dedup(text).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def clean_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate while preserving intentional augmentation.

    Records sharing the same normalized output are augmentation variants of
    one clean prompt (e.g. clean + several noisy renderings). Within such a
    group we drop only inputs that are near-identical to an already-kept
    member, preferring clean (non-synthetic) variants, which are sorted
    first. Records with DIFFERENT outputs are compared as whole pairs at a
    higher threshold so genuinely distinct-but-similar prompts survive.
    """
    stats: Counter = Counter()

    # validity + exact-dup pass
    valid: list[dict] = []
    seen_exact: set[tuple[str, str]] = set()
    for rec in records:
        stats["total"] += 1
        inp = rec.get("input", "").strip()
        outp = rec.get("output", "").strip()
        if not inp or not outp:
            stats["dropped_empty_side"] += 1
            continue
        if len(inp) > MAX_INPUT_CHARS or len(outp) > MAX_OUTPUT_CHARS:
            stats["dropped_too_long"] += 1
            continue
        if any(ord(c) < 9 or ord(c) == 127 for c in inp + outp):
            stats["dropped_control_chars"] += 1
            continue
        if outp.lower().startswith(ANSWER_STARTS):
            stats["dropped_answer_like"] += 1
            continue
        exact_key = (normalize_for_dedup(inp), normalize_for_dedup(outp))
        if exact_key in seen_exact:
            stats["dropped_exact_duplicate"] += 1
            continue
        seen_exact.add(exact_key)
        valid.append(rec)

    # sort so that within each output group clean variants come first and are
    # preferred by the near-dup filter
    def sort_key(rec: dict) -> tuple[str, int]:
        return (normalize_for_dedup(rec["output"]),
                0 if not rec.get("synthetic", True) else 1)

    valid.sort(key=sort_key)

    out: list[dict] = []
    group_inputs: dict[str, list[set[str]]] = {}
    pair_tokens: list[tuple[str, set[str]]] = []   # (output_norm, input|output tokens)
    group_sizes: Counter = Counter()

    for rec in valid:
        inp_toks = token_set(rec["input"])
        out_norm = normalize_for_dedup(rec["output"])

        # 1. within-group: input too similar to an already-kept variant
        prev_inputs = group_inputs.get(out_norm, [])
        if any(jaccard(inp_toks, prev) >= NEAR_DUP_SAME_TARGET for prev in prev_inputs):
            stats["dropped_near_duplicate_variant"] += 1
            continue

        # 2. cross-group: whole pair nearly identical to a kept record with a
        #    different output
        toks = inp_toks | token_set(rec["output"])
        if any(prev_out != out_norm and jaccard(toks, prev_toks) >= NEAR_DUP_CROSS_TARGET
               for prev_out, prev_toks in pair_tokens[-4000:]):
            stats["dropped_near_duplicate"] += 1
            continue

        # 3. per-output cap (augmentation should not exceed a sane bound)
        if group_sizes[out_norm] >= MAX_PER_OUTPUT_GROUP:
            stats["dropped_target_collision"] += 1
            continue

        group_inputs.setdefault(out_norm, []).append(inp_toks)
        pair_tokens.append((out_norm, toks))
        group_sizes[out_norm] += 1
        out.append(rec)
        stats["kept"] += 1

    stats["final"] = len(out)
    return out, dict(stats)


# ---------------------------------------------------------------------------
# Built-dataset checks (CLI)
# ---------------------------------------------------------------------------

def check_split_overlap(split_files: dict[str, Path]) -> bool:
    """Hard check: no output string may appear in more than one split."""
    outputs: dict[str, set[str]] = {}
    for split, path in split_files.items():
        with path.open(encoding="utf-8") as f:
            outputs[split] = {normalize_for_dedup(json.loads(l)["output"])
                              for l in f if l.strip()}
    all_norm = {}
    leaked = 0
    for split, outs in outputs.items():
        for o in outs:
            if o in all_norm and all_norm[o] != split:
                leaked += 1
            all_norm[o] = split
    return leaked == 0


def run_checks() -> bool:
    files = {
        "train": DATASET_DIR / "train.jsonl",
        "validation": DATASET_DIR / "validation.jsonl",
        "test": DATASET_DIR / "test.jsonl",
    }
    ok = True
    for split, path in files.items():
        if not path.exists():
            print(f"  [FAIL] {path.name} missing")
            ok = False
            continue
        bad = 0
        n = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            if (not rec.get("input", "").strip() or not rec.get("output", "").strip()
                    or len(rec["output"]) > MAX_OUTPUT_CHARS
                    or rec["output"].lower().startswith(ANSWER_STARTS)):
                bad += 1
        status = "OK" if bad == 0 else "FAIL"
        print(f"  [{status}] {split}: {n} rows, {bad} malformed/answer-like")
        ok = ok and bad == 0
    if not check_split_overlap(files):
        print("  [FAIL] train/validation/test output overlap detected")
        ok = False
    else:
        print("  [OK]   no output overlap between train/validation/test")
    return ok


if __name__ == "__main__":
    print("Dataset quality checks:")
    sys.exit(0 if run_checks() else 1)
