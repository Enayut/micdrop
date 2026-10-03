"""Build the final MicDrop dataset: train / validation / test + reports + preview.

Pipeline position: download -> transform -> [this] -> quality checks.

Split policy (leak prevention):
  * Records are grouped by normalized target output ("output group"). All
    noisy variants of the same clean prompt land in the SAME split, so no
    output string can appear in both train and validation/test. This is the
    leakage that would otherwise inflate evaluation scores.
  * Splits are drawn group-wise with fixed RNG seed for reproducibility:
    test gets ~6.5%, validation ~6.5%, train the rest.
  * After splitting, an assertion verifies zero overlap of normalized outputs
    across splits.

Outputs:
  dataset/train.jsonl, dataset/validation.jsonl, dataset/test.jsonl
  dataset/preview.jsonl                 (>=100 random examples, human inspection)
  dataset/processed/all_records.jsonl   (pre-split records, for analysis)
  dataset/eval_report.json + eval_report.md

Usage:
    python data_pipeline/build_dataset.py [--target-total N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_dataset import clean_records, normalize_for_dedup  # noqa: E402
from transform_datasets import run_all  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"

TEST_FRACTION = 0.065
VAL_FRACTION = 0.065
DEFAULT_TARGET_TOTAL = 1_400       # ~1,000 train + 200 + 200
RNG_SEED = 20260925

# Order in which sources fill the dataset when the total must be trimmed.
# micdrop_real (the user's own recordings) has top priority.
SOURCE_PRIORITY = ["micdrop_seed", "micdrop_real", "dowis", "dolly", "codealpaca",
                   "magicoder", "oasst1", "gooaq"]


def output_group_key(rec: dict) -> str:
    return normalize_for_dedup(rec["output"])


def build(target_total: int, rng_seed: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    rng = random.Random(rng_seed)

    # 1. transform
    by_source = run_all(rng_seed)

    # 2. flatten and clean (dedup etc.) over everything
    all_records: list[dict] = []
    for name in SOURCE_PRIORITY:
        all_records.extend(by_source.get(name, []))
    kept, clean_stats = clean_records(all_records)

    # 3. group by output to keep variants of one prompt in one split
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in kept:
        groups[output_group_key(rec)].append(rec)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    n_test = max(1, round(len(kept) * TEST_FRACTION))
    n_val = max(1, round(len(kept) * VAL_FRACTION))

    test_groups: list[str] = []
    acc = 0
    for key in group_keys:                     # test first
        if acc >= n_test:
            break
        test_groups.append(key)
        acc += len(groups[key])

    test_set = set(test_groups)
    val_groups: list[str] = []
    acc = 0
    for key in group_keys:                     # then validation, from the rest
        if key in test_set or acc >= n_val:
            continue
        val_groups.append(key)
        acc += len(groups[key])
    val_set = set(val_groups)

    train_groups = [k for k in group_keys if k not in test_set and k not in val_set]

    def flatten(group_key_list: list[str]) -> list[dict]:
        out = []
        for k in group_key_list:
            out.extend(groups[k])
        return out

    test = flatten(test_groups)
    val = flatten(val_groups)
    train = flatten(train_groups)

    # 4. optional trimming to target size, respecting group integrity
    total = len(train) + len(val) + len(test)
    if total > target_total:
        excess = total - target_total
        while excess > 0 and train_groups:      # drop whole groups from train
            key = train_groups.pop()
            excess -= len(groups[key])
        train = flatten(train_groups)

    # 5. verify no leakage of outputs across splits
    def outputs_of(rows: list[dict]) -> set[str]:
        return {output_group_key(r) for r in rows}

    assert not (outputs_of(train) & outputs_of(val)), "train/val output overlap"
    assert not (outputs_of(train) & outputs_of(test)), "train/test output overlap"
    assert not (outputs_of(val) & outputs_of(test)), "val/test output overlap"

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rng_seed": rng_seed,
        "clean_stats": clean_stats,
        "groups_total": len(group_keys),
    }
    return train, val, test, meta


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_report(train: list[dict], val: list[dict], test: list[dict],
                meta: dict) -> dict:
    all_recs = {"train": train, "validation": val, "test": test}
    report: dict = {
        "generated_at": meta["built_at"],
        "rng_seed": meta["rng_seed"],
        "totals": {k: len(v) for k, v in all_recs.items()},
        "total_examples": sum(len(v) for v in all_recs.values()),
        "per_category": {},
        "per_source": {},
        "synthetic_vs_real": {},
        "duplicates_removed": meta["clean_stats"],
        "length_stats": {},
    }
    for split, rows in all_recs.items():
        report["per_category"][split] = dict(Counter(r["category"] for r in rows))
        report["per_source"][split] = dict(Counter(r["source"] for r in rows))
        report["synthetic_vs_real"][split] = {
            "synthetic": sum(1 for r in rows if r.get("synthetic")),
            "real_seed_clean": sum(1 for r in rows if not r.get("synthetic")),
        }
    for split, rows in all_recs.items():
        in_lens = sorted(len(r["input"]) for r in rows)
        out_lens = sorted(len(r["output"]) for r in rows)
        def _stat(lens: list[int]) -> dict:
            if not lens:
                return {}
            return {
                "min": lens[0], "max": lens[-1],
                "mean": round(sum(lens) / len(lens), 1),
                "p95": lens[min(len(lens) - 1, int(0.95 * len(lens)))],
            }
        report["length_stats"][split] = {
            "input_chars": _stat(in_lens), "output_chars": _stat(out_lens),
        }
    # longest/shortest overall
    all_rows = train + val + test
    by_in = sorted(all_rows, key=lambda r: len(r["input"]))
    report["longest_inputs"] = [
        {"input": r["input"][:160], "output": r["output"][:120],
         "source": r["source"]} for r in by_in[-3:]]
    report["shortest_inputs"] = [
        {"input": r["input"][:160], "output": r["output"][:120],
         "source": r["source"]} for r in by_in[:3]]
    return report


def report_markdown(report: dict) -> str:
    lines = [
        "# MicDrop dataset evaluation report",
        "",
        f"Generated: {report['generated_at']}  (rng seed {report['rng_seed']})",
        "",
        "## Totals",
        "",
        "| split | examples |",
        "|---|---|",
    ]
    for split, n in report["totals"].items():
        lines.append(f"| {split} | {n} |")
    lines.append(f"| **total** | **{report['total_examples']}** |")

    lines += ["", "## Examples per category", "",
              "| category | train | validation | test |", "|---|---|---|---|"]
    cats = sorted({c for split in report["per_category"].values() for c in split})
    for cat in cats:
        row = [report["per_category"][split].get(cat, 0)
               for split in ("train", "validation", "test")]
        lines.append(f"| {cat} | {row[0]} | {row[1]} | {row[2]} |")

    lines += ["", "## Examples per source", "",
              "| source | train | validation | test |", "|---|---|---|---|"]
    srcs = sorted({s for split in report["per_source"].values() for s in split})
    for s in srcs:
        row = [report["per_source"][split].get(s, 0)
               for split in ("train", "validation", "test")]
        lines.append(f"| {s} | {row[0]} | {row[1]} | {row[2]} |")

    lines += ["", "## Synthetic vs real-seed clean", ""]
    for split, d in report["synthetic_vs_real"].items():
        lines.append(f"- **{split}**: synthetic={d['synthetic']}, "
                     f"real_seed_clean={d['real_seed_clean']}")

    lines += ["", "## Duplicates removed (across all records before split)", ""]
    for k, v in report["duplicates_removed"].items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Length statistics (characters)", "",
              "| split | input mean | input p95 | output mean | output p95 |",
              "|---|---|---|---|---|"]
    for split, ls in report["length_stats"].items():
        lines.append(
            f"| {split} | {ls['input_chars']['mean']} | {ls['input_chars']['p95']} "
            f"| {ls['output_chars']['mean']} | {ls['output_chars']['p95']} |")

    lines += ["", "## Longest inputs", ""]
    for r in report["longest_inputs"]:
        lines.append(f"- `{r['source']}`: {r['input']}...")
    lines += ["", "## Shortest inputs", ""]
    for r in report["shortest_inputs"]:
        lines.append(f"- `{r['source']}`: {r['input']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    print(f"Building MicDrop dataset (target total {args.target_total})...")
    train, val, test, meta = build(args.target_total, args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROCESSED_DIR / "all_records.jsonl", train + val + test)
    write_jsonl(DATASET_DIR / "train.jsonl", train)
    write_jsonl(DATASET_DIR / "validation.jsonl", val)
    write_jsonl(DATASET_DIR / "test.jsonl", test)

    # preview: >=100 random examples for manual inspection
    rng = random.Random(args.seed + 1)
    pool = train + val + test
    preview = rng.sample(pool, min(120, len(pool)))
    write_jsonl(DATASET_DIR / "preview.jsonl", preview)

    report = make_report(train, val, test, meta)
    (DATASET_DIR / "eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATASET_DIR / "eval_report.md").write_text(
        report_markdown(report), encoding="utf-8")

    print(f"\ntrain      : {len(train)}")
    print(f"validation : {len(val)}")
    print(f"test       : {len(test)}")
    print(f"total      : {len(train) + len(val) + len(test)}")
    print(f"preview    : {len(preview)} examples (dataset/preview.jsonl)")
    print(f"\nWrote dataset/*.jsonl, dataset/eval_report.{{json,md}}")


if __name__ == "__main__":
    main()
