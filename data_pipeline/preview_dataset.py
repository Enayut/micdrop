"""Generate dataset/preview.jsonl: random examples for manual inspection.

Draws >=100 examples (or all, if fewer) from the three splits, printed and
written to dataset/preview.jsonl. Run any time after build_dataset.py; also
supports filtering by category/source for focused review.

Usage:
    python data_pipeline/preview_dataset.py
    python data_pipeline/preview_dataset.py --category technical_correction
    python data_pipeline/preview_dataset.py --source dowis --n 40
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
PREVIEW_SIZE = 120


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PREVIEW_SIZE)
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    pool: list[dict] = []
    for split in ("train", "validation", "test"):
        path = DATASET_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                rec["_split"] = split
                pool.append(rec)

    if args.category:
        pool = [r for r in pool if r.get("category") == args.category]
    if args.source:
        pool = [r for r in pool if r.get("source") == args.source]

    rng = random.Random(args.seed)
    preview = rng.sample(pool, min(args.n, len(pool)))

    out = DATASET_DIR / "preview.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in preview:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Preview pool: {len(pool)} examples; wrote {len(preview)} -> {out}\n")
    for r in preview[:10]:
        print(f"[{r['_split']}/{r['source']}/{r['category']}]")
        print(f"  IN : {r['input']}")
        print(f"  OUT: {r['output']}\n")
    print("(...see dataset/preview.jsonl for the rest)")


if __name__ == "__main__":
    main()
