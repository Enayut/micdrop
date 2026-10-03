"""Inspect raw dataset extracts (and optionally live HF sources).

For each raw/*.jsonl extract this prints:
  * row count
  * fields present
  * character-length stats (min/median/p95/max)
  * 3 preview rows

Also verifies the registered license table (dataset_sources.py) against the
Hugging Face API when the network is available, so DATASET_SOURCES.md can be
checked rather than trusted. Run with --live for the network check.

Usage:
    python data_pipeline/inspect_datasets.py
    python data_pipeline/inspect_datasets.py --live
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_sources import SOURCES, license_report  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw"

# Which field of each raw extract holds the "clean text" candidates.
TEXT_FIELDS = {
    "dowis_en_prompts": "text_prompt",
    "dolly_15k": "instruction",
    "code_alpaca_20k": "instruction",
    "magicoder_40k": "instruction",
    "oasst1_en_prompts": "text",
    "gooaq_questions": "question",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inspect_extract(path: Path) -> None:
    rows = load_jsonl(path)
    if not rows:
        print(f"\n=== {path.name}: EMPTY ===")
        return
    fields = sorted({k for r in rows for k in r})
    lens = [len(r.get(TEXT_FIELDS.get(path.stem, ""), "") or "") for r in rows]
    lens = [n for n in lens if n > 0]
    lens.sort()
    print(f"\n=== {path.name} ===")
    print(f"  rows       : {len(rows)}")
    print(f"  fields     : {fields}")
    if lens:
        p95 = lens[int(0.95 * len(lens)) - 1] if len(lens) > 1 else lens[-1]
        print(f"  text len   : min={min(lens)} median={int(statistics.median(lens))} "
              f"p95={p95} max={max(lens)}")
    for r in rows[:3]:
        text = r.get(TEXT_FIELDS.get(path.stem, ""), "")
        print(f"  e.g.       : {str(text)[:100]!r}")


def live_license_check() -> None:
    """Verify license tags of HF-hosted sources via the Hub API."""
    from huggingface_hub import HfApi
    api = HfApi()
    print("Live HF license check (when network is available):")
    for src in SOURCES.values():
        if not src.hf_id or src.role in {"rejected", "unused"}:
            continue
        try:
            info = api.dataset_info(src.hf_id)
            tags = info.card_data.to_dict() if info.card_data else {}
            lic = tags.get("license", info.tags and [t for t in info.tags if t.startswith("license:")])
            print(f"  {src.key:<12} {src.hf_id:<45} license={lic}")
        except Exception as exc:  # network off, gated repo, etc.
            print(f"  {src.key:<12} {src.hf_id:<45} unavailable: {type(exc).__name__}")


def main() -> None:
    if "--live" in sys.argv:
        live_license_check()
        print()
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        print(f"No raw extracts found in {RAW_DIR}. Run download_datasets.py first.")
        return
    print("Raw extracts:")
    for path in files:
        inspect_extract(path)
    print("\nRegistered sources and roles:")
    for row in license_report():
        print(f"  {row['key']:<14} {row['role']:<22} {row['license_status']}")


if __name__ == "__main__":
    main()
