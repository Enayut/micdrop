"""Download raw extracts of the external datasets used by the MicDrop pipeline.

Design notes
------------
* Only user-side instruction/question TEXT is extracted. Assistant answers are
  never stored, which makes the "output is a prompt, not an answer" property
  structural rather than filtered-in afterwards.
* Bandwidth-aware: DOWIS is fetched through the HF datasets-server /rows API
  (1,320 small JSON pages, zero of its 1.8 GB audio), Magicoder is streamed
  with a byte cap (the repo only ships one 255 MB JSONL; we keep the first
  chunk of complete lines), oasst1 uses the small ready-reviewed JSONL (34 MB)
  instead of the full parquet.
* gooaq is NOT downloaded by default: the allenai/gooaq repo contains no data
  files (a loading script pulls ~2 GB externally), so it stays optional.

Usage:
    python data_pipeline/download_datasets.py            # all active sources
    python data_pipeline/download_datasets.py dowis dolly
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_sources import SOURCES  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw"

# Sources downloaded by default, in order.
ACTIVE_SOURCES = ["dowis", "dolly", "codealpaca", "magicoder", "oasst1"]

ROW_CAPS = {
    "oasst1": 60_000,      # prompter messages before filtering
    "magicoder": 60 * 1024 * 1024,   # byte cap on the 255 MB JSONL stream
}

UA = {"User-Agent": "micdrop-dataset-pipeline/1.0"}


def _save(rows: list[dict], name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  saved {len(rows):>6} rows -> {out}")
    return out


def _get_with_retry(url: str, *, stream: bool = False, timeout: int = 120,
                    max_retries: int = 6, headers: dict | None = None):
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, stream=stream, timeout=timeout,
                                headers={**UA, **(headers or {})})
            if resp.status_code >= 500 or resp.status_code == 429:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = min(60.0, 2.0 ** attempt)
            print(f"  transient error ({exc}); retry {attempt}/{max_retries} in {wait:.0f}s")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# --------------------------------------------------------------------- dowis --

def download_dowis() -> None:
    """Fetch DOWIS text prompts without downloading audio bytes.

    DOWIS ships text and human-recorded audio side by side (1.8 GB of
    parquet). We use the Hugging Face datasets-server /rows API instead,
    which paginates over rows as small JSON documents; we keep only the four
    text fields for English rows. Audio is never transferred.
    """
    src = SOURCES["dowis"]
    url = "https://datasets-server.huggingface.co/rows"
    session = requests.Session()
    params = {"dataset": src.hf_id, "config": "default", "split": "test",
              "offset": 0, "length": 100}
    rows: list[dict] = []
    total = None
    retries = 0
    while True:
        try:
            resp = session.get(url, params=params, timeout=60, headers=UA)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} server error")
            resp.raise_for_status()
        except requests.RequestException as exc:
            retries += 1
            if retries > 6:
                raise
            wait = min(30.0, 2.0 ** retries)
            print(f"\n  transient error ({exc}); retry {retries}/6 in {wait:.0f}s")
            time.sleep(wait)
            continue
        retries = 0
        data = resp.json()
        total = data.get("num_rows_total") or total
        batch = data.get("rows", [])
        for item in batch:
            r = item.get("row", {})
            if r.get("language") == "en" and r.get("text_prompt"):
                rows.append({
                    "text_prompt": r["text_prompt"],
                    "task": r.get("task", ""),
                    "prompt_type": r.get("prompt_type", ""),
                })
        params["offset"] += len(batch)
        if not batch or (total is not None and params["offset"] >= total):
            break
        print(f"  {params['offset']}/{total} rows scanned", end="\r")
    print()
    _save(rows, "dowis_en_prompts")


# --------------------------------------------------------------------- dolly --

DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
             "resolve/main/databricks-dolly-15k.jsonl")


def download_dolly() -> None:
    resp = _get_with_retry(DOLLY_URL, timeout=300)
    lines = resp.text.splitlines()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"instruction": r["instruction"], "context": r["context"] or "",
                     "category": r["category"]})
    _save(rows, "dolly_15k")


# ---------------------------------------------------------------- codealpaca --

CODEALPACA_URL = ("https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/"
                  "resolve/main/code_alpaca_20k.json")


def download_codealpaca() -> None:
    resp = _get_with_retry(CODEALPACA_URL, timeout=300)
    records = json.loads(resp.text)
    rows = [{"instruction": r["instruction"], "input": r.get("input") or "",
             "output": r.get("output") or ""} for r in records]
    _save(rows, "code_alpaca_20k")


# ---------------------------------------------------------------- magicoder --

MAGICODER_URL = ("https://huggingface.co/datasets/ise-uiuc/"
                 "Magicoder-Evol-Instruct-110K/resolve/main/"
                 "data-evol_instruct-decontaminated.jsonl")


def download_magicoder() -> None:
    """Stream the first ROW_CAPS['magicoder'] bytes of the single 255 MB JSONL.

    The repo ships no smaller/parquet variant, so we cap the stream and keep
    only complete lines. Roughly 25k instructions -- far more than the
    transform step consumes.
    """
    cap = ROW_CAPS["magicoder"]
    resp = _get_with_retry(MAGICODER_URL, stream=True, timeout=600,
                           headers={"Range": f"bytes=0-{cap - 1}"})
    rows: list[dict] = []
    buffer = b""
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        downloaded += len(chunk)
        buffer += chunk
        *complete, buffer = buffer.split(b"\n")
        for line in complete:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append({"instruction": r["instruction"]})
        if downloaded >= cap:
            break
    _save(rows, "magicoder_40k")


# ------------------------------------------------------------------- oasst1 --

OASST1_READY_URL = ("https://huggingface.co/datasets/OpenAssistant/oasst1/"
                    "resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz")


def download_oasst1() -> None:
    """Keep prompter-role messages from the quality-reviewed subset.

    The 'ready' JSONL (34 MB gz) contains only messages that passed OASST's
    quality review -- smaller and cleaner than the full export. Conversation
    structure is not needed here; we keep candidate user prompts only and
    drop duplicate texts.
    """
    cap = ROW_CAPS["oasst1"]
    resp = _get_with_retry(OASST1_READY_URL, stream=True, timeout=600)
    seen: set[str] = set()
    rows: list[dict] = []
    with gzip.open(resp.raw, "rt", encoding="utf-8") as gz:
        for line in gz:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("role") != "prompter":
                continue
            lang = r.get("lang", "")
            if lang and lang != "en":
                continue
            text = r["text"].strip()
            if not text or text in seen:
                continue
            seen.add(text)
            rows.append({"text": text})
            if len(rows) >= cap:
                break
    _save(rows, "oasst1_en_prompts")


DOWNLOADERS = {
    "dowis": download_dowis,
    "dolly": download_dolly,
    "codealpaca": download_codealpaca,
    "magicoder": download_magicoder,
    "oasst1": download_oasst1,
}


def main() -> None:
    wanted = sys.argv[1:] or ACTIVE_SOURCES
    unknown = [w for w in wanted if w not in DOWNLOADERS]
    if unknown:
        print(f"Unknown sources: {unknown}. Available: {sorted(DOWNLOADERS)}")
        sys.exit(1)
    print(f"Downloading raw extracts to {RAW_DIR}")
    for name in wanted:
        status = SOURCES[name].license_status
        if "REJECTED" in status.upper():
            print(f"[skip] {name}: license status is {status}")
            continue
        print(f"[{name}] license: {SOURCES[name].license} ({status})")
        DOWNLOADERS[name]()
    print("Done.")


if __name__ == "__main__":
    main()
