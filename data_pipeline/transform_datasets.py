"""Transform raw extracts + seed data into unified input->output records.

Every produced record has the shape:
    {"input": str, "output": str, "category": str, "source": str,
     "synthetic": bool, "topic": str}

`category` uses the MicDrop noise/function taxonomy:
    clean_simple, filler_removal, repetition, speech_correction,
    technical_correction, programming, research, creative, questions,
    multi_part, numbers_versions, file_paths, commands, long_requests,
    mixed_noise, speech_style_normalization

`topic` is a separate analysis field (coding, systems, ai_ml, data, general)
so that semantic topic and noise function do not get conflated.

Record kinds:
  1. Seed-based noisy pairs (synthetic=True):  noise engine applied to
     hand-written MicDrop seeds; output = the original clean seed.
  2. Seed-based clean pairs (synthetic=False): seeds kept as-is
     (input == output modulo casing/punctuation normalization), so the model
     learns that not every input needs aggressive rewriting.
  3. External-source noisy pairs (synthetic=True): clean instruction text
     from public datasets becomes the OUTPUT; a Whisper-like noisy rendering
     becomes the INPUT.

Filtering (applied to external text before use):
  * must look like an instruction/question/request, not an answer
  * must be self-contained (no context-dependence, no embedded code blocks)
  * length within spoken-utterance bounds
  * English only
  * minimal quality heuristics (no markup, no all-caps shouting, no lists)
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noisy_inputs import noise_for_category  # noqa: E402
from seed_data import SEED_PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"

# ---------------------------------------------------------------------------
# Tunable budget per source
# ---------------------------------------------------------------------------

SEED_NOISY_PER_PROMPT = 3          # noisy variants per seed prompt
# Every seed also becomes one clean pair (input ~= output). The filter's
# per-output cap is 4, so 1 clean + 3 noisy fits exactly; the clean variant is
# sorted first and always survives. This gives ~10% clean_simple examples so
# the model does not learn that every input needs aggressive rewriting.
SEED_CLEAN_FRACTION = 1.0
EXTERNAL_QUOTA = {                 # target records per external source
    "dolly": 450,
    "codealpaca": 500,
    "magicoder": 350,
    "oasst1": 500,
    "gooaq": 400,
}

MAX_INPUT_CHARS = 320              # spoken utterances are short
MAX_OUTPUT_CHARS = 300
MIN_OUTPUT_CHARS = 12

# ---------------------------------------------------------------------------
# Text-quality filters for external prompts
# ---------------------------------------------------------------------------

QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which", "can", "could",
    "should", "is", "are", "do", "does", "did", "will", "would", "shall",
    "may", "am", "was", "were", "has", "have", "had",
)
COMMAND_STARTERS = (
    "write", "create", "make", "build", "show", "list", "find", "explain",
    "give", "implement", "generate", "convert", "add", "fix", "set", "run",
    "install", "configure", "use", "compare", "design", "refactor", "parse",
    "draft", "translate", "summarize", "research", "open", "remove", "move",
    "copy", "delete", "enable", "disable", "start", "stop", "deploy", "test",
)
CONV_PREFIX_RE = re.compile(
    r"^(hey|hi|hello|okay|ok|so|um|uh|alright|yo|hiya)[,!\s]+", re.I)


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_list_markup(text: str) -> str:
    """OASST1 texts sometimes carry markdown bullets/headers; drop those rows."""
    if re.search(r"(^|\n)\s*[-*•]\s", text) or re.search(r"(^|\n)#{1,6}\s", text):
        return ""
    return text


def has_code_block(text: str) -> bool:
    return "```" in text or re.search(r"^\s{4,}\S", text, re.M) is not None


def is_english(text: str) -> bool:
    if not text:
        return False
    return sum(1 for c in text if ord(c) < 128) / len(text) > 0.97


def strip_conversational_prefix(text: str) -> str:
    """Normalize 'Hey, can you ...' style openings to a bare imperative."""
    t = CONV_PREFIX_RE.sub("", text).strip()
    t = re.sub(r"^(can you|could you|please|i need you to)\s+", "", t, flags=re.I)
    if t:
        t = t[0].upper() + t[1:]
    return t


def topic_of(text: str) -> str:
    low = text.lower()
    if any(k in low for k in (
            "python", "javascript", "typescript", "react", "next.js", "node",
            "fastapi", "pytorch", "tensorflow", "transformers", "hugging face",
            "langchain", "openai", "claude", "gemini", "qwen", "sql", "postgres",
            "mongodb", "docker", "kubernetes", "aws", "git", "github", "api",
            "css", "rust", "django", "flask", "regex", "bash", "cli", "script",
            "function", "algorithm", "refactor", "code", "bug", "json", "yaml",
            "csv", "sdk", "endpoint", "auth", "deploy", "typescript")):
        return "coding"
    if any(k in low for k in (
            "linux", "fedora", "systemd", "wayland", "cuda", "gpu", "server",
            "ssh", "nginx", "cron", "kernel", "shell", "disk", "memory",
            "docker ", "kubernetes", "container")):
        return "systems"
    if any(k in low for k in (
            "machine learning", "deep learning", "llm", "neural", "model",
            "training", "fine-tune", "fine tune", "embedding", "rag",
            "vector", "dataset", "transformer", "inference", "quantiz",
            "lora", "agent", "whisper", "speech", "transcri")):
        return "ai_ml"
    if any(k in low for k in (
            "database", "sql", "postgres", "mongo", "query", "table",
            "schema", "migration", "parquet", "kafka", "etl", "pipeline")):
        return "data"
    return "general"


TECH_HINTS = ("python", "javascript", "typescript", "react", "next.js", "node",
              "fastapi", "pytorch", "tensorflow", "transformers", "hugging face",
              "langchain", "openai", "sql", "postgres", "mongodb", "docker",
              "kubernetes", "aws", "linux", "git", "api", "machine learning",
              "deep learning", "llm", "rag", "vector database", "cuda", "bash",
              "css", "rust", "django", "flask", "pandas", "numpy", "database",
              "server", "script", "function", "algorithm", "deploy")


def looks_like_instruction(text: str) -> bool:
    """Heuristic: accept prompts a user could plausibly say to an assistant."""
    t = text.strip()
    low = t.lower()
    if not t or len(t) < MIN_OUTPUT_CHARS or len(t) > MAX_OUTPUT_CHARS:
        return False
    if not is_english(t):
        return False
    if has_code_block(t) or any(m in t for m in ("{", "}", "<", ">", "|", "```", "$(")):
        return False
    if re.search(r"(^|\n)\s*[-*•\d]+[.)]?\s", t):      # lists
        return False
    if t.isupper() or "\n" in t:
        return False
    if t.count("!") > 1 or t.count("?") > 2:
        return False
    if re.search(r"https?://|www\.", t):
        return False
    first_word = re.match(r"[a-z']+", low)
    if first_word and (first_word.group(0) in COMMAND_STARTERS
                       or first_word.group(0) in QUESTION_STARTERS):
        return True
    if t.endswith("?"):                                # questions, any form
        return True
    if low.startswith(("do ", "help me ", "let me ", "tell me ", "show me ")):
        return True
    return False


def looks_like_answer(text: str) -> bool:
    """Reject rows that are answers, not instructions."""
    low = text.lower()
    answer_starts = (
        "the answer", "you should", "you can", "you need", "here is", "here's",
        "sure", "certainly", "as an ai", "sorry", "unfortunately", "i think",
        "yes", "no,", "first,", "second,", "in summary", "to summarize",
    )
    return low.startswith(answer_starts)


FOLLOWUP_OPENERS = (
    "make it", "now ", "and ", "also ", "then ", "instead ", "again ",
    "what about", "how about", "continue", "go on", "next ", "repeat",
    "shorter", "simpler", "longer", "in english", "translate it",
    "explain it to", "more detail", "less detail", "that one", "this one",
    "these examples", "can you give me an example of that",
)


def looks_like_followup(text: str) -> bool:
    """Reject conversation follow-ups whose meaning depends on prior context
    (e.g. 'Make it simpler', 'Now explain it to a dog')."""
    low = text.lower().strip()
    return low.startswith(FOLLOWUP_OPENERS)


def auto_category(output_text: str) -> str:
    """Functional category from the text itself (for external prompts)."""
    low = output_text.lower()
    if output_text.endswith("?"):
        return "questions"
    if any(k in low for k in TECH_HINTS):
        return "programming"
    if re.search(r"\d", output_text):
        return "numbers_versions"
    if "/" in output_text or "\\" in output_text:
        return "file_paths"
    if low.startswith(("run ", "start ", "stop ", "show ", "list ")):
        return "commands"
    if any(low.startswith(w) for w in ("compare", "research", "summarize", "explain")):
        return "research"
    if " and " in low and output_text.count(",") >= 1:
        return "multi_part"
    if len(output_text) > 200:
        return "long_requests"
    return "creative"


def normalize_output(text: str) -> str:
    """Canonical output formatting: single-spaced, capitalized, punctuated.

    Questions end with '?', everything else with '.' -- the model must
    preserve questions as questions. wh-words always signal a question;
    auxiliaries only when followed by a pronoun ("do you...", "is it...")
    so imperatives like "Do not commit secrets" keep a period.
    """
    text = clean_whitespace(text)
    if not text:
        return text
    text = text[0].upper() + text[1:]
    low = text.lower()
    is_question = bool(re.match(r"^(what|why|how|when|where|who|which|whose)\b", low))
    if not is_question:
        m = re.match(r"^(can|could|should|is|are|do|does|did|will|would|shall|may|"
                     r"was|were|has|have|had)\b", low)
        if m:
            second = low.split()[1] if len(low.split()) > 1 else ""
            is_question = second in {"you", "i", "we", "it", "they", "he",
                                     "she", "there", "this"}
    if text.endswith(("?", "!")):
        return text
    if text.endswith("."):
        # single-sentence wh-question with a source period -> restore '?'
        if is_question and "?" not in text and text.count(".") == 1:
            return text[:-1] + "?"
        return text
    return text + ("?" if is_question else ".")


def normalize_clean_input(text: str) -> str:
    """Clean examples keep input≈output; minor casual variation only."""
    text = clean_whitespace(text)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Category assignment from noise ops
# ---------------------------------------------------------------------------

def category_from_ops(ops: list[str], fallback: str) -> str:
    """Map the noise operators that fired to the MicDrop taxonomy."""
    op_set = set(ops)
    if not op_set:
        return "clean_simple"
    if op_set == {"technical_confusion"}:
        return "technical_correction"
    if "false_start" in op_set or "repetition" in op_set:
        return "repetition" if "false_start" not in op_set else "speech_correction"
    if op_set == {"fillers"} and fallback in {
            "programming", "technical_correction", "commands", "file_paths",
            "numbers_versions", "multi_part", "long_requests"}:
        # pure uh/um over technical content: the content is the signal
        return fallback
    if op_set <= {"fillers", "conversational", "informal"}:
        return "filler_removal"
    if len(op_set) >= 3:
        return "mixed_noise"
    return fallback


# ---------------------------------------------------------------------------
# Seed-based records
# ---------------------------------------------------------------------------

def build_seed_records(rng: random.Random) -> list[dict]:
    records: list[dict] = []
    for clean, category in SEED_PROMPTS:
        target = normalize_output(clean)
        topic = topic_of(target)
        # several distinct noisy renderings per seed, profile-diverse
        seen_inputs: set[str] = set()
        tries = 0
        made = 0
        while made < SEED_NOISY_PER_PROMPT and tries < SEED_NOISY_PER_PROMPT * 3:
            tries += 1
            variant = noise_for_category(target, category, rng)
            noisy_in = variant["input"].strip()
            if noisy_in == target or noisy_in in seen_inputs:
                continue
            seen_inputs.add(noisy_in)
            records.append({
                "input": variant["input"],
                "output": target,
                "category": category_from_ops(variant.get("ops", []), category),
                "source": "micdrop_seed",
                "synthetic": True,
                "topic": topic,
            })
            made += 1
        # clean pair for a fraction of seeds
        if rng.random() < SEED_CLEAN_FRACTION:
            records.append({
                "input": normalize_clean_input(target),
                "output": target,
                "category": "clean_simple",
                "source": "micdrop_seed",
                "synthetic": False,
                "topic": topic,
            })
    return records


# ---------------------------------------------------------------------------
# External sources
# ---------------------------------------------------------------------------

def _take_quota(rows: list[str], quota: int, rng: random.Random) -> list[str]:
    rng.shuffle(rows)
    return rows[:quota]


def _make_pair(text: str, source: str, rng: random.Random,
               force_category: str | None = None) -> dict | None:
    """Common external-source path: clean text -> (noisy input, clean output)."""
    target = normalize_output(strip_conversational_prefix(text))
    if not target or target.lower() in ("the.", "a."):
        return None
    if len(target) > MAX_OUTPUT_CHARS:
        return None
    cat = force_category or auto_category(target)
    noisy = noise_for_category(target, cat, rng)
    if noisy["input"].strip() == target:
        return None                       # noise produced nothing; skip
    return {
        "input": noisy["input"],
        "output": target,
        "category": category_from_ops(noisy.get("ops", []), cat),
        "source": source,
        "synthetic": True,
        "topic": topic_of(target),
    }


def transform_dolly(rng: random.Random) -> list[dict]:
    path = RAW_DIR / "dolly_15k.jsonl"
    if not path.exists():
        return []
    freeform = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") not in {"open_qa", "brainstorming", "generation"}:
                continue                      # closed_qa/IE depend on 'context'
            if r.get("context", "").strip():
                continue
            text = clean_whitespace(r["instruction"])
            if not looks_like_instruction(text) or looks_like_answer(text):
                continue
            freeform.append(text)
    quota = EXTERNAL_QUOTA["dolly"]
    picked = _take_quota(freeform, quota * 2, rng)   # oversample; filters reduce
    records = []
    for text in picked:
        rec = _make_pair(text, "dolly", rng)
        if rec:
            records.append(rec)
        if len(records) >= quota:
            break
    return records


def transform_codealpaca(rng: random.Random) -> list[dict]:
    path = RAW_DIR / "code_alpaca_20k.jsonl"
    if not path.exists():
        return []
    candidates = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("input", "").strip():
                continue                      # references a separate input field
            text = clean_whitespace(r["instruction"])
            if not looks_like_instruction(text) or looks_like_answer(text):
                continue
            candidates.append(text)
    quota = EXTERNAL_QUOTA["codealpaca"]
    picked = _take_quota(candidates, quota * 2, rng)
    records = []
    for text in picked:
        rec = _make_pair(text, "codealpaca", rng)
        if rec:
            records.append(rec)
        if len(records) >= quota:
            break
    return records


def transform_magicoder(rng: random.Random) -> list[dict]:
    path = RAW_DIR / "magicoder_40k.jsonl"
    if not path.exists():
        return []
    candidates = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = clean_whitespace(r["instruction"])
            if not looks_like_instruction(text) or looks_like_answer(text):
                continue
            if not (20 <= len(text) <= 200):  # spoken-plausible lengths only
                continue
            candidates.append(text)
    quota = EXTERNAL_QUOTA["magicoder"]
    picked = _take_quota(candidates, quota * 2, rng)
    records = []
    for text in picked:
        rec = _make_pair(text, "magicoder", rng)
        if rec:
            records.append(rec)
        if len(records) >= quota:
            break
    return records


def transform_oasst1(rng: random.Random) -> list[dict]:
    path = RAW_DIR / "oasst1_en_prompts.jsonl"
    if not path.exists():
        return []
    candidates = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = strip_list_markup(clean_whitespace(r["text"]))
            if not text:
                continue
            if not looks_like_instruction(text) or looks_like_answer(text):
                continue
            if looks_like_followup(text):
                continue
            candidates.append(text)
    quota = EXTERNAL_QUOTA["oasst1"]
    picked = _take_quota(candidates, quota * 2, rng)
    records = []
    for text in picked:
        rec = _make_pair(text, "oasst1", rng)
        if rec:
            records.append(rec)
        if len(records) >= quota:
            break
    return records


def transform_gooaq(rng: random.Random) -> list[dict]:
    path = RAW_DIR / "gooaq_questions.jsonl"
    if not path.exists():
        return []
    candidates = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = clean_whitespace(r["question"])
            if not (10 <= len(text) <= 150) or not is_english(text):
                continue
            if any(m in text for m in ("{", "}", "<", ">", "|", "```", "$(", "http")):
                continue
            if looks_like_answer(text):
                continue
            candidates.append(text)
    quota = EXTERNAL_QUOTA["gooaq"]
    picked = _take_quota(candidates, quota * 2, rng)
    records = []
    for text in picked:
        rec = _make_pair(text, "gooaq", rng, force_category="questions")
        if rec:
            records.append(rec)
        if len(records) >= quota:
            break
    return records


def transform_dowis(rng: random.Random) -> list[dict]:
    """DOWIS English text prompts -> spoken-style normalization examples.

    DOWIS prompts are human-written spoken instruction phrasings for SLLM
    tasks (5 styles incl. informal). The clean text becomes the output;
    synthetic Whisper-like noise creates the input. The style/task tags are
    preserved for analysis.
    """
    path = RAW_DIR / "dowis_en_prompts.jsonl"
    if not path.exists():
        return []
    seen: set[str] = set()
    candidates: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = clean_whitespace(r.get("text_prompt", ""))
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            if not (15 <= len(text) <= MAX_OUTPUT_CHARS) or not is_english(text):
                continue
            if len(text.split()) < 3:                 # 'summarize.' etc.
                continue
            if re.fullmatch(r"[A-Z0-9\- ]+", text):   # 'S2ST English' task codes
                continue
            if any(m in text for m in ("{", "}", "<", ">", "|", "```", "$(")):
                continue                              # markup / placeholders
            candidates.append({"text": text, "style": r.get("prompt_type", ""),
                               "task": r.get("task", "")})
    records = []
    for cand in candidates:
        target = normalize_output(cand["text"])
        if len(target) > MAX_OUTPUT_CHARS:
            continue
        noisy = noise_for_category(target, "general", rng)
        if noisy["input"].strip() == target:
            # keep as clean pair so spoken phrasings are represented
            records.append({"input": normalize_clean_input(target), "output": target,
                            "category": "speech_style_normalization",
                            "source": "dowis", "synthetic": False,
                            "topic": topic_of(target)})
        else:
            records.append({"input": noisy["input"], "output": target,
                            "category": "speech_style_normalization",
                            "source": "dowis", "synthetic": True,
                            "topic": topic_of(target)})
    return records


def transform_micdrop_real(rng: random.Random) -> list[dict]:
    """Passthrough for YOUR real Whisper recordings.

    Drop a file at dataset/raw/micdrop_real.jsonl with one JSON object per
    line:
        {"input": "<raw Whisper output>", "output": "<prompt you meant>"}
    Records are lightly validated (dedup/length/answer checks) but the text
    is used verbatim -- no synthetic noise is applied. This is the highest-
    value data MicDrop can train on; see DATASET.md 'Adding real recordings'.
    """
    path = RAW_DIR / "micdrop_real.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inp, outp = clean_whitespace(r.get("input", "")), clean_whitespace(r.get("output", ""))
            if not inp or not outp or len(outp) > MAX_OUTPUT_CHARS:
                continue
            if looks_like_answer(outp):
                continue
            records.append({"input": inp, "output": outp,
                            "category": r.get("category", "real_recording"),
                            "source": "micdrop_real", "synthetic": False,
                            "topic": topic_of(outp)})
    return records


EXTERNAL_TRANSFORMS = {
    "micdrop_real": transform_micdrop_real,
    "dowis": transform_dowis,
    "dolly": transform_dolly,
    "codealpaca": transform_codealpaca,
    "magicoder": transform_magicoder,
    "oasst1": transform_oasst1,
    "gooaq": transform_gooaq,
}


def run_all(rng_seed: int = 20260925) -> dict[str, list[dict]]:
    rng = random.Random(rng_seed)
    result = {"micdrop_seed": build_seed_records(rng)}
    print(f"micdrop_seed    pairs: {len(result['micdrop_seed'])}")
    for name, fn in EXTERNAL_TRANSFORMS.items():
        recs = fn(rng)
        print(f"{name:<15} pairs: {len(recs)}")
        result[name] = recs
    return result


if __name__ == "__main__":
    out = run_all()
    total = sum(len(v) for v in out.values())
    print(f"\ntotal transformed records: {total}")
