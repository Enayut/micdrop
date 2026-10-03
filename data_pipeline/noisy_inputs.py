"""Realistic speech/Whisper-like noise generation for clean prompts.

Design principles
-----------------
* Noise must be REALISTIC, not random corruption. Noise categories mirror
  what actually happens between "user speaks" and "Whisper writes":
    1. fillers (uh, um, like, you know, i mean)
    2. repetitions / stutters
    3. false starts and self-corrections
    4. conversational frames (hey can you, i need you to, please)
    5. loss of punctuation and capitalization
    6. homophone-like technical misrecognitions (malloc -> malok)
    7. spoken-form numbers ("three point nine" for 3.9)
  and the *target* is always the original clean prompt, so the model learns
  normalization, not corruption.

* Not every example is noisy. A fraction of seeds stays clean so the model
  learns to leave already-clean prompts (mostly) alone.

* Technical misrecognitions are context-driven word substitutions, not a
  post-hoc dictionary of whole utterances: the generator still has to pick a
  consistent spoken rendering for the whole sentence, so the model must use
  context (surrounding technical words) to recover the intended term.
"""

from __future__ import annotations

import random
import re

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

FILLERS = ["uh", "um", "uhm", "like", "you know", "i mean", "so yeah", "kind of", "sort of"]
REPETITION_WORDS = {
    "write", "create", "make", "build", "show", "find", "add", "fix",
    "set", "run", "use", "a", "the", "that", "with", "and", "to", "for",
}
CONV_PREFIXES = [
    "hey can you {clean_lower}",
    "can you {clean_lower}",
    "please {clean_lower}",
    "i need you to {clean_lower}",
    "could you {clean_lower}",
]
CONV_SUFFIXES = [
    "{clean} thanks",
    "{clean} if possible",
    "{clean} asap",
    "{clean} please",
]

# Technical terms and their realistic spoken/ASR-degraded renderings.
# Key insight: these are *word-level substitutions* the generator applies in
# context, NOT utterance-level hardcoded corrections. The model must learn to
# map them back from surrounding context (e.g. "malok ... kallak in c" -> C
# memory functions -> malloc/calloc).
TECH_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "malloc": ("malok", "mal lock", "mallok"),
    "calloc": ("kallak", "callac", "caloc"),
    "pytorch": ("pie torch", "pie-torch", "pi torch"),
    "postgresql": ("post grass sql", "postgres q l", "postgre sql"),
    "kubernetes": ("cooper netties", "koo bern etees", "cuberneties"),
    "hugging face": ("hugging phase", "huggingface"),
    "tensorflow": ("tensor flow",),
    "fastapi": ("fast api",),
    "qwen": ("gwen", "q when"),
    "claude": ("clod",),
    "cuda": ("cooda", "kuda"),
    "lora": ("laura",),
    "docker": ("dock her",),
    "kafka": ("cough ka",),
    "nginx": ("engine x",),
    "linux": ("linnux",),
    "github": ("git hub",),
    "gitlab": ("git lab",),
    "typescript": ("type script",),
    "javascript": ("java script",),
    "sqlite": ("sequel lite",),
    "sql": ("sequel",),
    "csv": ("c s v",),
    "json": ("dot jason", "jason"),
    "yaml": ("yam mel",),
    "api": ("a p i",),
    "cli": ("c l i",),
    "gpu": ("g p u",),
    "s3": ("s three",),
    "vector database": ("vector d b",),
    "pgvector": ("pg vector",),
    "oauth": ("o auth",),
    "sigterm": ("sig term",),
    "pydantic": ("pie dantic",),
    "fastapi.": ("fast api.",),
    "systemd": ("system d",),
    "rsync": ("r sync",),
    "pytest": ("pie test",),
    "redis": ("red iss",),
    "mongodb": ("mongo d b",),
    "openapi": ("open api",),
    "oidc": ("o i d c",),
    "ssd": ("s s d",),
    "vram": ("v ram",),
    "numpy": ("num pie",),
    "transformer": ("transform her",),
    "langchain": ("lang chain",),
    "tiktoken": ("tick token",),
    "bitsandbytes": ("bits and bytes",),
    "wayland": ("way land",),
    "evdev": ("ev dev",),
    "ydotool": ("y do tool",),
    "wl-copy": ("w l copy",),
    "cifar": ("c far",),
    "sigterm.": ("sig term.",),
    "sigterm,": ("sig term,",),
}

# Words that are unambiguous spoken renderings of symbols and must NOT be
# re-corrupted by the number/technical noise passes.
SPOKEN_SYMBOL_WORDS = {
    "point", "slash", "dot", "percent", "hundred", "thousand", "million",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "zero", "sixteen", "thirty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "twelve", "eleven", "twenty", "eighteen", "fourteen",
    "single", "double",
}

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
UNIT_WORDS = {  # spoken renderings of version/number suffixes
    "mb": "megabytes", "gb": "gigabytes", "kb": "kilobytes",
    "cpu": "c p u", "gpu": "g p u", "am": "a m", "pm": "p m",
}

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def lowercase_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def strip_punct(text: str) -> str:
    """Remove sentence punctuation (keep intra-token chars like dots in paths)."""
    text = re.sub(r"\s([.,;:!?])", r"", text)      # space before punct
    text = re.sub(r"([.,;:!?])(\s|$)", r"\2", text)  # trailing punct
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


# ---------------------------------------------------------------------------
# Noise operators
# ---------------------------------------------------------------------------

def op_fillers(text: str, rng: random.Random) -> str:
    words = tokenize(strip_punct(text))
    out: list[str] = []
    n_fillers = rng.randint(1, 3)
    positions = sorted(rng.sample(range(1, max(2, len(words))), min(n_fillers, max(1, len(words) - 1))))
    used = set()
    for i, w in enumerate(words):
        out.append(w)
        for p in positions:
            if p == i and i not in used and i > 0:
                out.append(rng.choice(FILLERS))
                used.add(i)
    return " ".join(out)


def op_repetition(text: str, rng: random.Random) -> str:
    words = tokenize(strip_punct(text))
    if len(words) < 4:
        return text
    out = []
    i = 0
    reps_done = 0
    restarts_done = 0
    inserted = False
    while i < len(words):
        w = words[i]
        out.append(w)
        if reps_done < 2 and w.lower() in REPETITION_WORDS and rng.random() < 0.5:
            # stutter once or twice, then possibly a "no wait" restart
            # (at most one "no wait" per utterance - more is not realistic)
            if restarts_done < 1 and rng.random() < 0.35:
                out.extend(["no wait", w])
                restarts_done += 1
            else:
                out.append(w)
                if rng.random() < 0.3:
                    out.append(w)
            reps_done += 1
            inserted = True
        i += 1
    # if no repetition was actually inserted, report "unchanged" so category
    # attribution does not over-count repetition (punctuation stripping alone
    # is not a repetition signal)
    return " ".join(out) if inserted else text


def op_false_start(text: str, rng: random.Random) -> str:
    """Cut off a first attempt, then say the real thing."""
    words = tokenize(strip_punct(text))
    if len(words) < 6:
        return " ".join(words)
    cut = rng.randint(2, max(3, len(words) // 3))
    first_attempt = " ".join(words[:cut])
    marker = rng.choice(["uh actually", "wait no", "sorry i mean", "actually let me", "no"])
    continuation = lowercase_first(" ".join(words))
    return f"{first_attempt} {marker} {continuation}"


def op_conversational(text: str, rng: random.Random) -> str:
    clean = strip_punct(text)
    clean_lower = lowercase_first(clean)
    if rng.random() < 0.5:
        return CONV_PREFIXES[rng.randrange(len(CONV_PREFIXES))].format(
            clean=clean, clean_lower=clean_lower)
    return CONV_SUFFIXES[rng.randrange(len(CONV_SUFFIXES))].format(clean=clean)


def op_technical_confusion(text: str, rng: random.Random) -> str:
    """Apply word-level ASR misrecognitions to technical terms, in context."""
    words = tokenize(strip_punct(text))
    out = []
    hits = 0
    for w in words:
        lw = w.lower()
        # try longest-match first (e.g. "vector database")
        replaced = False
        if lw in TECH_CONFUSIONS and TECH_CONFUSIONS[lw]:
            variants = TECH_CONFUSIONS[lw]
            if rng.random() < 0.85:
                out.append(rng.choice(variants))
                hits += 1
                replaced = True
        if not replaced:
            out.append(w)
    if hits == 0:
        # guarantee at least one substitution by rewriting deterministically
        for i, w in enumerate(out):
            lw = w.lower()
            if lw in TECH_CONFUSIONS and TECH_CONFUSIONS[lw]:
                out[i] = TECH_CONFUSIONS[lw][0]
                hits = 1
                break
    return " ".join(out)


def spoken_version(text: str) -> str:
    """Render digits/symbols as their spoken form ("3.9" -> "three point nine")."""
    s = text
    # versions like 3.9, 2.1, 12.1 -> "three point nine"
    def _ver(match: re.Match) -> str:
        a, b = match.group(1), match.group(2)
        return f"{DIGIT_WORDS.get(a, a)} point {DIGIT_WORDS.get(b, b)}"
    s = re.sub(r"\b(\d+)\.(\d+)\b", _ver, s)
    # standalone integers < 1000 -> words (keep version-like handled above)
    def _int(match: re.Match) -> str:
        n = int(match.group(0))
        if n < 10:
            return DIGIT_WORDS[str(n)]
        if n == 10:
            return "ten"
        if n == 12:
            return "twelve"
        if n == 16:
            return "sixteen"
        if n == 50:
            return "fifty"
        if n == 100:
            return "hundred"
        if n == 512:
            return "five hundred twelve"
        if n == 1000:
            return "thousand"
        words = DIGIT_WORDS.get(match.group(0))
        return words if words else str(n)
    s = re.sub(r"\b\d+\b", _int, s)
    # paths/symbols
    s = s.replace("/", " slash ").replace("~/.config", "tilde slash dot config")
    s = re.sub(r"\b~/", "tilde slash ", s)
    s = s.replace(":", " colon ").replace("%", " percent ")
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)  # tidy double spaces before punct
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def op_numbers_spoken(text: str, rng: random.Random) -> str:
    s = spoken_version(text)
    # units: "512 megabytes" spoken; keep as-is (already spoken by spoken_version)
    return s


def op_lowercase(text: str, rng: random.Random) -> str:
    return text.lower()


def op_merged_words(text: str, rng: random.Random) -> str:
    """Whisper sometimes merges short function words with neighbors.

    Restricted to plausible cases ("a lot" -> "alot", "to it" -> "toit"):
    both words lowercase, second word short, and neither is a filler or a
    spoken symbol word, and at most one merge per sentence.
    """
    words = tokenize(text)
    if len(words) < 5:
        return " ".join(words)
    mergeable = {"a", "to", "of", "the", "in", "on", "for"}
    candidates = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if (w1.lower() in mergeable and w2.islower() and w2.isalpha()
                and len(w2) <= 6 and w2.lower() not in FILLERS
                and w2.lower() not in SPOKEN_SYMBOL_WORDS):
            candidates.append(i)
    if candidates and rng.random() < 0.6:
        i = rng.choice(candidates)
        words = words[:i] + [words[i] + words[i + 1]] + words[i + 2:]
    return " ".join(words)


def op_informal(text: str, rng: random.Random) -> str:
    """Contractions + casual spellings, meaning preserved."""
    s = strip_punct(text)
    s = re.sub(r"\bit is\b", "it's", s, flags=re.I)
    s = re.sub(r"\bdo not\b", "don't", s, flags=re.I)
    s = re.sub(r"\bcannot\b", "can't", s, flags=re.I)
    s = re.sub(r"\bthat is\b", "that's", s, flags=re.I)
    s = re.sub(r"\bthere is\b", "there's", s, flags=re.I)
    s = re.sub(r"\bgonna\b", "going to", s, flags=re.I)  # opposite direction: keep informal spoken
    s = re.sub(r"\bI am\b", "I'm", s, flags=re.I)
    return lowercase_first(s)


# Noise profiles per category of noisy example. Each profile is a list of
# operator names applied with a probability, then a final pass strips
# punctuation/capitalization to mimic Whisper's raw output.
#
# "*_focus" profiles exist to balance training categories: they emphasize one
# noise type so the model gets dedicated examples of each skill instead of
# everything collapsing into filler-removal.
NOISE_PROFILES: dict[str, list[tuple[str, float]]] = {
    "light":   [("fillers", 0.9), ("conversational", 0.25), ("informal", 0.3)],
    "medium":  [("fillers", 0.8), ("repetition", 0.35), ("conversational", 0.3),
                ("technical_confusion", 0.6), ("numbers_spoken", 0.25)],
    "heavy":   [("fillers", 0.7), ("repetition", 0.4), ("false_start", 0.35),
                ("conversational", 0.4), ("technical_confusion", 0.8),
                ("numbers_spoken", 0.35), ("merged_words", 0.2)],
    "clean_tech":   [],
    "spoken_literal": [("numbers_spoken", 1.0), ("technical_confusion", 0.9)],
    "tech_focus":        [("technical_confusion", 0.95), ("fillers", 0.15)],
    "repetition_focus":  [("repetition", 0.8), ("fillers", 0.25)],
    "filler_only":       [("fillers", 1.0)],
    "false_start_focus": [("false_start", 0.85), ("fillers", 0.3)],
    "mixed_focus":       [("fillers", 0.7), ("repetition", 0.5),
                          ("conversational", 0.5), ("technical_confusion", 0.5),
                          ("numbers_spoken", 0.3)],
}


def _apply_profile(text: str, profile: str, rng: random.Random) -> tuple[str, list[str]]:
    ops = {
        "fillers": op_fillers,
        "repetition": op_repetition,
        "false_start": op_false_start,
        "conversational": op_conversational,
        "technical_confusion": op_technical_confusion,
        "numbers_spoken": op_numbers_spoken,
        "lowercase": op_lowercase,
        "merged_words": op_merged_words,
        "informal": op_informal,
    }
    out = text
    applied: list[str] = []
    for name, prob in NOISE_PROFILES[profile]:
        if rng.random() < prob:
            candidate = ops[name](out, rng)
            if candidate != out:
                applied.append(name)
            out = candidate
    return out, applied


def generate_noisy(text: str, profile: str, rng: random.Random) -> tuple[str, list[str]]:
    """Generate one noisy rendering of `text` under the given profile.

    Returns (noisy_text, list_of_noise_ops_that_changed_the_text).
    """
    return _apply_profile(text, profile, rng)


def make_noisy_variants(text: str, n: int, rng: random.Random,
                        allow_clean: bool = True) -> list[dict]:
    """Produce up to n variants; some may be clean when allow_clean is True.

    Every variant maps back to the SAME clean target (the original `text`),
    which is the whole point: input is a realistic spoken/ASR rendering,
    output is the clean intent-preserving prompt.
    """
    profiles = ["light", "medium", "heavy", "clean_tech", "spoken_literal"]
    variants: list[dict] = []
    for _ in range(n):
        profile = rng.choice(profiles)
        if profile == "clean_tech" and not allow_clean:
            profile = "light"
        noisy, applied = generate_noisy(text, profile, rng)
        # de-dup: skip if identical to the clean text AND profile wasn't clean_tech
        if noisy.strip() == text.strip() and profile != "clean_tech":
            continue
        variants.append({"input": noisy, "profile": profile, "ops": applied})
    return variants


# ---------------------------------------------------------------------------
# Category-aware noise application used by transform step
# ---------------------------------------------------------------------------

def noise_for_category(clean: str, category: str, rng: random.Random) -> dict:
    """Choose a noise profile appropriate to the example's category.

    Returns {"input", "profile", "ops"} where ops lists the noise operators
    that actually changed the text (used for category assignment downstream).
    Focus profiles are weighted in so every noise category is represented.
    """
    if category in {"programming", "technical_correction", "commands", "file_paths"}:
        profiles = ["tech_focus", "spoken_literal", "medium", "clean_tech",
                    "repetition_focus", "heavy", "false_start_focus"]
    elif category in {"numbers_versions"}:
        profiles = ["spoken_literal", "tech_focus", "medium"]
    elif category in {"research", "creative", "questions", "general"}:
        profiles = ["light", "medium", "clean_tech", "repetition_focus",
                    "false_start_focus"]
    elif category in {"multi_part", "long_requests"}:
        profiles = ["light", "mixed_focus", "clean_tech", "false_start_focus"]
    else:
        profiles = ["light", "medium", "clean_tech", "false_start_focus"]
    profile = rng.choice(profiles)
    noisy, applied = generate_noisy(clean, profile, rng)
    return {"input": noisy, "profile": profile, "ops": applied}


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    samples = [
        "What is the difference between malloc and calloc in C?",
        "Migrate our project from Python 3.9 to 3.12.",
        "Copy all JSON config files from /etc to my home directory.",
        "Set up a PyTorch training loop with gradient accumulation and mixed precision.",
        "What is the difference between append and extend in Python lists?",
    ]
    for s in samples:
        print("CLEAN:", s)
        for v in make_noisy_variants(s, 3, rng):
            print("  NOISY:", v["input"], f"[{v['profile']}]")
        print()
