"""Verified registry of external datasets for the MicDrop prompt-cleaning task.

Every entry was inspected against the actual source (Hugging Face dataset
metadata / GitHub repository / paper) on 2026-09-25. Fields were taken from the
dataset cards, API responses, or repository contents -- not invented.

Each entry defines:
  * provenance (source, license, verified evidence)
  * what is safe to use from it (transform contract): MicDrop only consumes
    user-side *instruction/question text*; assistant answers are never used.
  * how it contributes to the input->output training format.

License statuses:
  OK                       license verified, use allowed with attribution
  OK_WITH_ATTRIBUTION      share-alike: the final dataset must stay under the same license
  LICENSE_REVIEW_REQUIRED  unclear/missing license terms, do not use until reviewed
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSource:
    key: str                       # short internal id, used as "source" metadata
    hf_id: str | None = None       # Hugging Face dataset id, if any
    github: str | None = None      # GitHub repo, if any
    license: str = "UNKNOWN"       # as stated by the source
    license_status: str = "LICENSE_REVIEW_REQUIRED"
    approx_size: str = "unknown"
    languages: tuple[str, ...] = ()
    audio: bool = False
    has_transcripts: bool = False
    fields: tuple[str, ...] = ()
    what_we_use: str = ""
    transformation: str = ""
    notes: str = ""
    role: str = "unused"           # role in this pipeline
    # Only these columns are ever downloaded (leaves audio / answers behind).
    download_columns: tuple[str, ...] = ()
    text_extractors: dict = field(default_factory=dict)


SOURCES: dict[str, DatasetSource] = {}


def _register(src: DatasetSource) -> None:
    SOURCES[src.key] = src


# ---------------------------------------------------------------- selected ---

_register(DatasetSource(
    key="micdrop_seed",
    license="Original content written for MicDrop",
    license_status="OK",
    approx_size="hand-written seed (~1.2k examples)",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=("input", "output", "category"),
    what_we_use=(
        "Hand-written MicDrop-specific instruction pairs covering technical "
        "terms, file paths, numbers/versions, multi-part requests, and "
        "self-corrections; also used as raw material for synthetic noise."
    ),
    transformation="Used directly as clean examples; also used as noisy-input targets.",
    notes=(
        "Written for this project; represents the target distribution of "
        "MicDrop prompts (coding, tooling, ML/AI, system administration)."
    ),
    role="seed_and_noisy_pairs",
))

_register(DatasetSource(
    key="dowis",
    hf_id="maikezu/dowis",
    github="https://github.com/MaikeZuefle/DOWIS",
    license="CC-BY-4.0",
    license_status="OK (verified via HF card + arXiv:2603.09881 footnote)",
    approx_size="1,320 rows; ~990 unique written prompts, 9-11 tasks x 12 languages, 3h17m audio",
    languages=("en", "de", "it", "cs", "es", "fr", "hu", "nl", "pt", "ru", "sq", "sv"),
    audio=True,
    has_transcripts=True,
    fields=("text_prompt", "audio_prompt_female_1", "audio_prompt_female_2",
            "audio_prompt_male_1", "audio_prompt_male_2", "language", "task", "prompt_type"),
    what_we_use="Only the text_prompt column (English rows), never the audio files.",
    transformation=(
        "Clean spoken-instruction phrasings (5 styles incl. informal) are used as "
        "outputs; synthetic Whisper-like noise is applied to create inputs "
        "(category speech_style_normalization)."
    ),
    notes=(
        "Human-written-and-recorded parallel spoken/written prompts for SLLM "
        "evaluation. Audio is only 1.8 GB and is NOT needed: the parquet is "
        "read with column projection so only text_prompt is fetched. Prompt "
        "content is SLLM task phrasing (ASR/MT/summarization/TTS...), which "
        "teaches conversational spoken phrasing but not technical terms."
    ),
    role="clean_prompt_source",
    download_columns=("text_prompt", "language", "task", "prompt_type"),
    text_extractors={
        "clean": lambda r: r.get("text_prompt"),
        "task": lambda r: r.get("task"),
        "style": lambda r: r.get("prompt_type"),
    },
))

_register(DatasetSource(
    key="dolly",
    hf_id="databricks/databricks-dolly-15k",
    license="CC-BY-SA-3.0",
    license_status="OK_WITH_ATTRIBUTION (share-alike: license MicDrop dataset CC-BY-SA-3.0)",
    approx_size="15,015 records (free-form subset ~7,488)",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=("instruction", "context", "response", "category"),
    what_we_use=(
        "instruction field of the free-form categories only "
        "(open_qa, brainstorming, generation) where the instruction is a "
        "self-contained user request."
    ),
    transformation=(
        "Instruction used as the output; synthetic Whisper-like noise applied "
        "to create the input."
    ),
    notes=(
        "Human-written Databricks employee prompts. Self-contained only: "
        "closed_qa / information_extraction / summarization rows depend on a "
        "separate context field and would be unanswerable as a bare prompt, so "
        "they are filtered out. Share-alike obligations apply to the final "
        "dataset because dolly content is included."
    ),
    role="clean_prompt_source",
    download_columns=("instruction", "context", "response", "category"),
    text_extractors={"clean": lambda r: r.get("instruction"), "task": lambda r: r.get("category")},
))

_register(DatasetSource(
    key="codealpaca",
    hf_id="sahil2801/CodeAlpaca-20k",
    license="CC-BY-4.0",
    license_status="OK (verified via HF card)",
    approx_size="20,022 records",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=("instruction", "input", "output"),
    what_we_use="instruction field only (technical/programming prompts).",
    transformation=(
        "Instruction used as the output; synthetic Whisper-like noise applied "
        "to create the input. Examples where the instruction references a "
        "separate 'input' field are skipped."
    ),
    notes=(
        "Self-instruct style programming prompts covering Python, JS, SQL, "
        "APIs, data structures, etc. Some prompts are GPT-generated; noisy "
        "variants are still useful because the target is the original text."
    ),
    role="clean_prompt_source",
    download_columns=("instruction", "input", "output"),
    text_extractors={"clean": lambda r: r.get("instruction")},
))

_register(DatasetSource(
    key="magicoder",
    hf_id="ise-uiuc/Magicoder-Evol-Instruct-110K",
    license="Apache-2.0",
    license_status="OK (verified via HF card; decontaminated evol-codealpaca)",
    approx_size="110,182 records",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=("instruction", "response"),
    what_we_use="instruction field only, filtered for short, instruction-like prompts.",
    transformation=(
        "Instruction used as the output; synthetic Whisper-like noise applied "
        "to create the input. Prompts embedding code blocks are skipped."
    ),
    notes=(
        "Deep technical prompts across languages/frameworks. Evolved/instruct-"
        "style, so longer and more detailed than MicDrop's typical spoken "
        "utterance; length filter keeps only spoken-plausible prompts. The "
        "repo ships a single 255 MB JSONL, so the downloader byte-caps the "
        "stream (~25k instructions) - plenty for the transform quota."
    ),
    role="clean_prompt_source",
    download_columns=("instruction", "response"),
    text_extractors={"clean": lambda r: r.get("instruction")},
))

_register(DatasetSource(
    key="oasst1",
    hf_id="OpenAssistant/oasst1",
    license="Apache-2.0",
    license_status="OK (verified via HF card)",
    approx_size="161,443 messages / 66,497 conversation trees",
    languages=("en", "es", "ru", "de", "pl", "th", "vi", "sv", "bn", "da", "he", "it",
               "fa", "sk", "id", "nb", "el", "nl", "hu", "eu", "zh", "eo", "ja", "ca",
               "cs", "bg", "fi", "pt", "tr", "ro", "ar", "uk", "gl", "fr", "ko"),
    audio=False,
    has_transcripts=False,
    fields=("text", "role", "lang", "synthetic", "deleted", "review_result", "..."),
    what_we_use=(
        "prompter-role message text from the quality-reviewed 'ready' subset, "
        "filtered to question/instruction-like prompts."
    ),
    transformation=(
        "Prompter text used as the output; synthetic Whisper-like noise applied "
        "to create the input."
    ),
    notes=(
        "Crowd-sourced real user prompts, often informally phrased - closest "
        "match to what people actually dictate. Question-heavy, which balances "
        "the command-heavy synthetic seeds. We read the 34 MB ready.messages "
        "JSONL instead of the full parquet."
    ),
    role="clean_prompt_source",
    download_columns=("text", "role", "lang", "message_tree_id", "created_date"),
    text_extractors={"clean": lambda r: r.get("text")},
))

_register(DatasetSource(
    key="gooaq",
    hf_id="allenai/gooaq",
    license="Apache-2.0",
    license_status="OK (verified via HF card) - not used in current build",
    approx_size="3,112,679 train questions (~2 GB download)",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=("id", "question", "short_answer", "answer", "answer_type"),
    what_we_use="Nothing in the current build (optional future source).",
    transformation="None.",
    notes=(
        "Question mining would ensure the model keeps questions as questions. "
        "REJECTED FOR THE CURRENT BUILD for a practical reason: the repo "
        "hosts no data files - its loading script downloads ~2 GB from an "
        "external URL, and it is not served by the HF datasets-server API. "
        "The transform step still supports it if the raw extract is produced "
        "manually (transform_datasets.transform_gooaq)."
    ),
    role="optional",
    download_columns=("question",),
    text_extractors={"clean": lambda r: r.get("question")},
))

_register(DatasetSource(
    key="whisper_code",
    github="https://github.com/openai/whisper (whisper/normalizers/basic.py, MIT)",
    license="MIT (via OpenAI whisper code)",
    license_status="OK (tooling only, no dataset content)",
    approx_size="code only",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=(),
    what_we_use=(
        "EnglishTextNormalizer (standardize spellings like 'wanna', 'gonna', "
        "drop commas) to convert clean outputs into Whisper-style text before "
        "noise injection."
    ),
    transformation="Normalization utility for the synthetic noise generator.",
    notes=(
        "Using Whisper's own normalizer makes synthetic inputs match real "
        "Whisper output conventions instead of a hand-rolled approximation."
    ),
    role="tooling",
))

# -------------------------------------------------------------- evaluated ----

_register(DatasetSource(
    key="sift50m",
    hf_id="amazon-agi/SIFT-50M",
    license="CDLA-Sharing-1.0",
    license_status="REJECTED_FOR_THIS_PIPELINE",
    approx_size="50M examples / 14K hours (per paper arXiv:2504.09081)",
    languages=("en", "de", "fr", "it", "es"),
    audio=True,
    has_transcripts=True,
    fields=("instruction", "audio (14K hours)", "response"),
    what_we_use="Nothing (rejected).",
    transformation="None.",
    notes=(
        "Closest content match (speech instructions), but instructions are "
        "mostly TTS/ASR/SLU task commands generated by LLMs over speech "
        "corpora - not user intent prompts - and much of its phrasing would "
        "need heavy filtering. Primary reason for rejection: ~50M-example "
        "download and 14K hours of audio is impractical here, and CDLA-"
        "Sharing terms add review overhead for marginal gain over DOWIS."
    ),
    role="rejected",
))

_register(DatasetSource(
    key="no_robots",
    hf_id="HuggingFaceH4/no_robots",
    license="CC-BY-NC-4.0",
    license_status="REJECTED_FOR_THIS_PIPELINE",
    approx_size="10,000 records",
    languages=("en",),
    audio=False,
    has_transcripts=False,
    fields=("prompt", "messages", "category"),
    what_we_use="Nothing (rejected).",
    transformation="None.",
    notes=(
        "Excellent quality human prompts, but the license is NonCommercial. "
        "Depending on how MicDrop is used/distributed this may be fine, but it "
        "blocks any commercial use of the fine-tuned artifact and would make "
        "the license picture of the final dataset messy. Excluded for "
        "license simplicity."
    ),
    role="rejected",
))

_register(DatasetSource(
    key="mcif",
    hf_id="FBK-MT/MCIF",
    license="CC-BY-4.0",
    license_status="OK (verified via HF card) - not used in current build",
    approx_size="<1K rows across configs (e.g. long_fixedprompt: 362 test rows)",
    languages=("en", "de", "it", "zh"),
    audio=True,
    has_transcripts=True,
    fields=("prompt_en", "prompt_de", "prompt_it", "prompt_zh", "metadata", "audio", "video", "text"),
    what_we_use="Nothing in the current build (optional future source).",
    transformation="None.",
    notes=(
        "Human-annotated multimodal instruction benchmark from scientific "
        "talks; prompt_en/prompt_de/prompt_it are human-written instructions "
        "similar in spirit to DOWIS. Small row count adds little on top of "
        "DOWIS; kept as an optional extension."
    ),
    role="optional",
))

_register(DatasetSource(
    key="lmsys_chat_1m",
    hf_id="lmsys/lmsys-chat-1m",
    license="LMSYS-Chat-1M Dataset License Agreement (custom, gated)",
    license_status="LICENSE_REVIEW_REQUIRED",
    approx_size="1,000,000 conversations",
    languages=("en", "multilingual"),
    audio=False,
    has_transcripts=False,
    fields=("conversation_id", "model", "conversation", "turn", "language", "openai_moderation"),
    what_we_use="Nothing (gated; requires agreeing to a custom license).",
    transformation="None.",
    notes=(
        "Real user prompts would be valuable, but access is gated behind a "
        "custom agreement and usage restrictions (redistribution limits). "
        "Marked LICENSE_REVIEW_REQUIRED rather than silently using it."
    ),
    role="rejected",
))

_register(DatasetSource(
    key="oasst2",
    hf_id="OpenAssistant/oasst2",
    license="Apache-2.0 (per OpenAssistant corpus terms)",
    license_status="OK (not needed; oasst1 suffices for scale)",
    approx_size="135K messages",
    languages=("multilingual",),
    audio=False,
    has_transcripts=False,
    fields=("text", "role", "lang", "..."),
    what_we_use="Nothing in the current build.",
    transformation="None.",
    notes="Optional extension source alongside oasst1 if more informal user prompts are needed.",
    role="optional",
))


def license_report() -> list[dict]:
    """Return the DATASET_SOURCES.md table rows, sorted by role."""
    order = {"seed_and_noisy_pairs": 0, "clean_prompt_source": 1, "tooling": 2,
             "optional": 3, "unused": 4, "rejected": 5}
    rows = []
    for src in sorted(SOURCES.values(), key=lambda s: order.get(s.role, 9)):
        rows.append({
            "key": src.key,
            "source": src.hf_id or src.github or src.key,
            "license": src.license,
            "license_status": src.license_status,
            "approx_size": src.approx_size,
            "what_we_use": src.what_we_use,
            "transformation": src.transformation,
            "notes": src.notes,
            "role": src.role,
        })
    return rows


if __name__ == "__main__":
    for row in license_report():
        print(f"{row['key']:<14} {row['license']:<22} {row['role']}")
    print(f"\nRegistered datasets: {len(SOURCES)}")
    print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
