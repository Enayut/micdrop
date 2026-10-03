# MicDrop dataset sources

Every external dataset below was inspected against its actual source (Hugging
Face dataset card / API response, GitHub repository, or the underlying paper)
on **2026-09-25**. Licenses and sizes are transcribed from those sources, not
invented. Rows marked `LICENSE_REVIEW_REQUIRED` are **not** used.

MicDrop's transformation task is: *messy spoken/Whisper-like transcription →
clean, intent-preserving prompt*. From every external dataset we therefore
consume **only user-side instruction/question text** — never assistant
answers — so the "output is a prompt, not an answer" property is structural.

## Summary

| Role | Datasets |
|---|---|
| Used in current build | micdrop_seed (original), dowis, dolly, codealpaca, magicoder, oasst1 |
| Optional / not needed | mcif, oasst2, gooaq |
| Rejected | sift50m, no_robots, lmsys_chat_1m |
| Tooling | whisper code (EnglishTextNormalizer ideas, MIT) |

## Selected datasets

| Dataset | Source | License | Approx. size | What we use from it | Transformation performed | License status | Notes |
|---|---|---|---|---|---|---|---|
| **DOWIS** (Do What I Say) | HF `maikezu/dowis`, github.com/MaikeZuefle/DOWIS | CC-BY-4.0 | 1,320 rows; ~990 unique written prompts; 11-12 languages; 3h17m audio; 9-11 tasks x 5 prompt styles | `text_prompt` column, English rows only. Audio is NEVER downloaded (fetched via datasets-server /rows API as small JSON pages). | Clean spoken-phrased instructions (incl. informal style) become outputs; synthetic Whisper-like noise creates inputs → `speech_style_normalization` examples; prompts that already resist noise become clean pairs. | OK — verified via HF card + arXiv:2603.09881 footnote | The user-recorded parallel spoken/written prompt dataset; closest public match to MicDrop's spoken-prompt distribution. Not technical, so it contributes style, not jargon. |
| **databricks-dolly-15k** | HF `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | 15,015 records (~7,488 free-form) | `instruction` of free-form categories only (open_qa, brainstorming, generation), self-contained rows | Instruction → output; synthetic noise → input | OK_WITH_ATTRIBUTION — share-alike: the combined dataset must be licensed CC-BY-SA-3.0 | Human-written Databricks-employee prompts. closed_qa / information_extraction rows depend on a separate `context` field and are filtered out. |
| **CodeAlpaca-20k** | HF `sahil2801/CodeAlpaca-20k` | CC-BY-4.0 | 20,022 records | `instruction` only; rows whose instruction references a separate `input` field are skipped | Instruction → output; synthetic noise → input; category `programming` | OK — verified via HF card | Self-instruct style technical prompts (Python, JS, SQL, APIs...). Some GPT-generated text; usable since the target is the original string. |
| **Magicoder-Evol-Instruct-110K** | HF `ise-uiuc/Magicoder-Evol-Instruct-110K` | Apache-2.0 | 110,182 records | `instruction` only, filtered to 20-200 chars (spoken-plausible) and instruction-like | Instruction → output; synthetic noise → input; category `programming` | OK — verified via HF card | Deep technical coverage across languages/frameworks; decontaminated evol-codealpaca. Single 255 MB JSONL in repo → downloader byte-caps the stream. |
| **OASST1** | HF `OpenAssistant/oasst1` | Apache-2.0 | 161,443 messages; we read the 34 MB quality-reviewed "ready" JSONL | `prompter`-role messages (English), question/instruction-like only | Prompter text → output; synthetic noise → input | OK — verified via HF card | Crowd-sourced real user prompts, often informally phrased — closest match to dictated phrasing; question-heavy to balance command-heavy seeds. |
| **MicDrop seed set** | this repo (`data_pipeline/seed_data.py`) | Original content written for MicDrop | ~120 hand-written prompts × ≤3 noisy variants + clean pairs | Clean prompts across programming, technical terms, numbers/versions, file paths, multi-part, commands, research, creative, questions + DOWIS-style general phrasings | Direct clean pairs + noise-engine targets | OK | Defines MicDrop's target distribution: technical jargon (malloc/calloc, PyTorch, PostgreSQL, Kubernetes, CUDA...), Whisper-style misrecognitions, spoken numbers. |

## Optional (verified, not needed for current build)

| Dataset | Source | License | Approx. size | What we would use | Why not now |
|---|---|---|---|---|---|
| **MCIF** | HF `FBK-MT/MCIF` | CC-BY-4.0 | <1K rows per config (e.g. 362 in long_fixedprompt) | `prompt_en`/`prompt_de` human-written instructions | Human-written instruction benchmark from scientific talks; small row count adds little beyond DOWIS. |
| **OASST2** | HF `OpenAssistant/oasst2` | Apache-2.0 | ~135K messages | prompter messages like oasst1 | oasst1 already covers informal user prompts for current scale. |
| **GooAQ** | HF `allenai/gooaq` | Apache-2.0 | 3,112,679 train questions | `question` field | Repo hosts no data files: the loading script pulls ~2 GB from an external URL, and it is not served by the HF datasets-server API. `transform_datasets.transform_gooaq` still supports it if the raw extract is produced manually. |

## Rejected datasets

| Dataset | Source | License | Approx. size | Why rejected |
|---|---|---|---|---|
| **SIFT-50M** | HF `amazon-agi/SIFT-50M` | CDLA-Sharing-1.0 | 50M examples / 14K hours of audio | Closest content match (speech instructions) but LLM-generated task commands over speech corpora (TTS/ASR/SLU phrasing, not user intent), a 50M-example / 14K-hour download is impractical here, and CDLA-Sharing terms add review overhead for marginal gain over DOWIS. |
| **no_robots** | HF `HuggingFaceH4/no_robots` | CC-BY-NC-4.0 | 10,000 records | Excellent human prompts, but NonCommercial: blocks any commercial use of the fine-tuned artifact and complicates the final dataset's license. Excluded for license simplicity. |
| **LMSYS-Chat-1M** | HF `lmsys/lmsys-chat-1m` | LMSYS-Chat-1M Dataset License Agreement (custom, gated) | 1,000,000 conversations | Gated behind a custom agreement with redistribution limits → `LICENSE_REVIEW_REQUIRED`; not silently used. |

## Tooling

| Tool | Source | License | Use |
|---|---|---|---|
| Whisper EnglishTextNormalizer conventions | github.com/openai/whisper (`whisper/normalizers/basic.py`) | MIT | Reference for Whisper-style output conventions (lowercase, contraction normalization); reimplemented as light heuristics in `noisy_inputs.py`, no code copied. |

## License compliance notes

1. **Attribution**: DOWIS, dolly, CodeAlpaca require attribution — keep their
   citation strings (see `dataset_sources.py` and the papers/cards) in any
   distribution of the dataset.
2. **Share-alike**: dolly is CC-BY-SA-3.0; if dolly content is included, the
   combined dataset must be distributed under CC-BY-SA-3.0 (the fine-tuned
   LoRA adapter is generally considered a derivative work; if strict
   permissiveness is required, rebuild with `--target-total` after removing
   dolly from `EXTERNAL_QUOTA`).
3. **No NC/ND terms** are present in any used source.
4. Gated or unclear-licensed datasets (LMSYS-Chat-1M) are marked
   `LICENSE_REVIEW_REQUIRED` and excluded rather than silently used.
