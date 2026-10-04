# MicDrop

Hold a hotkey, speak, release. Your words come back as either **cleaned-up text**
or a **finished artifact**, pasted wherever your cursor is.

MicDrop is a Linux desktop utility for Fedora with GNOME/Wayland. It records
from the microphone, transcribes with Whisper, processes the text with a local
or cloud language model, and pastes the result into the focused window using
`wl-copy` + `ydotool`.

## The two modes

| Hotkey     | Mode      | What it does                                                                                                     |
| ---------- | --------- | ---------------------------------------------------------------------------------------------------------------- |
| Right Ctrl | `REFINE`  | Cleans up **your words**. Removes filler and stutters, fixes obvious transcription errors, grammar, punctuation and capitalization, keeps technical terminology. Preserves your meaning. Never answers you. |
| Right Alt  | `GENERATE`| Carries out **your instruction** and returns the artifact itself: the email, the code, the explanation. Pasteable as-is. |

The difference is the whole point:

```
speech:  "uh can you make this email a little more professional and say that I'll be available tomorrow"

REFINE   → "Can you make this email a little more professional and say that I'll be available tomorrow?"

speech:  "write a professional email to my professor asking for an extension"

GENERATE → a complete professional email, ready to paste
```

The mode is explicit application state. The hotkey selects it *before* anything
is recorded, and it is carried through every stage. It is never guessed from the
words that were spoken, and there is no third or "auto" mode.

## Install

Requires Python 3.11+ on Fedora.

```bash
git clone https://github.com/enayut/micdrop
cd micdrop
python3 -m venv .venv
./.venv/bin/pip install .
```

That gives you a `micdrop` command. You do not need `python -m micdrop`.

### System packages

```bash
sudo dnf install wl-clipboard ydotool
```

### `/dev/uinput` access

`ydotool` needs to write to a virtual input device, which normally requires
membership in the `input` group. **This is the one step that needs `sudo`, and
it is yours to run — MicDrop never asks for root.**

```bash
echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' | \
  sudo tee /etc/udev/rules.d/80-uinput.rules

sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput

getent group input          # your user must be listed; log out and back in
```

### ydotool daemon

Run the daemon as your user, not as root. MicDrop must not require root.

```bash
sudo systemctl disable --now ydotool.service     # disable any system-level one
systemctl --user enable --now ydotool.service
systemctl --user status ydotool.service          # creates ~/.ydotool_socket
```

### Wayland clipboard

On GNOME/Wayland, install `wl-clipboard` for `wl-copy`. If clipboard access is
restricted, install the GNOME clipboard extension or allow MicDrop under
**Settings → Privacy → Clipboard**.

## Setup

```bash
micdrop setup
```

`setup` is interactive, checks every dependency, writes a `micdrop.toml`, and
tells you exactly what to do about anything that is missing. It never makes
system changes for you — it prints the commands and lets you decide.

```
MicDrop Setup

  ✓ Linux detected
  ✓ Wayland detected
  ✓ GNOME detected
  ✓ Microphone detected
  ✓ Whisper available
  ✓ Refinement model available (Qwen/Qwen3-0.6B)
  ✓ Generation model available (Qwen/Qwen3-0.6B)
  ✓ evdev available
  ✓ Input permissions OK
  ✓ /dev/uinput accessible
  ✓ wl-copy available
  ✓ ydotool available
  ✓ ydotool daemon running

  ✓ Configuration written to /home/you/micdrop/micdrop.toml

MicDrop is ready. Start it with:  micdrop run
```

## Commands

| Command                | Interactive | Purpose                                                          |
| ---------------------- | ----------- | ---------------------------------------------------------------- |
| `micdrop setup`        | yes         | First-run wizard: checks dependencies, writes the config file    |
| `micdrop run`          | no          | The app. Loads models once, then waits for hotkeys              |
| `micdrop doctor`       | no          | Non-interactive health report, one line per dependency           |
| `micdrop status`       | no          | What is configured, and whether it is ready to run               |
| `micdrop config`       | no          | Print the effective configuration, or `--example` for a template |
| `micdrop models`       | no          | List providers and which are usable right now                    |
| `micdrop test`         | no          | Run the dependency and pipeline checks in one go                 |
| `micdrop self-test`    | no          | Verify clipboard + paste actually work (pastes demo text)        |
| `micdrop hotkey-test`  | no          | Round-trip both hotkeys through evdev; no models, no paste       |
| `micdrop record-test`  | no          | Record N seconds and report RMS/peak levels                      |
| `micdrop transcribe-test [file]` | no | Transcribe a WAV file with Whisper                     |
| `micdrop refine-test --mode refine\|generate "text"` | no | Run the text processor only   |

The everyday set is `setup`, `doctor`, `run`. You should not need the CLI while
MicDrop is running.

### Running it

```bash
micdrop run
```

Models load once at startup, then the process sits idle until you press a
hotkey. Stop it with `Ctrl+C`. Startup looks like this:

```
Mode REFINE   -> Qwen3 local (Qwen/Qwen3-0.6B) [refine]
Mode GENERATE -> Qwen3 local (Qwen/Qwen3-0.6B) [generate]
Loading Whisper model openai/whisper-base...
Loading Qwen/Qwen3-0.6B (mode: refine)...
Model loaded.
Warming up GENERATE...
Keyboard: /dev/input/event3

MicDrop is listening. Hold Right Ctrl to REFINE, Right Alt to GENERATE.
Press Ctrl+C to stop.
```

## Configuration

MicDrop looks for a config file in this order, first match wins:

1. the path given with `--config`
2. `$MICRODROP_CONFIG`
3. `./micdrop.toml`
4. `~/.config/micdrop/config.toml`

Missing files fall back to defaults and unknown keys are ignored. Run
`micdrop config` to see what is actually in effect, and `micdrop setup` to
generate a file.

```toml
[hotkeys]
refine = "RIGHTCTRL"        # Right Ctrl -> REFINE
generate = "RIGHTALT"       # Right Alt  -> GENERATE
# device = "/dev/input/event3"   # omit to auto-detect

[models]
refine_provider = "local_qwen"     # per mode, chosen independently
generate_provider = "local_qwen"
refine_max_new_tokens = 128        # REFINE stays near the input length
generate_max_new_tokens = 512      # GENERATE has to fit a whole artifact

[models.local]
refine_model = "Qwen/Qwen3-0.6B"
generate_model = "Qwen/Qwen3-0.6B"

[models.groq]
refine_model = "llama-3.3-70b-versatile"
generate_model = "llama-3.3-70b-versatile"
temperature = 0.0
timeout = 30.0

[audio]
sample_rate = 16000
max_duration = 60.0        # safety cap; recording stops by itself

[refinement]
whisper_model = "openai/whisper-base"
language = "english"

[output]
paste_delay = 0.15
tool_timeout = 5.0

[debug]
enabled = false
```

None of these values are hardcoded into the pipeline. Hotkey names accept
`"RIGHTCTRL"`, `"KEY_RIGHTCTRL"` and `"Right Ctrl"` interchangeably. If the
configured device cannot be opened, MicDrop auto-detects a keyboard that has the
configured keys.

### Environment variables

| Variable                  | Effect                                            |
| ------------------------- | ------------------------------------------------- |
| `MICRODROP_CONFIG`        | Path to the config file                           |
| `MICRODROP_DEBUG=1`       | Turn on debug logging and artifact saving         |
| `MICRODROP_GROQ_API_KEY`  | Your Groq API key. Never stored in a config file  |

## Providers

Provider selection is **per mode**, so the two modes can run on different
backends. No pipeline code changes are needed to mix them.

```
TextProcessor            (abstract, bound to exactly one mode)
├── LocalQwenProcessor   provider = "local_qwen"   local, offline, free
├── GroqProcessor        provider = "groq"         cloud, needs a key + network
└── PassthroughProcessor provider = "none"        raw transcript, no model
```

Mixed configuration — local for one mode, cloud for the other:

```toml
[models]
refine_provider = "local_qwen"
generate_provider = "groq"
```

Adding a provider is one `TextProcessor` class plus one entry in the `PROVIDERS`
registry in `micdrop/refinement/__init__.py`. The pipeline, the CLI and the
config loader do not change.

### Local models

`local_qwen` runs `Qwen/Qwen3-0.6B` through `transformers` and `peft`. It needs
no network once the weights are cached, and it works fully offline:

```bash
# one-time download
./.venv/bin/python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"
```

Both modes share one loaded model instance when they name the same model, so
running both modes locally does not double your memory use.

If a LoRA adapter is present it is used automatically; otherwise MicDrop runs
the stock model. **MicDrop does not include a fine-tuned adapter** — output
quality reflects the stock base model plus the prompts.

### Groq (cloud)

Groq is **implemented** using the official `groq` Python client. To use it:

```bash
export MICRODROP_GROQ_API_KEY="gsk_..."
```

```toml
[models]
generate_provider = "groq"

[models.groq]
generate_model = "llama-3.3-70b-versatile"
```

The key is read from the environment at construction time. It is never written
to a config file, never logged, and never included in an error message or a
`repr`.

**If the key is missing, that mode fails loudly.** It does not fall back to a
local model or to the other mode's provider — a silent substitution would paste
the wrong kind of text. If the network is down or the API rejects the request,
the failure is reported and the hotkey listener keeps running.

MicDrop works fully offline with both modes on `local_qwen`; the Groq path is
the only part that needs a network.

## Privacy

- **Microphone audio is never logged.** It is captured into memory and handed
  straight to Whisper.
- **Nothing is written to disk** unless `[debug] enabled = true`. With debug off,
  recordings are never persisted.
- **Transcripts are not logged or saved by default.** The logs record the mode
  and the character count only:

  ```
  Transcription: 63 chars (hidden; set debug.enabled=true to show text).
  ```

- With `[debug] enabled = true`, the recording (`.wav`) and the transcript
  (`.txt`) are written to `debug_dir` for inspection. This is opt-in and is the
  only path that persists either.

## Architecture

The core is platform-independent. Platform specifics live behind interfaces and
are selected at exactly one boundary.

```text
micdrop/
├── app/                       # core: no platform imports
│   ├── pipeline.py            #   orchestration + per-stage error handling
│   ├── modes.py               #   Mode enum + the REFINE/GENERATE contract
│   ├── config.py              #   dataclass config, TOML loading, env overrides
│   ├── errors.py              #   MicDropError + mode-routing errors
│   └── diagnostics.py         #   shared, platform-agnostic diagnostic engine
├── audio/                     #   AudioRecorder  (ABC) + sounddevice impl
├── transcription/             #   SpeechRecognizer (ABC) + Whisper impl
├── refinement/                #   TextProcessor (ABC) + provider impls
│   ├── prompts.py             #   the two system prompts — the product definition
│   ├── router.py              #   ModeProcessorRouter: the mode safety invariant
│   ├── local_qwen.py
│   ├── groq.py
│   └── passthrough.py
├── output/                    #   OutputAdapter (ABC)
├── platform/                  # the ONE dispatch boundary
│   ├── __init__.py            #   PlatformBackend registry + select_platform()
│   ├── base.py                #   HotkeyManager (ABC)
│   └── linux/                 #   the only implemented platform today
│       ├── keyboard.py        #     LinuxHotkeyManager  (evdev)
│       ├── clipboard.py       #     LinuxOutputAdapter  (wl-copy + ydotool)
│       └── diagnostics.py     #     Linux-specific checks
├── diagnostics/               # thin re-export so callers need no platform import
├── cli/                       # argparse entry point, wiring, output formatting
└── tests/                     # stdlib unittest
```

`micdrop/app/pipeline.py` imports no platform module. It talks only to
`AudioRecorder`, `SpeechRecognizer`, `TextProcessor` and `OutputAdapter`.
`micdrop/tests/test_architecture.py` walks the import graph and fails if that
ever stops being true, so the boundary cannot rot silently.

### Mode safety invariant

> REFINE must never silently fall back to GENERATE, and GENERATE must never be
> handled with the REFINE prompt.

Three independent guards enforce this:

1. A `TextProcessor` is **constructed for one mode** and raises
   `ModeMismatchError` if asked for the other.
2. `ModeProcessorRouter` only dispatches to a processor whose bound mode matches,
   and raises `ModeProcessorUnavailableError` if a mode has no processor. It
   never borrows the other one.
3. `create_processor(config, mode)` takes the mode as a required argument.

A violation fails that hotkey cycle loudly and delivers nothing. A wrong paste
is worse than no paste.

## Platform support

**MicDrop currently supports Linux only.** Windows and macOS are **not
supported** — there is no code for them and no partial support to fall into.

The architecture is arranged so adding them is additive rather than a rewrite:
create `platform/windows/` (or `macos/`) with its own `HotkeyManager`,
`OutputAdapter` and diagnostics module, then register a `PlatformBackend` in
`micdrop/platform/__init__.py`. Selection happens in one place — `select_platform()` —
so there is no `if linux:` branching scattered through the application. The core
pipeline, the two modes, the prompts and the providers do not change.

## Tests

```bash
./.venv/bin/python -m unittest discover -s micdrop/tests -t .
```

Unit tests need no hardware, no models and no network. Cloud provider tests
inject a fake client, so nothing calls the real Groq API.

Coverage includes: CLI commands and argument parsing, config loading and
migration, per-mode provider selection, the Groq provider, missing API key,
Groq failure handling, offline local operation, model lifecycle (loaded once),
maximum recording duration, setup and doctor diagnostics, the packaging entry
point, REFINE semantics, GENERATE semantics, mode safety, the failure matrix,
privacy, and the architecture boundaries.

## Troubleshooting

Run `micdrop doctor` first — it names the failing component and what to do.

### `ydotool` reports `Permission denied`

```bash
ls -ln /dev/uinput          # should be group `input`, mode 0660
id                          # your user must be in that group
sudo systemctl --user restart ydotool.service
```

### Hotkey does nothing

```bash
micdrop hotkey-test         # verifies both hotkeys end to end
micdrop doctor              # shows which keyboard device was chosen
```

On GNOME, Right Ctrl and Right Alt sometimes arrive as `KEY_RIGHTCTRL` /
`KEY_RIGHTALT` only from a real keyboard; a virtual device may report different
names. `micdrop doctor` prints what was detected.

### Clipboard updates but nothing is pasted

The text is on the clipboard; paste it manually with `Ctrl+V`. Check
`systemctl --user status ydotool.service` and that the socket exists at
`/run/user/$UID/.ydotool_socket`.

### Generation is slow

`local_qwen` runs on CPU/GPU locally, and GENERATE writes up to
`generate_max_new_tokens` (512 by default). That is expected. For faster
generation, point GENERATE at `groq` and leave REFINE local.

### It transcribes silence

The microphone is picking up room tone. Check with `micdrop record-test --duration 5`;
if the RMS is very low, your input device or PipeWire default source is wrong.

## License

MIT
