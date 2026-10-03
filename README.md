# micdrop — voice-to-prompt controller for Fedora (GNOME/Wayland)

Hold a hotkey to record from the microphone, release it, and the audio is
transcribed with Whisper, cleaned up by a local LLM, then copied to the
clipboard and pasted at the current cursor.

Two modes, selected by the hotkey you hold:

| Hotkey      | Mode      | What it does                                                                                      |
| ----------- | --------- | ------------------------------------------------------------------------------------------------- |
| Right Ctrl  | REFINE    | Fixes transcription errors, grammar, punctuation and filler. Preserves meaning, invents nothing.    |
| Right Alt   | GENERATE  | Interprets the request and turns it into a useful, self-contained prompt. May expand/restructure.  |

The mode is explicit application state — it is chosen by the hotkey and passed
through the whole pipeline; it is never guessed from the text.

## Pipeline

```text
hotkey press (mode)
   ↓
AudioRecorder            sounddevice, 16 kHz mono, in-memory
   ↓  hotkey release
SpeechRecognizer          openai/whisper-base
   ↓
TextProcessor            Qwen/Qwen3-0.6B + LoRA adapter (local)
   ↓
OutputAdapter            wl-copy → ydotool Ctrl+V
```

A failure in any stage is logged clearly and skipped. The hotkey listener
stays alive and the next hotkey press works normally.

## Architecture

The core is platform independent; Linux specifics live behind interfaces.

```text
micdrop/
├── app/                    # core application logic (no platform imports)
│   ├── pipeline.py         #   orchestration + per-stage error handling
│   ├── modes.py            #   Mode enum: REFINE, GENERATE
│   ├── config.py           #   dataclass config, TOML loading, env overrides
│   └── errors.py           #   MicDropError + per-stage errors
├── audio/
│   ├── base.py             #   AudioRecorder   (ABC)
│   └── recorder.py         #   SoundDeviceRecorder
├── transcription/
│   ├── base.py             #   SpeechRecognizer (ABC)
│   └── whisper.py          #   WhisperRecognizer
├── refinement/
│   ├── base.py             #   TextProcessor   (ABC)
│   ├── local_qwen.py       #   LocalQwenProcessor (Qwen3 + LoRA, default)
│   ├── passthrough.py      #   PassthroughProcessor (provider "none")
│   └── __init__.py         #   create_processor() provider registry
├── output/
│   └── base.py             #   OutputAdapter   (ABC)
├── platform/
│   ├── base.py             #   HotkeyManager   (ABC)
│   └── linux/
│       ├── keyboard.py     #   LinuxHotkeyManager  (evdev)
│       └── clipboard.py    #   LinuxOutputAdapter  (wl-copy + ydotool)
├── cli/main.py             # command line entry point / wiring
└── tests/                  # unit tests (stdlib unittest)
```

`micdrop/app/pipeline.py` never imports evdev, ydotool, wl-copy or any other
platform module — it only talks to `AudioRecorder`, `SpeechRecognizer`,
`TextProcessor` and `OutputAdapter`. Adding Windows/macOS support means adding
`WindowsHotkeyManager` / `MacHotkeyManager` and
`WindowsOutputAdapter` / `MacOutputAdapter` plus a platform switch in
`micdrop/cli/main.py`; the core pipeline does not change.

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .        # or: pip install -r requirements
```

## Fedora setup

```bash
sudo dnf install wl-clipboard ydotool
```

Configure `/dev/uinput` so members of the `input` group can access it:

```bash
echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' | \
sudo tee /etc/udev/rules.d/80-uinput.rules

sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput
getent group input   # your user must be in it; log out/in after adding
```

Enable the user-level `ydotoold` daemon (a system-level one must be disabled —
the daemon needs `/dev/uinput` access as your user):

```bash
sudo systemctl disable --now ydotool.service
systemctl --user enable --now ydotool.service
systemctl --user status ydotool.service   # creates /run/user/$UID/.ydotool_socket
```

## Usage

```bash
./.venv/bin/python -m micdrop             # start the push-to-talk listener
./.venv/bin/python -m micdrop run         # same thing, explicitly
```

Install it to get a `micdrop` command: `./.venv/bin/pip install -e .`

## Commands

| Command                | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `micdrop run`          | The app: hotkey listener + full pipeline                               |
| `micdrop self-test`    | Clipboard/paste dependency check + demo copy/paste                     |
| `micdrop hotkey-test`  | Round-trips both hotkeys through evdev using `ydotool` (no models, no paste) |
| `micdrop doctor`       | Checks packages, audio devices, keyboard device, clipboard tooling     |
| `micdrop record-test`  | Records N seconds, reports RMS/peak levels                            |
| `micdrop transcribe-test [file]` | Transcribes a WAV file with Whisper                          |
| `micdrop refine-test --mode refine\|generate "text"` | Runs the text processor only       |
| `micdrop config`       | Shows the effective configuration                                      |
| `micdrop config --example` | Prints an example `micdrop.toml`                                   |

## Configuration

Copy `micdrop.toml.example` to `micdrop.toml` (or
`~/.config/micdrop/config.toml`). All values are optional; missing keys fall
back to defaults, unknown keys are ignored.

```toml
[hotkeys]
refine = "KEY_RIGHTCTRL"
generate = "KEY_RIGHTALT"
device = "/dev/input/event3"     # auto-detected if unreadable

[refinement]
provider = "local_qwen"
model_path = "Qwen/Qwen3-0.6B"

[debug]
enabled = false                  # true: keep recordings in ./debug/
```

Hotkey names are evdev key names, so they are portable across keyboards. If the
configured device cannot be opened, MicDrop auto-detects a keyboard that has
the configured keys.

Environment overrides: `MICDROP_CONFIG` (config path), `MICDROP_DEBUG=1`,
`MICDROP_ADAPTER` (LoRA adapter path).

**API keys are never stored in source or config files.** The future Groq
provider reads its key from `$GROQ_API_KEY` at runtime.

## The refinement model

`LocalQwenProcessor` loads `Qwen/Qwen3-0.6B` and, if present, a LoRA adapter
from `$MICDROP_ADAPTER` or `./adapters/prompt-refiner-lora`. Adapter resolution
and `peft` fallback behaviour are unchanged from the original
`prompt_refiner.py`.

A Groq/API provider only needs a new `TextProcessor` implementation plus one
`elif` in `micdrop/refinement/__init__.py:create_processor()`.

## Recordings and privacy

Audio is captured into memory and passed to Whisper from there. **Nothing is
written to disk** unless `[debug] enabled = true`, in which case recordings are
kept in `debug/` for inspection.

## Tests

```bash
./.venv/bin/python -m unittest discover -s micdrop/tests -t .   # 42 unit tests
./.venv/bin/python -m micdrop self-test                        # clipboard/paste
./.venv/bin/python -m micdrop doctor
```

Unit tests cover the platform-independent core: REFINE/GENERATE routing, empty
transcription, transcription/processor/output failures, a successful pipeline,
recording failures, hotkey→pipeline wiring, and configuration loading. They
need no hardware and no models.

## Troubleshooting

### `ydotool` reports `Permission denied`

```bash
ls -ln /dev/uinput          # should be group `input`, mode 0660
id                          # you must be in the `input` group
sudo systemctl --user restart ydotool.service
```

### Hotkey does nothing

```bash
./.venv/bin/python -m micdrop doctor      # checks the keyboard device
./.venv/bin/python -m micdrop hotkey-test # verifies both hotkeys end to end
```

### Clipboard updates but nothing is pasted

The text is still on the clipboard; paste it manually with `Ctrl+V`. Check
`systemctl --user status ydotool.service` and that the socket exists at
`/run/user/$UID/.ydotool_socket`.