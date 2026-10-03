# micdrop — Linux/Fedora voice-to-prompt controller

Hold **Right Ctrl** to record from the microphone. Release it to transcribe the recording with Whisper, then automatically copy the resulting text to the clipboard and paste it at the current cursor position.

## Pipeline

```text
Right Ctrl held
    ↓
Record microphone
    16 kHz mono / sounddevice
    ↓
Right Ctrl released
    ↓
Whisper transcription
    openai/whisper-base / English
    ↓
Copy to clipboard
    wl-copy
    ↓
Paste at current cursor
    ydotool / virtual Ctrl+V
```

An LLM prompt-refinement stage is planned between Whisper and the clipboard/paste stage.

## Files

| File                 | Purpose                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `main.py`            | Push-to-talk listener using evdev and the full recording → transcription → paste pipeline |
| `clipboard_paste.py` | Clipboard copy, virtual paste, dependency checks, and self-test CLI                       |
| `record.py`          | Standalone microphone recording test                                                      |
| `transcribe.py`      | Standalone Whisper transcription test                                                     |

## Fedora setup

Install the required system packages:

```bash
sudo dnf install wl-clipboard ydotool
```

`micdrop` uses `ydotool` to generate the virtual `Ctrl+V` keypress on GNOME Wayland. The `ydotoold` daemon must run as a user service.

Configure `/dev/uinput` so members of the `input` group can access it:

```bash
echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' | \
sudo tee /etc/udev/rules.d/80-uinput.rules

sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput
```

Make sure the current user belongs to the `input` group:

```bash
getent group input
id
```

If necessary:

```bash
sudo usermod -aG input "$USER"
```

A logout/login may be required after adding the user to the group.

Disable the system-level `ydotool` service:

```bash
sudo systemctl disable --now ydotool.service
```

Enable the user-level daemon:

```bash
systemctl --user enable --now ydotool.service
```

Verify it:

```bash
systemctl --user status ydotool.service
```

The daemon should create:

```text
/run/user/$UID/.ydotool_socket
```

## Verify

First test the clipboard/paste subsystem:

```bash
python clipboard_paste.py
```

It should report:

```text
Clipboard/paste dependencies: OK
Clipboard updated.
Pasted at cursor.
Self-test OK.
```

Then run the complete voice pipeline:

```bash
python main.py
```

Hold **Right Ctrl**, speak, and release **Right Ctrl**. The transcription should be pasted into the currently focused application.

## Troubleshooting

### `ydotool` reports `Permission denied`

Check the permissions on `/dev/uinput`:

```bash
ls -ln /dev/uinput
```

The device should be accessible by the `input` group, for example:

```text
crw-rw---- ... 0 104 ... /dev/uinput
```

Also verify that the current user belongs to the group:

```bash
id
```

If the udev rule has been installed but the existing `/dev/uinput` device still has the wrong ownership, apply the permissions immediately:

```bash
sudo chown root:input /dev/uinput
sudo chmod 660 /dev/uinput
```

Then restart the user daemon:

```bash
systemctl --user restart ydotool.service
```

### User daemon is not running

Check:

```bash
systemctl --user status ydotool.service
```

View recent logs:

```bash
journalctl --user -u ydotool.service -n 30 --no-pager
```

Check that the user socket exists:

```bash
ls -l /run/user/$UID/.ydotool_socket
```

### Clipboard updates but nothing is pasted

The text should still be available on the clipboard and can be pasted manually with `Ctrl+V`.

First verify that `ydotoold` is running and that the socket exists. Some applications may also behave differently with synthetic keyboard input.

## Current status

The following components have been tested successfully:

* Microphone recording with `sounddevice`
* 16 kHz mono audio capture
* Global Right Ctrl detection using `evdev`
* Whisper transcription using `openai/whisper-base`
* Wayland clipboard integration using `wl-copy`
* Virtual `Ctrl+V` using `ydotool`
* End-to-end clipboard/paste self-test

## Planned

The next major stage is local LLM prompt refinement:

```text
Speech
  ↓
Whisper
  ↓
Raw transcription
  ↓
Local LLM
  ↓
Clean / structured prompt
  ↓
Clipboard
  ↓
Paste at cursor
```

The intended local model is Qwen3 0.6B, allowing the system to turn natural spoken instructions into cleaner prompts before pasting them.
