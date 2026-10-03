import time
import threading

import evdev
import sounddevice as sd
import soundfile as sf
from transformers import pipeline

try:
    from clipboard_paste import copy_and_paste, report_dependencies
    from prompt_refiner import refine_prompt
except ImportError:
    copy_and_paste = None
    report_dependencies = None
    refine_prompt = None


# -----------------------------
# Configuration
# -----------------------------

KEYBOARD = "/dev/input/event3"
HOTKEY = evdev.ecodes.KEY_RIGHTCTRL

SAMPLE_RATE = 16000
AUDIO_FILE = "test.wav"


# -----------------------------
# Load Whisper
# -----------------------------

print("Loading Whisper...")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

print("Whisper loaded.\n")

# Report clipboard/paste tooling problems once, at startup.
if report_dependencies is not None:
    report_dependencies()
else:
    print(
        "clipboard_paste.py not found; "
        "transcribed text will not be pasted automatically."
    )


# -----------------------------
# Recording state
# -----------------------------

recording = False
frames = []
stream = None


def audio_callback(indata, frames_count, time_info, status):
    if status:
        print("Audio:", status)

    if recording:
        frames.append(indata.copy())


# -----------------------------
# Start recording
# -----------------------------

def start_recording():
    global recording, frames, stream

    if recording:
        return

    frames = []
    recording = True

    print("\nRecording...")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )

    stream.start()


# -----------------------------
# Stop recording
# -----------------------------

def stop_recording():
    global recording, stream

    if not recording:
        return

    recording = False

    stream.stop()
    stream.close()

    print("Recording stopped.")

    if not frames:
        print("No audio recorded.")
        return

    audio = __import__("numpy").concatenate(frames, axis=0)

    sf.write(
        AUDIO_FILE,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    print("Saved:", AUDIO_FILE)
    print("Transcribing...")

    result = pipe(
        AUDIO_FILE,
        return_timestamps=True,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
        },
    )

    text = result["text"].strip()

    print("\nTRANSCRIPTION:")
    print(text)
    print()

    print("\nRefining prompt with Qwen3...")
    refined_text = text
    if refine_prompt is not None:
        refined_text = refine_prompt(text)
    else:
        print("Cant refine prompt")

    print("\nREFINED PROMPT:")
    print(refined_text)
    print()

    # Pipeline: Whisper -> clipboard -> paste at current cursor.
    # (An LLM refinement stage will slot in here later.)
    if copy_and_paste is not None:
        copy_and_paste(refined_text)
    else:
        print("Skipping paste: clipboard_paste.py not found.")


# -----------------------------
# Keyboard listener
# -----------------------------

def keyboard_listener():

    device = evdev.InputDevice(KEYBOARD)

    print("Keyboard:", device.name)
    print()
    print("Hold RIGHT CTRL to record.")
    print("Release RIGHT CTRL to transcribe.")
    print("Press Ctrl+C in the terminal to quit.")
    print()

    for event in device.read_loop():

        if event.type != evdev.ecodes.EV_KEY:
            continue

        key_event = evdev.categorize(event)

        if key_event.scancode != HOTKEY:
            continue

        # Key pressed
        if key_event.keystate == evdev.KeyEvent.key_down:
            start_recording()

        # Key released
        elif key_event.keystate == evdev.KeyEvent.key_up:
            stop_recording()


# -----------------------------
# Main
# -----------------------------

try:
    keyboard_listener()

except KeyboardInterrupt:
    print("\nExiting...")