import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
DURATION = 10
OUTPUT = "test.wav"

print("Recording...")
print("Speak normally into your microphone.")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

# Check recording level
rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))

print("\nRecording finished.")
print(f"RMS level:  {rms:.6f}")
print(f"Peak level: {peak:.6f}")

if rms < 0.001:
    print("WARNING: Audio level is extremely low.")
elif rms < 0.01:
    print("WARNING: Audio level is quite low.")
else:
    print("Audio level looks reasonable.")

sf.write(
    OUTPUT,
    audio,
    SAMPLE_RATE,
    subtype="PCM_16",
)

print(f"Saved: {OUTPUT}")
