from transformers import pipeline

MODEL = "openai/whisper-base"

print("Loading Whisper...")

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL,
)

print("Transcribing...")

result = pipe(
    "test.wav",
    generate_kwargs={
        "language": "english",
        "task": "transcribe",
    },
)

print("\nTRANSCRIPTION:")
print(result["text"])
