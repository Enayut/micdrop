from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

print("Whisper loaded successfully.")

