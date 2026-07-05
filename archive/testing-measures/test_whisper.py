"""Phase 0: STT quality with your real mic, in Spanish, English, and Spanglish."""
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SECONDS = 6

print("Loading model...")
model = WhisperModel("small", device="cuda", compute_type="float16")

tests = [
    "1) Say in ENGLISH: 'Open Firefox and tell me the weather for today'",
    "2) Di en ESPAÑOL: 'Ábreme el navegador y dime qué clima hace hoy'",
    "3) MIXED (how you actually talk): 'Oye, abre el code editor y dime qué tengo mal'",
]

for instruction in tests:
    input(f"\n{instruction}\nPress Enter, then speak ({SECONDS}s)...")
    audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    segments, info = model.transcribe(audio.flatten(), beam_size=5)
    print(f"  Detected language: {info.language} (p={info.language_probability:.2f})")
    for s in segments:
        print(f"  → {s.text}")