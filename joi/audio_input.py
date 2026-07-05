"""Records your voice. Beep to start, Enter to stop."""
import numpy as np
import sounddevice as sd


def play_beep(sample_rate: int = 16000, freq: int = 880,
              duration: float = 0.15, volume: float = 0.3) -> None:
    """Short beep so you know recording has started."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()


def record_until_enter(sample_rate: int = 16000) -> np.ndarray:
    """Record from the mic until Enter is pressed. Returns mono float32 audio."""
    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"[audio] {status}")
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="float32", callback=callback):
        input()  # waits here while the mic fills `frames`

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()