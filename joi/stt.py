"""Speech-to-text using faster-whisper (validated in Phase 0)."""
from joi import cuda_libs

cuda_libs.preload()

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "cuda",
                 compute_type: str = "float16", language: str | None = None):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio) -> tuple[str, str]:
        """audio: mono float32 array at 16 kHz. Returns (text, detected_language)."""
        segments, info = self.model.transcribe(audio, beam_size=5, language=self.language)
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language