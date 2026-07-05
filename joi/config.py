"""Central settings for Joi. Every locked value here came from a Phase 0/2 test.

Single responsibility: hold ALL tunable values in one place, including the
normal/gaming mode profiles. No other module reads environment variables.
"""
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file(path: str = os.path.join(ROOT, ".env")) -> None:
    """Tiny stdlib .env loader: KEY=VALUE lines become environment defaults.
    Real environment variables always win (so `JOI_MODE=gaming python main.py`
    overrides whatever .env says). .env is gitignored — put machine-specific
    paths and future API keys there; commit documented defaults to .env.example."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except FileNotFoundError:
        pass


_load_env_file()

# "normal" = model on GPU (fast). "gaming" = model on CPU (frees the GPU for the game).
MODE = os.environ.get("JOI_MODE", "normal")

# "agent" = LlamaIndex FunctionAgent with tools (default).
# "plain" = direct ollama.chat, no tools — fallback if the agent path breaks.
BACKEND = os.environ.get("JOI_BACKEND", "agent")


@dataclass
class Config:
    mode: str = MODE
    backend: str = BACKEND

    # ---- LLM (locked in Phase 0) ----
    model: str = os.environ.get("JOI_MODEL", "qwen3:8b")
    think: bool = False        # think=False -> first token ~0.15s (vs 6.4s on)
    num_ctx: int = 8192        # fits 100% on GPU; 16384 spills to CPU — do not raise
    keep_alive: str = "30m"    # keep the model loaded between commands
    request_timeout: float = 120.0

    # ---- STT (faster-whisper) ----
    whisper_model: str = "small"        # try "medium" if Spanglish drops words
    whisper_device: str = "cuda"
    whisper_compute: str = "float16"
    whisper_language: str | None = os.environ.get("JOI_WHISPER_LANGUAGE") or None
    # (set JOI_WHISPER_LANGUAGE=es in .env if Spanglish drops verbs)

    # ---- TTS (Piper) ----
    piper_voice: str = os.environ.get(
        "JOI_VOICE", os.path.join(ROOT, "models", "tts", "en_US-lessac-medium.onnx"))

    # ---- Audio ----
    sample_rate: int = 16000

    # ---- Persona (placeholder; real persona system is Phase 6) ----
    system_prompt: str = (
        "You are Joi, a concise local voice assistant. "
        "Reply in at most two short sentences, written to be spoken aloud. "
        "Do not use lists, markdown, or emoji."
    )

    @property
    def num_gpu(self) -> int | None:
        """Ollama num_gpu option: 0 in gaming mode (CPU-only), default otherwise."""
        return 0 if self.mode == "gaming" else None

    @property
    def ollama_extra_options(self) -> dict:
        """Extra Ollama options merged into every request (used by the agent
        path via the LlamaIndex wrapper's additional_kwargs)."""
        return {"num_gpu": 0} if self.mode == "gaming" else {}
