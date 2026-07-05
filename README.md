# Joi — local voice assistant

Fully local voice pipeline on CachyOS: **faster-whisper** (STT, CUDA) →
**LlamaIndex FunctionAgent** on **Ollama qwen3:8b** (with real tools) →
**Piper** (TTS), with sentence-pipelined speech so Joi starts talking while
the reply is still being generated. Everything runs offline except the
weather tool (Open-Meteo) and Ollama on localhost.

## Run it (fish)

```fish
cd ~/Projects/joi-personal-assistant
source .venv/bin/activate.fish
python main.py
```

Press **Enter** to start recording, **Enter** again to stop, **Ctrl+C** to quit.
Each turn prints per-stage latencies:

```
⏱  STT 0.13s | LLM 1.02s (first token 0.35s) | first audio 1.21s | turn 4.50s
```

`first audio` is time-to-first-audio measured from agent start — the
perceived-latency metric the streaming TTS optimizes.

### Modes

| Mode | Command (fish) | Effect |
|---|---|---|
| normal | `python main.py` | qwen3:8b 100% on GPU (6.2 GB), fastest |
| gaming | `JOI_MODE=gaming python main.py` | CPU-only inference (`num_gpu=0`) — the game keeps all VRAM |
| plain fallback | `JOI_BACKEND=plain python main.py` | direct `ollama.chat`, no tools/agent |

**Automatic GPU handoff:** if you ask Joi to open a Steam game while in
normal mode, it finishes the reply, unloads the LLM from VRAM (`ollama
stop`) and switches itself to CPU inference — otherwise the game finds
only ~2 GB free and crashes before showing video. Restart Joi after
gaming to get GPU speed back. Start in gaming mode if you know you'll
be playing.

Verify what Ollama is doing at any time with `ollama ps` (PROCESSOR should
say `100% CPU` in gaming mode; CONTEXT should say `8192`).

### Private / per-machine settings (.env)

Copy `.env.example` to `.env` and edit it. `.env` is gitignored — private
paths and future API keys go there, never in code. Supported keys:
`JOI_MODE`, `JOI_BACKEND`, `JOI_MODEL`, `JOI_WHISPER_LANGUAGE`, `JOI_VOICE`.
Real environment variables always override `.env`.

### Smoke test (no mic needed)

```fish
python scripts/smoke_test.py
JOI_MODE=gaming python scripts/smoke_test.py
```

Checks imports, config, tools, TTS-to-wav, one real agent tool call,
streaming vs full-synthesis time-to-first-audio, and the live Ollama state.

## Architecture

```
main.py                     entry point (asyncio.run)
joi/
  assistant.py    VoiceAssistant — the turn loop + latency instrumentation
  agent.py        JoiAgent — ONE FunctionAgent with all registered tools
  capabilities/   ← THE extension point (see below)
    __init__.py   Capability dataclass + REGISTRY
    apps.py       open_app / list_installed_apps (.desktop scan)
    weather.py    get_weather (Open-Meteo)
  config.py       every tunable, incl. normal/gaming profiles + persona
  stt.py          Transcriber (faster-whisper; imports cuda_libs first)
  cuda_libs.py    ctypes preload of pip nvidia wheels (keep before whisper)
  tts.py          TTS (Piper) + StreamingSpeaker (sentence pipelining)
  sentences.py    SentenceBuffer — incremental sentence splitting
  llm.py          plain ollama.chat fallback (JOI_BACKEND=plain)
scripts/smoke_test.py
```

Single-agent by design (measured lower latency than multi-agent handoffs).
When Phase 3+ graduates to LlamaIndex `AgentWorkflow`, the only change point
is `JoiAgent.__init__` — see the comment there.

## How to add a new capability

Two steps, nothing else. Worked example — a `get_time` capability:

**1. Create `joi/capabilities/clock.py`:**

```python
"""Capability: tell the current date and time."""
from datetime import datetime

from joi.capabilities import Capability


def get_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("It is %H:%M on %A, %B %d.")


CAPABILITY = Capability(
    name="clock",
    description="Tell the current date and time.",
    tools=[get_time],
)
```

**2. Register it in `joi/capabilities/__init__.py`:**

```python
from joi.capabilities.clock import CAPABILITY as _clock

REGISTRY: list[Capability] = [
    _apps,
    _weather,
    _clock,   # <- the one new line
]
```

Done. The agent's system prompt and tool list are built from REGISTRY, so
Joi can now answer "what time is it?" by calling your function. Rules of
thumb: the **docstring is what the LLM reads** — write it as an instruction
("Get the current…"); tools take/return plain `str`; return sentences ready
to be spoken.

## Locked, empirically validated settings (do not change casually)

- `qwen3:8b`, `thinking=False` (first token ~0.15s vs 6.4s), `num_ctx=8192`
  (100% GPU; 16384 spills to CPU — never raise it).
- Whisper `small` / cuda / float16. `whisper_language="es"` in config if
  Spanglish drops verbs. Whisper's detected language is **display-only** —
  it misdetects even when transcribing correctly.
- Piper 1.4.2 API: `PiperVoice.load()` + `voice.synthesize()` yielding
  chunks with `.audio_int16_bytes` / `.sample_rate`
  (`synthesize_stream_raw` does not exist in this version).
- The Piper voice lives at `models/tts/en_US-lessac-medium.onnx` (not in
  git; download `.onnx` + `.onnx.json` from
  https://huggingface.co/rhasspy/piper-voices under `en/en_US/lessac/medium/`).
- `joi/cuda_libs.py` must be imported before faster-whisper (it preloads
  the pip-installed CUDA libs; no LD_LIBRARY_PATH needed, shell-agnostic).
