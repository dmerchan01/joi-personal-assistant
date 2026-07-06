"""The orchestrator: record -> transcribe -> agent -> speak, with latencies.

Single responsibility: own the turn loop and the per-stage latency
instrumentation (a hard requirement — do not remove). It knows nothing about
how tools work (joi.agent) or how audio is produced (joi.tts).

The loop is async because FunctionAgent.run is async. Audio recording stays
a blocking input() — nothing else needs the event loop mid-recording.

Latency metrics printed each turn:
  STT          — transcription time
  first token  — agent start -> first LLM text delta
  LLM          — agent start -> reply fully generated (incl. tool calls)
  first audio  — agent start -> first samples playing (time-to-first-audio;
                 the perceived-latency metric the streaming TTS optimizes)
  turn         — agent start -> playback finished
"""
import asyncio
import subprocess
import time

from joi import events
from joi.config import Config
from joi.audio_input import play_beep, record_until_enter
from joi.stt import Transcriber
from joi.sentences import SentenceBuffer
from joi.tts import TTS, StreamingSpeaker


class VoiceAssistant:
    """Owns the models and runs the voice loop."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.stt: Transcriber | None = None
        self.tts: TTS | None = None
        self.agent = None   # JoiAgent (agent backend) or LLM (plain backend)
        # set when a Steam game launches mid-turn; handled after the turn
        # ends (the turn's final reply still needs the model on GPU)
        self._pending_gpu_release: str | None = None
        events.GAME_LAUNCHED.append(self._on_game_launched)

    async def start(self) -> None:
        """Load Whisper, Piper and the agent in parallel, and warm the Ollama
        model so the first real turn doesn't pay the load cost."""
        cfg = self.cfg
        print("Loading models (first run downloads Whisper weights)...")

        if cfg.mode == "normal":
            self._ensure_model_not_stuck_on_cpu()

        def load_stt() -> Transcriber:
            return Transcriber(cfg.whisper_model, cfg.whisper_device,
                               cfg.whisper_compute, cfg.whisper_language)

        def load_tts() -> TTS:
            return TTS(cfg.piper_voice)

        if cfg.backend == "plain":
            from joi.llm import LLM
            self.agent = LLM(cfg.model, cfg.think, cfg.num_ctx, cfg.num_gpu,
                             cfg.system_prompt, cfg.keep_alive)
            self.stt, self.tts = await asyncio.gather(
                asyncio.to_thread(load_stt), asyncio.to_thread(load_tts),
            )
        else:
            from joi.agent import JoiAgent
            self.agent = JoiAgent(cfg)
            self.stt, self.tts, _ = await asyncio.gather(
                asyncio.to_thread(load_stt),
                asyncio.to_thread(load_tts),
                self.agent.warm_up(),
            )
            caps = ", ".join(c.name for c in self.agent.capabilities)
            print(f"Capabilities: {caps}")

        # reminders fire from a timer thread; TTS.speak serializes on
        # PLAYBACK_LOCK so they never talk over an in-progress reply
        from joi.capabilities import reminders
        reminders.set_speaker(self.tts.speak)

        print(f"\nJoi ready  |  mode: {cfg.mode}  |  backend: {cfg.backend}"
              f"  |  model: {cfg.model}")
        print("Press Enter to start talking, then Enter again to stop. "
              "Ctrl+C to quit.\n")

    async def run(self) -> None:
        """Main loop: one voice turn per iteration until Ctrl+C."""
        await self.start()
        while True:
            try:
                await self._turn()
            except KeyboardInterrupt:
                print("\nBye.")
                break

    async def _turn(self) -> None:
        cfg = self.cfg
        input("[Enter to start recording] ")
        play_beep(cfg.sample_rate)
        print("● recording... (Enter to stop)")
        audio = record_until_enter(cfg.sample_rate)

        if audio.size == 0:
            print("(no audio captured)\n")
            return

        t0 = time.perf_counter()
        text, lang = self.stt.transcribe(audio)
        t_stt = time.perf_counter() - t0
        if not text:
            print("(nothing transcribed)\n")
            return
        # lang is display-only: Whisper's detection is unreliable (measured).
        print(f"You [{lang}]: {text}")

        if cfg.backend == "plain":
            m = await self._respond_plain(text)
        else:
            m = await self._respond_agent(text)

        first_tok = f"{m['first_token']:.2f}s" if m["first_token"] else "-"
        first_aud = f"{m['first_audio']:.2f}s" if m["first_audio"] else "-"
        print(f"⏱  STT {t_stt:.2f}s | LLM {m['llm']:.2f}s (first token {first_tok})"
              f" | first audio {first_aud} | turn {m['turn']:.2f}s\n")

        if self._pending_gpu_release:
            self._release_gpu()

    def _ensure_model_not_stuck_on_cpu(self) -> None:
        """After the game-launch GPU handoff (or a gaming-mode session) the
        model may still be loaded CPU-only with a long keep_alive. A normal-
        mode request does NOT force Ollama to move it back to GPU — it happily
        reuses the CPU instance. Unload it so warm-up reloads it on the GPU."""
        try:
            ps = subprocess.run(["ollama", "ps"], capture_output=True,
                                text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        line = next((ln for ln in ps.splitlines() if self.cfg.model in ln), "")
        if "100% CPU" in line:  # partial spill just means VRAM is tight — leave it
            print(f"  [gpu] {self.cfg.model} was loaded on CPU — reloading on GPU")
            subprocess.run(["ollama", "stop", self.cfg.model],
                           capture_output=True, timeout=30)

    def _on_game_launched(self, app_name: str) -> None:
        """Called (synchronously, mid-turn) by the apps capability when a
        Steam game starts. Just flag it; _turn() acts once the reply is done."""
        if self.cfg.mode != "gaming":  # gaming mode is already CPU-only
            self._pending_gpu_release = app_name

    def _release_gpu(self) -> None:
        """A game needs the VRAM the LLM is holding (6.2 GB of 8 GB): unload
        it from Ollama and switch Joi to CPU-only inference (like gaming
        mode) until restarted."""
        game = self._pending_gpu_release
        self._pending_gpu_release = None
        if hasattr(self.agent, "to_cpu"):
            self.agent.to_cpu()          # JoiAgent
        else:
            self.agent.options["num_gpu"] = 0   # plain LLM fallback
        subprocess.run(["ollama", "stop", self.cfg.model],
                       capture_output=True, timeout=30)
        print(f"  [gpu] {game} needs the VRAM — model unloaded, "
              "Joi now runs on CPU (restart Joi to go back to GPU)\n")

    async def _respond_agent(self, text: str) -> dict:
        """Agent turn with sentence-pipelined speech: sentences are spoken
        while the LLM is still generating the rest of the reply."""
        t0 = time.perf_counter()
        speaker = StreamingSpeaker(self.tts)
        buf = SentenceBuffer()
        reply, first_token = "", None

        try:
            async for ev in self.agent.stream_chat(text):
                if ev.tool_note:
                    print(f"  [tool] {ev.tool_note}")
                if ev.delta:
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    reply += ev.delta
                    for sentence in buf.feed(ev.delta):
                        speaker.feed(sentence)
            speaker.feed(buf.flush() or "")
            t_llm = time.perf_counter() - t0
            print(f"Joi: {reply.strip()}")
        finally:
            first_audio = await asyncio.to_thread(speaker.finish)

        return {"llm": t_llm, "first_token": first_token,
                "first_audio": first_audio, "turn": time.perf_counter() - t0}

    async def _respond_plain(self, text: str) -> dict:
        """Fallback: blocking ollama.chat (no tools), then pipelined speech."""
        t0 = time.perf_counter()
        reply = await asyncio.to_thread(self.agent.chat, text)
        t_llm = time.perf_counter() - t0
        print(f"Joi: {reply}")

        speaker = StreamingSpeaker(self.tts)
        buf = SentenceBuffer()
        for sentence in buf.feed(reply + " "):
            speaker.feed(sentence)
        speaker.feed(buf.flush() or "")
        first_audio = await asyncio.to_thread(speaker.finish)
        if first_audio is not None:
            first_audio += t_llm  # align metric: measured from agent start

        return {"llm": t_llm, "first_token": None,
                "first_audio": first_audio, "turn": time.perf_counter() - t0}


def run() -> None:
    """Synchronous entry point used by main.py."""
    try:
        asyncio.run(VoiceAssistant().run())
    except KeyboardInterrupt:
        print("\nBye.")
