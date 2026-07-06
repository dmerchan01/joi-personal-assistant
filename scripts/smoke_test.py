"""No-mic smoke test for Joi. Run from the repo root:

    .venv/bin/python scripts/smoke_test.py
    JOI_MODE=gaming .venv/bin/python scripts/smoke_test.py   # CPU-only check

Verifies: imports, config, tools standalone, TTS synthesis to a wav file,
one agent turn with a real tool call (latency printed), streaming
time-to-first-audio vs the old full-synthesis approach, and that Ollama is
actually running with num_ctx=8192 (and CPU-only in gaming mode).

Speakers are NOT used (audio goes to a wav / timing-only sink); the mic is
never touched. Launching apps is also skipped (open_app would really open
windows) — app listing is verified instead; run the voice loop for that.
"""
import asyncio
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES: list[str] = []


def section(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("== 1. imports & config ==")
    from joi.config import Config
    from joi.capabilities import REGISTRY, all_tools
    from joi.tts import TTS
    from joi.agent import JoiAgent

    cfg = Config()
    section("config", cfg.num_ctx == 8192 and cfg.think is False,
            f"mode={cfg.mode} model={cfg.model} num_ctx={cfg.num_ctx} "
            f"think={cfg.think} keep_alive={cfg.keep_alive}")
    section("registry", len(all_tools()) >= 3,
            f"{[c.name for c in REGISTRY]} -> {[t.__name__ for t in all_tools()]}")

    print("\n== 2. tools standalone ==")
    from joi.capabilities.apps import list_installed_apps
    from joi.capabilities.weather import get_weather

    apps = list_installed_apps()
    section("list_installed_apps", bool(apps), f"{len(apps.split(','))} apps found")
    wx = get_weather("Madrid")
    section("get_weather", "degrees" in wx, wx)

    print("\n== 3. TTS synthesis to wav ==")
    tts = TTS(cfg.piper_voice)
    t0 = time.perf_counter()
    chunks = list(tts.voice.synthesize("Hello, this is Joi speaking."))
    t_synth = time.perf_counter() - t0
    out_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "smoke_test_output.wav")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(chunks[0].sample_rate)
        for c in chunks:
            w.writeframes(c.audio_int16_bytes)
    n_samples = sum(len(c.audio_int16_bytes) for c in chunks) // 2
    audio_s = n_samples / chunks[0].sample_rate
    section("piper", n_samples > 0,
            f"{audio_s:.1f}s audio in {t_synth:.2f}s -> {out_wav}")

    print("\n== 4. agent one-shot with tool call ==")
    agent = JoiAgent(cfg)

    async def one_shot(msg: str) -> tuple[str, list[str], float, float | None, float | None]:
        """Returns (reply, tool_notes, total_s, first_token_s, first_sentence_s)."""
        from joi.sentences import SentenceBuffer
        buf = SentenceBuffer()
        t0 = time.perf_counter()
        reply, notes = "", []
        first_token = first_sentence = None
        async for ev in agent.stream_chat(msg):
            if ev.tool_note:
                notes.append(ev.tool_note)
            if ev.delta:
                if first_token is None:
                    first_token = time.perf_counter() - t0
                reply += ev.delta
                if buf.feed(ev.delta) and first_sentence is None:
                    first_sentence = time.perf_counter() - t0
        return reply.strip(), notes, time.perf_counter() - t0, first_token, first_sentence

    async def agent_sections() -> tuple:
        # One event loop for both turns: the Ollama async client's connection
        # pool is bound to the loop it first runs on.
        a = await one_shot("What's the weather in Madrid?")
        b = await one_shot("Tell me something nice in exactly two short sentences.")
        return a, b

    (reply, notes, dt, ft, _), turn2 = asyncio.run(agent_sections())
    print(f"  reply: {reply}")
    print(f"  latency: {dt:.2f}s (first token {ft:.2f}s)")
    section("agent tool routing", any("get_weather" in n for n in notes),
            f"tools called: {notes}")

    print("\n== 5. time-to-first-audio: streaming vs old full-synthesis ==")
    reply, _, t_full, _, t_first_sent = turn2
    print(f"  reply: {reply}")
    sentences = reply.split(". ")
    first_sentence_text = sentences[0] if sentences else reply
    # time Piper takes to produce the FIRST audio chunk of sentence 1
    t0 = time.perf_counter()
    next(iter(tts.voice.synthesize(first_sentence_text)))
    t_first_chunk = time.perf_counter() - t0
    if t_first_sent is None:  # single-sentence reply: streaming waits for it all
        t_first_sent = t_full
    ttfa_old = t_full + t_first_chunk        # old: speak only after full reply
    ttfa_new = t_first_sent + t_first_chunk  # new: speak at first sentence
    section("streaming TTFA", ttfa_new <= ttfa_old,
            f"old {ttfa_old:.2f}s -> streaming {ttfa_new:.2f}s "
            f"(LLM full {t_full:.2f}s, first sentence {t_first_sent:.2f}s, "
            f"first chunk synth {t_first_chunk:.2f}s)")

    print("\n== 6. ollama state (num_ctx / processor) ==")
    import subprocess
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    line = next((ln for ln in ps.splitlines() if cfg.model in ln), "")
    print(f"  {line or ps}")
    section("num_ctx=8192 reached Ollama", " 8192 " in f" {line} ")
    if cfg.mode == "gaming":
        section("gaming mode CPU-only", "100% CPU" in line, f"processor: {line}")
    else:
        section("normal mode on GPU", "GPU" in line)

    print("\n== 7. notes path confinement ==")
    import tempfile
    import joi.capabilities.notes as notes
    notes._ROOT = tempfile.mkdtemp(prefix="joi-notes-test-")
    escapes = ["../escape", "/etc", "a/b", "..", ".hidden", "x/../../y", "~root"]
    rejected = [e for e in escapes if notes._safe_folder(e) is None]
    section("escape attempts rejected", len(rejected) == len(escapes),
            f"{len(rejected)}/{len(escapes)} rejected")
    ok_write = "Noted" in notes.add_note("ToDo", "confinement test note")
    inside = os.path.isfile(os.path.join(notes._ROOT, "ToDo",
                                         time.strftime("%Y-%m-%d") + ".md"))
    section("valid note lands inside root", ok_write and inside)

    print("\n== 8. routing test (EN/ES, ~27 tools on one agent) ==")
    from llama_index.core.workflow import Context
    from joi.capabilities import reminders
    import joi.capabilities.anime as anime
    anime._run_ani_cli = lambda *a, **k: ""  # route, but never launch mpv

    # (query, expected tool name or None for plain chat)
    ROUTES = [
        ("What's the astronomy picture of the day?", "nasa_picture_of_the_day"),
        ("¿Hay asteroides cerca de la Tierra esta semana?", "asteroids_near_earth"),
        ("Show me a Mars rover photo", "mars_rover_photo"),
        ("Any wildfires or storms on Earth right now?", "earth_natural_events"),
        ("What's the top tech news?", "tech_news_top"),
        ("Open the second news story", "tech_news_open"),
        ("Latest AI papers on arXiv", "ai_papers_arxiv"),
        ("¿Cuáles son los papers de moda en Hugging Face?", "ai_papers_trending"),
        ("Pause the music", "music_pause"),
        ("¿Qué canción está sonando?", "music_now_playing"),
        ("Play Bohemian Rhapsody by Queen", "music_play_song"),
        ("Remind me in 10 minutes to stretch", "set_reminder"),
        ("Cancela todos los recordatorios", "cancel_reminder"),
        ("Add a note to my ToDo folder that I must finish the homework", "add_note"),
        ("Ponme una nota en Projects que quiero un agente de visión", "add_note"),
        ("What's the weather in Madrid?", "get_weather"),
        ("Abre Firefox", "open_app"),
        ("I want to keep watching Fire Force", "anime_watch"),
        ("Pon el siguiente episodio de Dr. Stone", "anime_watch"),
        ("Play episode 3 of Frieren", "anime_watch"),
        ("What anime am I watching?", "anime_progress"),
        ("Tell me a fun fact about space", None),
        ("¿Cómo estás hoy?", None),
        ("What's two plus two?", None),
    ]

    real_popen = subprocess.Popen
    subprocess.Popen = lambda *a, **kw: None  # no browsers/apps during routing

    async def route_all() -> tuple[int, list[str], list[float]]:
        # fresh agent: its Ollama async client must bind to THIS event loop
        r_agent = JoiAgent(cfg)
        hits, misses, chat_times = 0, [], []
        for query, expected in ROUTES:
            r_agent.ctx = Context(r_agent.agent)  # fresh conversation per query
            called: list[str] = []
            t0 = time.perf_counter()
            async for ev in r_agent.stream_chat(query):
                if ev.tool_note:
                    called.append(ev.tool_note.split("(")[0])
            dt = time.perf_counter() - t0
            ok = (expected in called) if expected else not called
            hits += ok
            if expected is None:
                chat_times.append(dt)
            mark = "ok " if ok else "MISS"
            print(f"  {mark} {dt:4.1f}s  {query!r} -> {called or 'no tool'}"
                  f"{'' if ok else f' (expected {expected})'}")
            if not ok:
                misses.append(query)
        return hits, misses, chat_times

    try:
        hits, misses, chat_times = asyncio.run(route_all())
    finally:
        subprocess.Popen = real_popen
        reminders.cancel_reminder(0)  # clean up any timer the test set

    score = hits / len(ROUTES)
    avg_chat = sum(chat_times) / len(chat_times) if chat_times else 0.0
    print(f"  routing accuracy: {hits}/{len(ROUTES)} = {score:.0%}"
          f" | avg no-tool turn: {avg_chat:.2f}s")
    section("routing accuracy >= 80%", score >= 0.8, f"{score:.0%}")
    section("no-tool latency ~1s", avg_chat < 2.0, f"{avg_chat:.2f}s warm")

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
