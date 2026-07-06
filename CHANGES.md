# Refactor notes

## 2026-07-06 — anime capability (ani-cli)

Verified against the installed /usr/bin/ani-cli before writing any code:

- History file (`~/.local/state/ani-cli/ani-hsts`) stores
  `last_started_ep \t id \t title (N episodes)` and is updated the moment
  an episode starts — so "continue" = last + 1, and "next episode" right
  after finishing one works with the same logic.
- A blind `-S 1` is WRONG: searching "fire force" returns 6 entries
  (seasons, a spin-off, one unrelated show) and the base series is 6th.
  The capability runs the same allanime GraphQL search ani-cli uses (same
  order), finds the right entry, and passes its index to `-S`.
- English↔romaji names ("Fire Force" vs "Enen no Shouboutai") are resolved
  by ID intersection: the allanime search for the English name returns the
  same show id the history stores. No alias table needed.
- `--exit-after-play` is NOT used — it forces mpv into the foreground and
  would block the voice loop until the episode ends. Default detached mode
  + no tty makes ani-cli exit right after launching mpv.
- Sources for a specific entry can be temporarily dead ("Episode is
  released, but no valid sources!", seen live for several first-page
  entries) — the tool retries up to 3 closest candidates and reports
  honestly if none work.
- One merged tool (`anime_watch`, episode optional) instead of separate
  continue/play tools: qwen3 kept passing `episode: 1` for "siguiente
  episodio", which would restart the series. With one tool the wrong
  choice is impossible; routing suite back to 100% (24/24).
- Playback is CONFIRMED before claiming success: ani-cli can fetch a link
  whose stream is dead — mpv then dies (or hangs) with no window while the
  tool reports success (seen live with Jujutsu Kaisen 0). mpv only opens a
  window once video decodes, so the tool polls `hyprctl clients` for a
  window titled "... Episode N" (up to 15s, returns as soon as it shows);
  fallback outside Hyprland: mpv-still-alive after 5s. A hung playerless
  mpv we started gets killed (only our pid, never a pre-existing mpv).
- Intra-episode resume is NOT possible: stream URLs are ephemeral
  (tokenized, change per fetch), so mpv's watch-later can't match them.
  History is episode-granular; to rewatch say "play episode N of X".

## 2026-07-05 — Phase 3a: six new capabilities

Every endpoint verified with a live request before writing the code:

- **APOD / NeoWs**: work as documented on api.nasa.gov (DEMO_KEY fallback,
  `NASA_API_KEY` from .env).
- **Mars Rover Photos API is DEAD** — every path under
  `api.nasa.gov/mars-photos` (and the mars-photos.herokuapp.com backend)
  returns 404 "No such app". `mars_rover_photo` uses the official NASA
  Image and Video Library (`images-api.nasa.gov/search`, no key) instead;
  it reports title/date rather than camera/sol (that metadata doesn't
  exist there).
- **EONET**: confirmed at `eonet.gsfc.nasa.gov/api/v3/events`, no key needed.
- **Hacker News**: `hacker-news.firebaseio.com/v0` confirmed current.
- **arXiv**: must be called as `https://` — `http://export.arxiv.org`
  301-redirects and the tool doesn't follow redirects.
- **Hugging Face `api/daily_papers`**: EXISTS and works (JSON with
  `paper.id/title/summary`, `?limit=` param honored). No scraping needed.
- **playerctl 2.4.1** already installed; `-p NAME` flag verified. Spotify
  Web API Stage B left as honest stubs (needs dev app + OAuth + Premium
  for playback — not implementable without real credentials).
- **Reminders design**: fire from a `threading.Timer`; speech serializes
  on `joi.tts.PLAYBACK_LOCK` so a reminder waits for an in-progress reply.
  Chosen over a checked-between-turns queue because the loop is usually
  blocked on input(), where a queue would never drain. v1 is in-memory.
- **Routing at 24 tools (single agent)**: 20-query EN/ES suite scores
  **100%** after two prompt tweaks (live-state questions need a tool /
  trivia needs none; the music_now_playing docstring alone didn't fix the
  one miss). Warm no-tool turns: ~0.5s — routing did NOT get slower with
  6x more tools. No multi-agent refactor needed yet.

## 2026-07-06 — automatic GPU handoff for Steam games

Field-tested failure: "open doom the dark ages" launched the game but it
died before showing video — the LLM was holding 6.2 GB of the 8 GB VRAM
(confirmed with nvidia-smi + ollama ps). Fix: `open_app` emits a
game-launched event (joi/events.py) when the Exec contains
`steam://rungameid`; VoiceAssistant finishes the turn, then runs
`ollama stop` and flips inference to CPU (`num_gpu=0`), same as gaming
mode. Verified live: turn after the launch runs 100% CPU.

## 2026-07-05 — refactor

Everything the mission brief assumed was checked against the installed
packages (`llama-index-core 0.14.22`, `llama-index-llms-ollama 0.10.1`,
`ollama 0.6.2`, `piper-tts 1.4.2`). Deviations and confirmations:

## Confirmed (previously unverified)

- **`Ollama(context_window=8192)` DOES reach Ollama's `num_ctx`.**
  The wrapper builds `options={"num_ctx": self.get_context_window(), ...}`
  on every request (`base.py::_model_kwargs`). Verified live: `ollama ps`
  shows `CONTEXT 8192`, 6.2 GB, 100% GPU. No `additional_kwargs` needed
  for num_ctx.
- **Gaming mode through LlamaIndex:** `additional_kwargs={"num_gpu": 0}`
  merges into the same per-request `options` dict. Verified live:
  `ollama ps` shows `100% CPU` under `JOI_MODE=gaming`.
- **`keep_alive="30m"`** is accepted (field type `float | str`, passed
  straight to `client.chat`). Verified live: UNTIL column ~30 minutes.
  Also added to the plain fallback (`ollama.chat` accepts `keep_alive` too).
- **Agent streaming is cleanly supported:** `agent.run()` returns a
  `WorkflowHandler`; `handler.stream_events()` yields `AgentStream` events
  whose `.delta`s concatenate to exactly the final response text, plus
  `ToolCallResult` events. So the LLM→TTS sentence pipeline streams
  straight from the agent (no fallback to post-hoc splitting needed).
- **Multi-turn memory:** pass the same `Context(agent)` to every
  `run(ctx=...)` call (verified: turn 2 recalled turn 1).

## Deviations from the brief

- None that change behavior. The only API not in the brief that the code
  now relies on: `Context` from `llama_index.core.workflow` (memory) and
  `AgentStream`/`ToolCallResult` events (streaming) — all verified above.

## Structural changes

- `phase-2/tools.py` code moved into `joi/capabilities/{apps,weather}.py`
  (docstrings and behavior unchanged). `phase-2/`, `multi-agent-test/`,
  `testing-measures/` and `setup_env.sh` moved to `archive/` (historical
  PoCs and measurements; the live code no longer imports from them).
- `joi/llm.py` kept as the `JOI_BACKEND=plain` fallback (now also takes
  `keep_alive`).
- New: `joi/agent.py`, `joi/capabilities/`, `joi/sentences.py`,
  `StreamingSpeaker` in `joi/tts.py`, async `VoiceAssistant` in
  `joi/assistant.py`, `scripts/smoke_test.py`.

## Measured results (this machine, 2026-07-05)

- Smoke test: ALL PASS in both normal and gaming mode.
- Time-to-first-audio, 2-sentence reply (streaming vs old full-synthesis):
  - normal (GPU, warm): **0.50s vs 0.74s**
  - gaming (CPU): **3.42s vs 4.84s**
  The gap grows with reply length: old TTFA ≈ full LLM time + first-chunk
  synth; streaming TTFA ≈ first-sentence LLM time + first-chunk synth.
- Warm agent turn (no tool): ~0.6s total; with weather tool: ~1s warm
  (first turn after model load is slower). Matches the validated baselines.
- The README "add a capability" example (`clock`/`get_time`) was created,
  tested live (agent routed "What time is it?" to the new tool), and then
  removed — one new file + one registry line is confirmed to be all it takes.
