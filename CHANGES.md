# Refactor notes

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
