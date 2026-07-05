# Archive — código histórico (nada de esto se usa en producción)

Se conserva como referencia de las fases de validación; el código vivo no
importa nada de aquí.

- `phase-2/` — PoC del FunctionAgent y tools originales. Migradas a
  `joi/capabilities/` y `joi/agent.py`.
- `multi-agent-test/` — experimento multi-agente con LlamaIndex/Ollama.
  Descartado: los handoffs añadían latencia frente al agente único.
- `testing-measures/` — scripts de medición de Phase 0. De aquí salieron
  los valores "validados" de `joi/config.py` (thinking=False, num_ctx=8192,
  Whisper small/cuda/float16, gaming mode en CPU).
- `setup_env.sh` — fijaba LD_LIBRARY_PATH para las libs CUDA (bash-only).
  Obsoleto: `joi/cuda_libs.py` lo resuelve desde Python.
