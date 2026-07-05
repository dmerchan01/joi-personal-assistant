"""
LLM via Ollama.

This wrapper uses exactly the API surface you already validated in Phase 0:
    ollama.chat(model=..., messages=..., think=False, options={...})
- think=False gave first token in ~0.15s on qwen3:8b.
- options['num_ctx']=8192 keeps the model 100% on GPU.
- options['num_gpu']=0 (gaming mode) forces CPU-only inference.

It keeps a running message history so conversation stays coherent. Real
persistent memory + logging is Phase 4; this is just an in-RAM list.
"""
import ollama


class LLM:
    """Plain-chat fallback backend (JOI_BACKEND=plain): no tools, no agent —
    just ollama.chat with in-RAM message history."""

    def __init__(self, model: str = "qwen3:8b", think: bool = False,
                 num_ctx: int = 8192, num_gpu: int | None = None,
                 system_prompt: str | None = None,
                 keep_alive: str | None = None):
        self.model = model
        self.think = think
        self.keep_alive = keep_alive
        self.options: dict = {"num_ctx": num_ctx}
        if num_gpu is not None:
            self.options["num_gpu"] = num_gpu

        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        resp = ollama.chat(
            model=self.model,
            messages=self.messages,
            think=self.think,
            options=self.options,
            keep_alive=self.keep_alive,
        )
        content = resp["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": content})
        return content
