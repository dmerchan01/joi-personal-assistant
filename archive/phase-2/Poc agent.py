"""
Phase 2 PoC — agent with REAL tools (app launcher + weather).

Run from inside the phase-2 folder so `from tools import ...` resolves:
    cd phase-2
    python poc_agent.py

Verified so far on your machine:
- thinking=False works through LlamaIndex (turns dropped from ~20s to ~1s)
- Tool calling works through FunctionAgent (3/3 correct routing)
- All three real tools work standalone

Still to verify (from `ollama ps` while this runs):
- whether context_window=8192 actually reaches Ollama's num_ctx
"""
import asyncio
import time

from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent

from tools import list_installed_apps, open_app, get_weather


async def main() -> None:
    llm = Ollama(
        model="qwen3:8b",
        request_timeout=120.0,
        thinking=False,        # verified: collapses latency ~10x
        context_window=8192,   # intended num_ctx=8192; pending ollama ps check
    )

    agent = FunctionAgent(
        name="joi",
        description="A local voice assistant that can open apps and check weather.",
        llm=llm,
        tools=[open_app, list_installed_apps, get_weather],
        system_prompt=(
            "You are Joi, a local voice assistant. "
            "Use open_app to open applications, list_installed_apps if unsure "
            "what is installed, and get_weather for weather questions. "
            "Reply in at most two short sentences, no emoji, written to be spoken aloud."
        ),
    )

    tests = [
        "What's the weather in Madrid?",
        "Open the calculator",
        "Abre Cyberpunk",                  # Spanish + a Steam game
        "Tell me a fun fact about space",  # expect: NO tool call
    ]

    for msg in tests:
        t0 = time.perf_counter()
        resp = await agent.run(user_msg=msg)
        dt = time.perf_counter() - t0
        print(f"\nYou: {msg}")
        print(f"Joi ({dt:.2f}s): {str(resp)}")


if __name__ == "__main__":
    asyncio.run(main())