"""Phase 0: thinking-off latency benchmark + negative tool-call test."""
import time
import ollama

MODELS = ["qwen3:8b", "qwen3:4b"]  # 14B already discarded

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

def timed_chat(model, content, think, tools=None):
    """Returns (first_token_latency, total_time, n_chars, used_tool)."""
    start = time.perf_counter()
    first_token = None
    text = ""
    used_tool = False
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": content}],
        think=think,
        tools=tools,
        stream=True,
    )
    for chunk in stream:
        msg = chunk.get("message", {})
        piece = msg.get("content", "")
        if piece and first_token is None:
            first_token = time.perf_counter() - start
        text += piece
        if msg.get("tool_calls"):
            used_tool = True
            if first_token is None:
                first_token = time.perf_counter() - start
    total = time.perf_counter() - start
    return first_token, total, len(text), used_tool


for model in MODELS:
    print(f"\n{'='*50}\n{model}\n{'='*50}")

    # Warmup (avoid cold-start skew — we learned this lesson)
    ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}])

    # A) Latency: thinking ON vs OFF
    prompt = "Explain in 3 sentences what a context window is"
    for think in [True, False]:
        ft, total, chars, _ = timed_chat(model, prompt, think)
        print(f"  think={think}:  first_token={ft:.2f}s  total={total:.2f}s  chars={chars}")

    # B) Negative tool test: should NOT call get_weather (5 runs)
    print("  Negative tool test ('tell me a joke'):")
    for i in range(5):
        _, _, _, used = timed_chat(model, "Tell me a short joke", think=False, tools=tools)
        print(f"    run {i+1}: {'✗ FALSE POSITIVE — called tool!' if used else '✓ no tool (correct)'}")

    # C) Positive control with thinking off (confirm tools still work)
    print("  Positive tool test with think=False:")
    for i in range(3):
        _, _, _, used = timed_chat(model, "What's the weather in San Jose?", think=False, tools=tools)
        print(f"    run {i+1}: {'✓ tool called' if used else '✗ MISS'}")