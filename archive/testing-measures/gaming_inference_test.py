"""Run while Cyberpunk is running. Tests gaming-mode candidates."""
import time
import ollama

PROMPT = "Give me 3 quick tips for a boss fight in an FPS game"

scenarios = [
    ("qwen3:4b on GPU", "qwen3:4b", {}),
    ("qwen3:8b CPU-only", "qwen3:8b", {"num_gpu": 0}),
]

for name, model, opts in scenarios:
    input(f"\n>>> Next: {name}. Note your in-game FPS now, then press Enter...")
    start = time.perf_counter()
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        think=False,
        options=opts,
    )
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.1f}s total")
    print(f"Response: {resp['message']['content'][:150]}...")
    input(">>> What did FPS do during generation? Note it, press Enter to unload...")
    subprocess_cmd = ["ollama", "stop", model]  # free VRAM before next scenario