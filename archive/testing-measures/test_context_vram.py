"""Phase 0: how much VRAM does each context size cost?"""
import subprocess
import ollama

def vram_used():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip())

for num_ctx in [4096, 8192, 16384]:
    ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "hi"}],
        options={"num_ctx": num_ctx},
    )
    print(f"num_ctx={num_ctx}:  VRAM={vram_used()} MiB")
    print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)