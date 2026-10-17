import argparse
from . import chat, load

ap = argparse.ArgumentParser(prog="python -m dory_mlx", description="Chat with a Dory MLX build.")
ap.add_argument("prompt")
ap.add_argument("--model", default="netprojects/nemotron-dory1.0-4b-mlx-8bit")
ap.add_argument("--loops", type=int, default=2, choices=[1, 2, 3, 4])
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--no-thinking", action="store_true", help="ask for an empty <think></think> block (not NVIDIA's recommended mode)")
a = ap.parse_args()
model, tok = load(a.model, loops=a.loops)
for piece in chat(model, tok, a.prompt, max_tokens=a.max_tokens, temp=a.temp, thinking=not a.no_thinking):
    print(piece, end="", flush=True)
print()
