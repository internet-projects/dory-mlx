"""Convert NVIDIA's Dory BF16 checkpoint to an MLX build.
    python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-8bit --bits 8
    python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-bf16 --bits 16
The q/k/v/g and gate/up projections are fused on load (see Model.sanitize), so the saved files use the fused layout."""
import argparse, json, shutil
from pathlib import Path
import dory_mlx
dory_mlx.register()
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model, quantize_model, save_config, save_model

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--bits", type=int, default=8, choices=[4, 8, 16]); ap.add_argument("--group-size", type=int, default=64)
a = ap.parse_args()
src = Path(a.src) if Path(a.src).exists() else Path(snapshot_download(a.src))
dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
model, _ = load_model(src)
cfg = json.load(open(src / "config.json"))
if a.bits < 16:
    model, cfg = quantize_model(model, cfg, a.group_size, a.bits)
save_model(dst, model, donate_model=False); save_config(cfg, dst / "config.json")
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(src / f, dst / f)
print("saved", dst)
