"""Convert NVIDIA's Dory BF16 checkpoint to an MLX build.
    python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-8bit --bits 8
    python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-bf16 --bits 16
    python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-mixed-4-8 --recipe mixed-4-8
The q/k/v/g and gate/up projections are fused on load (see Model.sanitize), so the saved files use the fused layout.

Mixed recipes set the bits per block. The 40 looped layers run 1 to 4 times per token, so their
quantization error is reused on every loop; the recipes give them more bits than the input/output layers.
  mixed-4-8  looped 8-bit, input/output 4-bit, embeddings + head 8-bit   (~3.5 GB)
  mixed-4-6  looped 6-bit, input/output 4-bit, embeddings + head 8-bit   (~3.0 GB)
  4-head8    everything 4-bit except embeddings + head 8-bit             (~2.6 GB)
--keep-bf16 head (or embed, or head,embed) leaves those unquantized instead."""
import argparse, json, re, shutil
from pathlib import Path
import dory_mlx
dory_mlx.register()
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model, quantize_model, save_config, save_model

RECIPES = {  # (looped, input/output, embeddings + head)
    "mixed-4-8": (8, 4, 8),
    "mixed-4-6": (6, 4, 8),
    "4-head8": (4, 4, 8),
}


def make_predicate(cfg, recipe, keep_bf16, group_size):
    lo = cfg["n_input_layers"]
    hi = lo + cfg["n_recurrent_layers"]
    looped, io, head = RECIPES[recipe] if recipe else (None, None, None)

    def predicate(path, module):
        if path == "lm_head" and "head" in keep_bf16:
            return False
        if path == "backbone.embeddings" and "embed" in keep_bf16:
            return False
        if recipe is None:
            return True
        if path in ("lm_head", "backbone.embeddings"):
            bits = head
        else:
            m = re.match(r"backbone\.layers\.(\d+)\.", path)
            bits = looped if m and lo <= int(m.group(1)) < hi else io
        return {"group_size": group_size, "bits": bits}

    return predicate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=8, choices=[4, 6, 8, 16])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--recipe", choices=sorted(RECIPES))
    ap.add_argument("--keep-bf16", default="", help="comma list: head, embed")
    a = ap.parse_args()
    keep = {s.strip() for s in a.keep_bf16.split(",") if s.strip()}
    if not keep <= {"head", "embed"}:
        ap.error("--keep-bf16 takes head and/or embed")
    src = Path(a.src) if Path(a.src).exists() else Path(snapshot_download(a.src))
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(src)
    cfg = json.load(open(src / "config.json"))
    if a.recipe or a.bits < 16:
        bits = 8 if a.recipe else a.bits
        model, cfg = quantize_model(model, cfg, a.group_size, bits,
                                    quant_predicate=make_predicate(cfg, a.recipe, keep, a.group_size))
        # record unquantized layers explicitly (mlx-lm otherwise infers it from the weights)
        for name, path in (("head", "lm_head"), ("embed", "backbone.embeddings")):
            if name in keep:
                cfg["quantization"][path] = False
                cfg["quantization_config"][path] = False
    save_model(dst, model, donate_model=False); save_config(cfg, dst / "config.json")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    size = sum(p.stat().st_size for p in dst.glob("*.safetensors")) / 1e9
    print(f"saved {dst} ({size:.2f} GB of weights)")


if __name__ == "__main__":
    main()
