"""Sanity checks for a Dory MLX build.
  python scripts/check.py MODEL_DIR                      # cache/window consistency (no torch needed)
  python scripts/check.py MODEL_DIR --ref NVIDIA_BF16_DIR  # also compare with an independent fp32 PyTorch reference (needs torch, ~16 GB RAM)
Consistency test: with the sliding window shrunk to 64, a 300-token text is run as one pass, in chunks, and token by token with caches;
the logits must agree (top-1 100%, tiny KL)."""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx
import dory_mlx
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--ref"); a = ap.parse_args()

def kl_top1(x, ref):
    x = x.astype(np.float64); ref = ref.astype(np.float64)
    lx = x - x.max(-1, keepdims=True); lx -= np.log(np.exp(lx).sum(-1, keepdims=True))
    lr = ref - ref.max(-1, keepdims=True); lr -= np.log(np.exp(lr).sum(-1, keepdims=True))
    return float((np.exp(lr) * (lr - lx)).sum(-1).mean()), float((x.argmax(-1) == ref.argmax(-1)).mean())

TEXT = ("The Parthenon is a former temple on the Athenian Acropolis, Greece, dedicated to the goddess Athena. In football, the offside rule prevents "
        "attackers from waiting near the opponent's goal. To make a basic tomato sauce, heat olive oil in a pan and add garlic. Interest rates influence "
        "borrowing costs across the economy. ") * 6
ok = True
model, tok = dory_mlx.load(a.model, loops=2); model.args.sliding_window = 64; model._all_logits = True
ids = tok.encode(TEXT, add_special_tokens=False)[:300]; x = mx.array(ids)[None]
full = np.array(model(x)[0].astype(mx.float32))
for split in ((100, 200, 300), (37, 150, 300)):
    c = make_prompt_cache(model); outs = []; prev = 0
    for s in split:
        outs.append(np.array(model(x[:, prev:s], cache=c)[0].astype(mx.float32))); prev = s
    kl, t1 = kl_top1(np.concatenate(outs), full); ok &= t1 > 0.97 and kl < 0.01
    print(f"chunked {split}: top-1 agreement {t1:.3f}, KL {kl:.6f}")
c = make_prompt_cache(model); model(x[:, :200], cache=c)
dec = np.concatenate([np.array(model(x[:, i:i + 1], cache=c)[0].astype(mx.float32)) for i in range(200, 300)])
kl, t1 = kl_top1(dec, full[200:300]); ok &= t1 > 0.97 and kl < 0.01
print(f"prefill 200 + decode 100: top-1 agreement {t1:.3f}, KL {kl:.6f}")
if a.ref:
    model.args.sliding_window = 1024
    model2, tok2 = dory_mlx.load(a.model, loops=2); model2._all_logits = True
    ids = tok2.encode(TEXT, add_special_tokens=False)[:55]
    mine = np.array(model2(mx.array(ids)[None])[0].astype(mx.float32))
    out = Path(tempfile.mkdtemp()) / "ref.npy"
    subprocess.run([sys.executable, str(Path(__file__).parent / "dory_ref_torch.py"), "--model", a.ref, "--loops", "2",
                    "--ids", ",".join(map(str, ids)), "--out", str(out)], check=True)
    kl, t1 = kl_top1(mine, np.load(out)); ok &= t1 > 0.95 and kl < 0.02
    print(f"vs fp32 PyTorch reference ({len(ids)} tokens, loops=2): top-1 agreement {t1:.3f}, KL {kl:.5f}")
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
