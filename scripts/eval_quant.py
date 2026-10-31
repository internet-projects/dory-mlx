"""Compare Dory MLX builds with a reference build (normally bf16) at several loop counts.
    python scripts/eval_quant.py --ref out-bf16 --builds out-8bit out-mixed-4-8 out-4bit --text-file wiki.test.raw
For each build and loop count: mean KL divergence from the reference, top-1 agreement and perplexity on the
same tokens. Runs on any MLX backend (Metal, CUDA or CPU); speed is not measured.
The reference log-probabilities are kept on disk, so only one model is in memory at a time."""
import argparse, gc, json, tempfile
from pathlib import Path

import numpy as np
import mlx.core as mx

import dory_mlx


def log_softmax(x):
    x = x.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def run(path, loops, chunks):
    dory_mlx.register()
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(path), model_config={"n_recurrent_loops": loops})
    model._all_logits = True
    for c in chunks:  # each chunk is scored independently, from an empty cache
        yield np.array(log_softmax(model(mx.array(c)[None])[0]))
    del model
    gc.collect(); mx.clear_cache()


def size_gb(path):
    return sum(p.stat().st_size for p in Path(path).glob("*.safetensors")) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--builds", nargs="+", required=True)
    ap.add_argument("--loops", default="1,2,4")
    ap.add_argument("--text-file", help="plain text, e.g. WikiText-2 test")
    ap.add_argument("--random-tokens", type=int, help="testing only: random ids instead of text")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="eval_quant.json")
    a = ap.parse_args()

    if a.random_tokens:
        ids = np.random.default_rng(0).integers(0, a.random_tokens, a.tokens).tolist()
    else:
        _, tok = dory_mlx.load(a.ref, loops=1)
        ids = tok.encode(Path(a.text_file).read_text(), add_special_tokens=False)[: a.tokens]
    chunks = [ids[i : i + a.chunk] for i in range(0, len(ids), a.chunk)]
    loops = [int(x) for x in a.loops.split(",")]
    tmp = Path(tempfile.mkdtemp(prefix="dory_eval_"))
    rows = []

    for k in loops:
        refs = []
        for j, lp in enumerate(run(a.ref, k, chunks)):
            f = tmp / f"ref_k{k}_{j}.npy"; np.save(f, lp.astype(np.float32)); refs.append(f)
        ppl_ref = None
        for b in [a.ref] + a.builds:
            kls, agree, nll, n = [], [], 0.0, 0
            for f, c, lp in zip(refs, chunks, run(b, k, chunks)):
                r = np.load(f).astype(np.float32)
                kls.append((np.exp(r) * (r - lp)).sum(-1))
                agree.append(r.argmax(-1) == lp.argmax(-1))
                tgt = np.array(c[1:])
                nll -= lp[np.arange(len(tgt)), tgt].sum(); n += len(tgt)
            ppl = float(np.exp(nll / n))
            if b == a.ref:
                ppl_ref = ppl
            rows.append(dict(build=Path(b).name, gb=round(size_gb(b), 2), loops=k,
                             kl=float(np.concatenate(kls).mean()), top1=float(np.concatenate(agree).mean()),
                             ppl=ppl, ppl_vs_ref=ppl / ppl_ref - 1))
            r_ = rows[-1]
            print(f"k={k} {r_['build']:<24} {r_['gb']:>5} GB  KL {r_['kl']:.4f}  top-1 {r_['top1']:.3f}  "
                  f"ppl {r_['ppl']:.3f} ({r_['ppl_vs_ref']:+.2%})", flush=True)

    Path(a.out).write_text(json.dumps(dict(tokens=len(ids), chunk=a.chunk, rows=rows), indent=1))
    md = ["| Build | Size | Loops | KL vs reference | Top-1 agreement | Perplexity |", "|---|---|---|---|---|---|"]
    md += [f"| {r['build']} | {r['gb']} GB | {r['loops']} | {r['kl']:.4f} | {r['top1']:.1%} | {r['ppl']:.2f} |" for r in rows]
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("wrote", a.out, "and", Path(a.out).with_suffix(".md"))


if __name__ == "__main__":
    main()
