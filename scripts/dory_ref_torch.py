"""Independent plain-PyTorch fp32 CPU reference for NVIDIA Dory (DoryForCausalLM).

Math follows vllm/model_executor/models/dory.py (vLLM PR #60149).
Usage: python3 -I dory_ref_torch.py --model DIR --loops 2 --ids "1,2,3" --out logits.npy [--topk]
       python3 -I dory_ref_torch.py --selftest
"""
import argparse, json, math, os, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6


def justnorm(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + EPS)


def slerp_norm(a, b, lr):
    ya, yb = justnorm(a), justnorm(b)
    return justnorm(ya + lr * (yb - ya))


def rope_neox(x, pos, theta, rot_dim):
    """x: (T, H, D); split-half (NeoX) rotation of first rot_dim dims."""
    inv = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
    ang = pos.float()[:, None] * inv[None, :]  # (T, rot/2)
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    x1, x2 = xr[..., : rot_dim // 2], xr[..., rot_dim // 2:]
    xr = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return torch.cat([xr, xp], -1)


class DoryRef:
    def __init__(self, model_dir, loops=None, kv_mode=None):
        from safetensors.torch import load_file
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        self.c = c
        self.H = c["hidden_size"]
        self.nh, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.L = c["num_hidden_layers"]
        self.std = c["initializer_range"]
        self.n_in = c.get("n_input_layers", 0)
        self.n_out = c.get("n_output_layers", 0)
        self.n_rec = c.get("n_recurrent_layers")
        if self.n_rec is None:
            self.n_rec = self.L - self.n_in - self.n_out
        assert self.n_in + self.n_rec + self.n_out == self.L
        self.loops = int(loops if loops is not None else c.get("n_recurrent_loops", 1))
        assert self.loops >= 1
        self.kv_mode = kv_mode or c.get("recurrent_kv_cache_mode", "per_loop")
        assert self.kv_mode in ("per_loop", "last_loop")
        self.types = c.get("layer_types") or [
            "mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
            for i in range(self.L)]
        self.window = c.get("sliding_window")
        rp = c.get("rope_parameters") or {}
        self.rope = {}
        for k, v in rp.items():
            self.rope[k] = (float(v["rope_theta"]), int(self.hd * v.get("partial_rotary_factor", 1.0)))
        g0 = c.get("attention_output_gate_logit_scale_init")
        if g0 is None:
            g0 = self.H ** 0.5
        self.sqk_f = self.hd ** 0.5 / self.std
        self.sgate_f = g0 ** 0.5 / self.std
        self.suv_f = self.H ** 0.5
        self.alpha_scale = c["ngpt_alpha_init"] / self.std
        w = load_file(os.path.join(model_dir, "model.safetensors"))
        self.w = {k: v.float().contiguous() for k, v in w.items()}

    # ---- sublayers
    def _attn(self, i, h, pos, cache):
        W = lambda n: self.w[f"backbone.layers.{i}.mixer.{n}"]
        T = h.shape[0]
        q = (h @ W("q_proj.weight").T).view(T, self.nh, self.hd)
        k = (h @ W("k_proj.weight").T).view(T, self.nkv, self.hd)
        v = (h @ W("v_proj.weight").T).view(T, self.nkv, self.hd)
        gate = h @ W("g_proj.weight").T
        theta, rd = self.rope[self.types[i]]
        q, k = rope_neox(q, pos, theta, rd), rope_neox(k, pos, theta, rd)  # RoPE BEFORE qk-norm
        sqk = (W("sqk") * self.sqk_f).view(1, self.nh, self.hd)
        q = sqk.square() * justnorm(q)
        k = justnorm(k)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # (heads,T,d)
        off = int(pos[0])
        if cache is not None:
            # replace slots [off, off+T) (matches vLLM ReferenceAttention)
            if cache.get("k") is not None:
                k = torch.cat((cache["k"][:, :off], k), 1)
                v = torch.cat((cache["v"][:, :off], v), 1)
            cache["k"], cache["v"] = k, v
        kpos = torch.arange(k.shape[1])
        qpos = torch.arange(off, off + T)[:, None]
        mask = kpos <= qpos
        if self.types[i] == "sliding_attention":
            mask &= kpos > qpos - self.window
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(self.hd),
                                           enable_gqa=True)
        o = o.transpose(0, 1).reshape(T, self.nh * self.hd)
        sg = (W("s_gate") * self.sgate_f).square()
        o = o * torch.sigmoid(gate * sg)
        o = justnorm(o)
        return o @ W("o_proj.weight").T

    def _mlp(self, i, h):
        W = lambda n: self.w[f"backbone.layers.{i}.mixer.{n}"]
        g = (h @ W("gate_proj.weight").T) * (W("suv_gate") * self.suv_f)
        u = (h @ W("up_proj.weight").T) * (W("suv_up") * self.suv_f)
        m = justnorm(F.silu(g) * u)
        return m @ W("down_proj.weight").T

    def _layer(self, i, h, pos, cache):
        if self.types[i] == "mlp":
            out = self._mlp(i, h)
            alpha = self.w[f"backbone.layers.{i}.mlp_alpha"]
        else:
            out = self._attn(i, h, pos, cache)
            alpha = self.w[f"backbone.layers.{i}.attn_alpha"]
        return slerp_norm(h, out, (alpha * self.alpha_scale).abs())

    def hidden(self, ids, start=0, caches=None):
        """ids: LongTensor (T,). caches: dict (layer, loop)->{} for incremental use (optional)."""
        pos = torch.arange(start, start + len(ids))
        h = self.w["backbone.embeddings.weight"][ids]

        def cache_for(i, loop):
            if caches is None or self.types[i] == "mlp":
                return None
            key = (i, loop if (self.kv_mode == "per_loop" and self.n_in <= i < self.n_in + self.n_rec) else 0)
            return caches.setdefault(key, {})
        for i in range(self.n_in):
            h = self._layer(i, h, pos, cache_for(i, 0))
        input_h = h
        state = input_h
        for lp in range(self.loops):
            h = justnorm(state + input_h)
            for i in range(self.n_in, self.n_in + self.n_rec):
                h = self._layer(i, h, pos, cache_for(i, lp))
            state = justnorm(h)
        h = state
        for i in range(self.n_in + self.n_rec, self.L):
            h = self._layer(i, h, pos, cache_for(i, 0))
        return h

    @torch.inference_mode()
    def logits(self, ids, start=0, caches=None):
        h = self.hidden(torch.as_tensor(ids, dtype=torch.long), start, caches)
        lg = h @ self.w["lm_head.weight"].T
        return lg * (self.w["lm_head.logit_scale"] / self.std)


# ---------------------------------------------------------------- selftest
def make_tiny(d, layer_types, n_in, n_rec, n_out, vocab=40, H=64, nh=4, nkv=2, hd=16, ff=96,
              window=4, loops=2, seed=0, **extra):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    cfg = dict(vocab_size=vocab, hidden_size=H, num_hidden_layers=len(layer_types), intermediate_size=ff,
               num_attention_heads=nh, num_key_value_heads=nkv, head_dim=hd, max_position_embeddings=64,
               tie_word_embeddings=False, ngpt_alpha_init=0.05, attention_output_gate_logit_scale_init=None,
               n_input_layers=n_in, n_recurrent_layers=n_rec, n_output_layers=n_out, n_recurrent_loops=loops,
               hidden_act="silu", initializer_range=0.0177, sliding_window=window, layer_types=layer_types,
               rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 100, "partial_rotary_factor": 1.0},
                                "full_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0}})
    cfg.update(extra)
    r = lambda *s, sc=0.05: torch.randn(*s, generator=g) * sc
    t = {"backbone.embeddings.weight": r(vocab, H, sc=1.0), "lm_head.weight": r(vocab, H, sc=0.3),
         "lm_head.logit_scale": r(vocab, sc=0.01).abs() + 0.01}
    for i, ty in enumerate(layer_types):
        p = f"backbone.layers.{i}."
        if ty == "mlp":
            t[p + "mlp_alpha"] = r(H, sc=0.02)
            for n, s in (("gate_proj", (ff, H)), ("up_proj", (ff, H)), ("down_proj", (H, ff))):
                t[p + f"mixer.{n}.weight"] = r(*s)
            t[p + "mixer.suv_gate"] = 1 + r(ff, sc=0.1)
            t[p + "mixer.suv_up"] = 1 + r(ff, sc=0.1)
        else:
            t[p + "attn_alpha"] = r(H, sc=0.02)
            for n, s in (("q_proj", (nh * hd, H)), ("k_proj", (nkv * hd, H)), ("v_proj", (nkv * hd, H)),
                         ("g_proj", (nh * hd, H)), ("o_proj", (H, nh * hd))):
                t[p + f"mixer.{n}.weight"] = r(*s)
            t[p + "mixer.sqk"] = 0.02 + r(nh * hd, sc=0.002)
            t[p + "mixer.s_gate"] = 0.02 + r(nh * hd, sc=0.002)
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in t.items()}, os.path.join(d, "model.safetensors"))


def selftest():
    # 3 + 4 + 3 layers following attn/mlp alternation
    lt = ["sliding_attention", "mlp", "full_attention",
          "mlp", "sliding_attention", "mlp", "full_attention",
          "mlp", "sliding_attention", "mlp"]
    ids = torch.tensor([3, 17, 5, 9, 33, 2, 8, 21, 14, 6, 30, 1])
    with tempfile.TemporaryDirectory() as d:
        make_tiny(d, lt, 3, 4, 3, window=4)
        outs = {}
        for L in (1, 2, 3):
            m = DoryRef(d, loops=L)
            lg = m.logits(ids)
            assert lg.shape == (12, 40) and torch.isfinite(lg).all()
            outs[L] = lg
            # chunked prefill + decode == full (per_loop caches kept separate)
            caches = {}
            parts = [m.logits(ids[:5], 0, caches)] + [m.logits(ids[i:i + 1], i, caches) for i in range(5, 12)]
            torch.testing.assert_close(torch.cat(parts), lg, atol=2e-5, rtol=2e-4)
            n_attn_rec = sum(1 for i in range(3, 7) if lt[i] != "mlp")
            n_attn_out = sum(1 for i in list(range(0, 3)) + list(range(7, 10)) if lt[i] != "mlp")
            assert len(caches) == n_attn_out + n_attn_rec * L, (len(caches), L)
            # causality: changing a later token does not alter earlier logits
            ids2 = ids.clone(); ids2[-1] = (ids2[-1] + 1) % 40
            torch.testing.assert_close(m.logits(ids2)[:-1], lg[:-1])
        assert not torch.allclose(outs[1], outs[2]) and not torch.allclose(outs[2], outs[3])
        # sliding window actually matters: window 4 vs window 64 differ
        m = DoryRef(d, loops=2); a = m.logits(ids); m.window = 64; b = m.logits(ids)
        assert not torch.allclose(a, b)
        # last_loop mode: single-shot prefill identical to per_loop; shares caches
        m1 = DoryRef(d, loops=3, kv_mode="last_loop")
        torch.testing.assert_close(m1.logits(ids), outs[3])
        c = {}; m1.logits(ids, 0, c)
        assert len(c) == sum(1 for t in lt if t != "mlp"), len(c)  # one cache per attn layer
        # justnorm sanity
        x = torch.randn(5, 7); assert torch.allclose(justnorm(x).norm(dim=-1), torch.ones(5), atol=1e-3)
        # rope: position 0 is identity; rotation preserves norm
        v = torch.randn(3, 2, 8); pos = torch.tensor([0, 1, 2])
        r = rope_neox(v, pos, 100.0, 8)
        torch.testing.assert_close(r[0], v[0]); torch.testing.assert_close(r.norm(dim=-1), v.norm(dim=-1))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--loops", type=int)
    ap.add_argument("--ids"); ap.add_argument("--ids-file"); ap.add_argument("--out")
    ap.add_argument("--topk", action="store_true"); ap.add_argument("--kv-mode")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.ids_file:
        txt = open(a.ids_file).read().replace("\n", ",").replace(" ", ",")
    else:
        txt = a.ids
    ids = [int(x) for x in txt.split(",") if x.strip()]
    m = DoryRef(a.model, a.loops, a.kv_mode)
    lg = m.logits(ids)
    if a.out:
        np.save(a.out, lg.numpy().astype(np.float32))
    if a.topk:
        print("top5:", torch.topk(lg[-1], 5).indices.tolist())
    print("logits", tuple(lg.shape), "finite", bool(torch.isfinite(lg).all()))


if __name__ == "__main__":
    main()
