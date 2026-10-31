"""Cache tests for dory_mlx on a tiny random model (runs on CPU or Metal)."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import KVCache

from dory_mlx.model import Model, ModelArgs, RingKVCache

STD = 0.02


def tiny(mode, window=8, loops=2):
    # 16 layers: input 0-3, recurrent 4-11, output 12-15; attention on even
    # layers, full attention at i % 8 == 6, sliding elsewhere.
    args = ModelArgs(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=window,
        n_input_layers=4,
        n_recurrent_layers=8,
        n_output_layers=4,
        n_recurrent_loops=loops,
        recurrent_kv_cache_mode=mode,
        initializer_range=STD,
    )
    mx.random.seed(0)
    model = Model(args)
    # Store the nGPT scale vectors as a checkpoint does (about initializer_range);
    # with all-ones the attention logits blow up and nothing is comparable.
    scale_names = ("sqk", "s_gate", "suv_gate", "suv_up", "attn_alpha", "mlp_alpha", "logit_scale")
    params = [
        (k, mx.full(v.shape, STD) if k.split(".")[-1] in scale_names else v)
        for k, v in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model


def run_cached(model, tokens, chunk, cache):
    """Chunked prefill of all but the decode tail, then token-by-token decode.
    Returns the logits for each decoded position."""
    n_prefill = 20
    out = []
    for s in range(0, n_prefill, chunk):
        model(tokens[:, s : min(s + chunk, n_prefill)], cache=cache)
    for t in range(n_prefill, tokens.shape[1]):
        out.append(model(tokens[:, t : t + 1], cache=cache)[:, -1])
    return mx.stack(out, axis=1), n_prefill


def test_cache_types():
    a = tiny("last_loop").args
    lo, hi = a.n_input_layers, a.n_input_layers + a.n_recurrent_layers
    plan, _ = tiny("last_loop")._cache_plan()
    caches = tiny("last_loop").make_cache()
    for (i, _l), idx in plan.items():
        sliding = a.layer_types[i] == "sliding_attention"
        expect = RingKVCache if (sliding and not lo <= i < hi) else KVCache
        assert type(caches[idx]) is expect, (i, type(caches[idx]))

    m = tiny("per_loop")
    plan, _ = m._cache_plan()
    caches = m.make_cache()
    for (i, _l), idx in plan.items():
        sliding = m.args.layer_types[i] == "sliding_attention"
        assert type(caches[idx]) is (RingKVCache if sliding else KVCache)


@pytest.mark.parametrize("chunk", [5, 12])
def test_per_loop_past_window_matches_full_pass(chunk):
    m = tiny("per_loop")
    tokens = mx.random.randint(0, 64, (1, 40), key=mx.random.key(1))
    m._all_logits = True
    full = m(tokens)
    m._all_logits = False
    got, n = run_cached(m, tokens, chunk, m.make_cache())
    assert mx.allclose(got, full[:, n:], atol=1e-4).item()


@pytest.mark.parametrize("chunk", [5, 12])
def test_last_loop_past_window_matches_full_kv(chunk):
    # last_loop keeps only the last loop's K/V for recurrent layers, so it differs
    # from an uncached pass by design; compare with the pre-fix all-KVCache layout.
    m = tiny("last_loop")
    tokens = mx.random.randint(0, 64, (1, 40), key=mx.random.key(2))
    new, _ = run_cached(m, tokens, chunk, m.make_cache())
    old_caches = [KVCache() for _ in m.make_cache()]
    old, _ = run_cached(m, tokens, chunk, old_caches)
    assert mx.allclose(new, old, atol=1e-4).item()
    assert (mx.argmax(new, -1) == mx.argmax(old, -1)).all().item()


def test_last_loop_cache_is_smaller():
    m = tiny("last_loop")
    tokens = mx.random.randint(0, 64, (1, 40), key=mx.random.key(3))
    new = m.make_cache()
    m(tokens, cache=new)
    old = [KVCache() for _ in new]
    m(tokens, cache=old)
    assert sum(c.nbytes for c in new) < sum(c.nbytes for c in old)
