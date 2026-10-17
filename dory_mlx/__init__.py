"""dory-mlx: run NVIDIA Nemotron-Labs Dory 1.0 4B on Apple Silicon with MLX."""
import sys
from pathlib import Path

__version__ = "0.1.0"


def register():
    """Make mlx-lm find the `dory` architecture (until it ships in mlx-lm itself)."""
    from . import model as _model
    sys.modules.setdefault("mlx_lm.models.dory", _model)
    return _model


def load(path_or_repo, loops=2, kv_mode="per_loop"):
    """Load a Dory MLX build (local folder or Hugging Face repo id). Returns (model, tokenizer).

    loops: 1-4 passes through the shared block (2 is NVIDIA's default; more loops = slower, usually more accurate).
    kv_mode: "per_loop" (default, matches NVIDIA's launch commands) or "last_loop" (smaller KV cache).
    """
    register()
    from huggingface_hub import snapshot_download
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import load_model
    from transformers import PreTrainedTokenizerFast

    p = Path(path_or_repo)
    if not p.exists():
        p = Path(snapshot_download(path_or_repo))
    model, _ = load_model(p, model_config={"n_recurrent_loops": int(loops), "recurrent_kv_cache_mode": kv_mode})
    # mlx_lm.load goes through transformers' AutoConfig, which rejects Dory's `mlp` layer type; load the tokenizer directly.
    tok = PreTrainedTokenizerFast(tokenizer_file=str(p / "tokenizer.json"), bos_token="<s>", eos_token="<|im_end|>", pad_token="<|im_end|>")
    tok.chat_template = (p / "chat_template.jinja").read_text()
    return model, TokenizerWrapper(tok, eos_token_ids={2, 11})


def chat(model, tokenizer, messages, max_tokens=2048, temp=1.0, top_p=0.95, thinking=True):
    """Yield text pieces. `messages` is a string or a list of {"role", "content"} dicts.
    NVIDIA recommends temperature 1.0 and top_p 0.95. The model reasons first (inside <think>...</think>) unless thinking=False."""
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    for r in stream_generate(model, tokenizer, prompt=tokenizer.encode(prompt, add_special_tokens=False), max_tokens=max_tokens,
                             sampler=make_sampler(temp=temp, top_p=top_p)):
        if r.token in (2, 11):
            break
        yield r.text
