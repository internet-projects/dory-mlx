# dory-mlx

Run NVIDIA's [Nemotron-Labs Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) (released 16 October 2026) on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

Dory is a looped 4B reasoning model: a shared block of layers runs 1 to 4 times (2 by default), so one set of weights gives you a speed-versus-accuracy dial. This package adds the `dory` architecture to MLX-LM (`dory_mlx/model.py`) with a small loader and chat helper. First released 17 October 2026.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with NVIDIA.

## Install and run

```bash
pip install "git+https://github.com/internet-projects/dory-mlx"
python -m dory_mlx "Explain why the sky is blue in three sentences." --loops 2
```

The default model is [netprojects/nemotron-dory1.0-4b-mlx-8bit](https://huggingface.co/netprojects/nemotron-dory1.0-4b-mlx-8bit) (4.2 GB); the [bf16 build](https://huggingface.co/netprojects/nemotron-dory1.0-4b-mlx-bf16) (8 GB) is also available.

```python
import dory_mlx
model, tok = dory_mlx.load("netprojects/nemotron-dory1.0-4b-mlx-8bit", loops=2)
for piece in dory_mlx.chat(model, tok, "What is 17 * 23?", max_tokens=1500):
    print(piece, end="")
```

The model reasons first (`<think>...</think>`) and then answers. Defaults follow NVIDIA's recommendation (temperature 1.0, top_p 0.95). `dory_mlx.load` builds the tokenizer directly, so it works with the current transformers releases.

## Convert your own

```bash
python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-8bit --bits 8
python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-bf16 --bits 16
```

The converter fuses the query/key/value/gate projections and the MLP gate/up projections for faster decoding.

## Check a build

```bash
pip install torch safetensors numpy         # only for the reference comparison
python scripts/check.py out-8bit             # cache and sliding-window consistency
python scripts/check.py out-bf16 --ref /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16   # also vs an independent fp32 PyTorch implementation
pip install pytest && python -m pytest tests # cache tests on a tiny random model (no download, CPU or Metal)
```

`scripts/dory_ref_torch.py` is a plain PyTorch implementation written from the vLLM code, independent of the MLX one, used to check the port.

## What we verified

| Check | Result |
|---|---|
| Logits vs the independent fp32 PyTorch implementation (real weights, 2 loops, 55 tokens) | top-1 agreement 100%, probability drift (KL) 0.001 to 0.003 |
| 8-bit vs bf16 (5 short texts, 2 loops) | top-1 agreement 98%, KL 0.003 |
| WikiText-2 test perplexity, loops 1 / 2 / 4 | bf16 12.17 / 11.76 / 11.69, 8-bit 12.18 / 11.77 / 11.70 |
| Sliding-window caches, window shrunk to 64 | top-1 agreement at least 98.7%, KL at most 0.0006 |

For NVIDIA's own accuracy results, see the [original model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16). Recommended build: 8-bit, which stays very close to bf16 (KL 0.003, top-1 agreement 98%) and is half the size.

## Speed and memory

Measured on a MacBook Pro M5 Max (128 GB) on 20 October 2026, with no other models running; medians of repeated runs (run-to-run spread about 2%).

**8-bit**

| Loops | Generation, short prompt | Generation after a 32K-token prompt | Reading a 32K-token prompt | Peak memory, 32K-token prompt |
|---|---|---|---|---|
| 1 | 101 tokens/s | 82 tokens/s | 3,200 tokens/s (10 s) | 7.7 GB |
| 2 | 69 tokens/s | 54 tokens/s | 1,980 tokens/s (17 s) | 9.0 GB |
| 4 | 39 tokens/s | 31 tokens/s | 1,070 tokens/s (31 s) | 11.5 GB |

**bf16**

| Loops | Generation, short prompt | Generation after a 32K-token prompt | Reading a 32K-token prompt | Peak memory, 32K-token prompt |
|---|---|---|---|---|
| 1 | 58 tokens/s | 55 tokens/s | 3,270 tokens/s (10 s) | 11.3 GB |
| 2 | 39 tokens/s | 36 tokens/s | 2,180 tokens/s (15 s) | 12.6 GB |
| 4 | 23 tokens/s | 22 tokens/s | 1,270 tokens/s (26 s) | 15.1 GB |

With the Dory model file proposed for MLX-LM itself ([mlx-lm PR 1988](https://github.com/ml-explore/mlx-lm/pull/1988)), MLX-LM's batch generation serves several requests at once: about 200 tokens/s in total for 4 requests and 450 tokens/s for 16 (8-bit, 2 loops).

## Notes

English text in, text out; replies include the reasoning chain. Verified with prompts up to 24K tokens. Cache modes: `per_loop` (default, matches NVIDIA's launch commands) keeps a cache for each loop of the shared layers, and `last_loop` keeps only the last loop's (less memory at 4 loops, slightly different outputs). In both modes the sliding-window layers outside the loop keep only the last 1,024 tokens.

## How it was written

From the vLLM implementation of Dory (vllm-project/vllm PR 60149), cross-checked against the llama.cpp implementation: the unit-norm (nGPT) residual updates, query/key norm after RoPE, output gate and loop re-injection follow those sources.

## More from NetProjects

[netprojects.com](https://netprojects.com) · [Insights](https://netprojects.com/insights) · [H2O-Lightning-4B on MLX](https://github.com/internet-projects/h2o-lightning-mlx) · [Hugging Face](https://huggingface.co/netprojects) · [How we use AI](https://netprojects.com/standards/ai)

## License

Code: Apache-2.0 (see `LICENSE`). The model weights are NVIDIA's, under OpenMDW-1.1; see NVIDIA's model card. Credit: Paris Michalas, Partner, NetProjects (Internet Projects Ltd).
