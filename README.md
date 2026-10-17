# dory-mlx

Run NVIDIA's [Nemotron-Labs Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

Dory is a looped 4B reasoning model: a shared block of layers is run 1 to 4 times (2 by default), so one set of weights gives you a speed/accuracy dial. MLX-LM does not support the `dory` architecture yet; this package adds it (`dory_mlx/model.py`) and a small loader.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with NVIDIA.

## Install and run

```bash
pip install "git+https://github.com/internet-projects/dory-mlx"
python -m dory_mlx "Explain why the sky is blue in three sentences." --loops 2
```

The default model is [netprojects/nemotron-dory1.0-4b-mlx-8bit](https://huggingface.co/netprojects/nemotron-dory1.0-4b-mlx-8bit) (4.2 GB); a [bf16 build](https://huggingface.co/netprojects/nemotron-dory1.0-4b-mlx-bf16) (8 GB) is also available.

```python
import dory_mlx
model, tok = dory_mlx.load("netprojects/nemotron-dory1.0-4b-mlx-8bit", loops=2)
for piece in dory_mlx.chat(model, tok, "What is 17 * 23?", max_tokens=1500):
    print(piece, end="")
```

The model reasons first (`<think>...</think>`) and then answers. Defaults follow NVIDIA's recommendation (temperature 1.0, top_p 0.95). `dory_mlx.load` exists because `mlx_lm.load` goes through transformers' `AutoConfig`, which rejects Dory's `mlp` layer type in `config.json`.

## Convert your own

```bash
python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-8bit --bits 8
python scripts/convert.py nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 out-bf16 --bits 16
```

The converter fuses the query/key/value/gate projections and the MLP gate/up projections (faster decoding), so converted files are not interchangeable with NVIDIA's checkpoint.

## Check a build

```bash
pip install torch safetensors numpy         # only for the reference comparison
python scripts/check.py out-8bit             # cache and sliding-window consistency
python scripts/check.py out-bf16 --ref /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16   # also vs an independent fp32 PyTorch implementation
```

`scripts/dory_ref_torch.py` is a plain PyTorch implementation written from the vLLM code, independent of the MLX one, used only to check the port.

## What we measured

We checked that the port reproduces the original model; we did not re-run NVIDIA's benchmarks (see NVIDIA's model card for accuracy).

| Check | Result |
|---|---|
| Logits vs the independent fp32 PyTorch implementation (real weights, 2 loops, 55 tokens) | top-1 agreement 100%, probability drift (KL) 0.001 to 0.003 |
| 8-bit vs bf16 (5 short texts, 2 loops) | top-1 agreement 98%, KL 0.003 |
| WikiText-2 test perplexity, loops 1 / 2 / 4 | bf16 12.17 / 11.76 / 11.69, 8-bit 12.18 / 11.77 / 11.70 |
| Cache consistency, window shrunk to 64 | top-1 agreement at least 98.7%, KL at most 0.0006 |
| 4-bit (several recipes, including MLX's NVFP4 mode) | KL 0.065 to 0.12, so no 4-bit build is published |

Speed on one MacBook Pro M5 Max (128 GB), single runs, roughly plus or minus 30%: decode about 35 / 26 / 15 tokens/s at 1 / 2 / 4 loops for the 8-bit build and about 27 / 20 / 12 for bf16; peak memory with a 24K-token prompt 7.7 / 9.0 / 11.5 GB (8-bit) and 11.3 / 12.6 / 15.1 GB (bf16).

## Limits

English and text only. Always reasons first. Tested up to a 24K-token prompt (the model supports 256K). Saving prompt caches to disk is not supported. The `last_loop` cache mode is implemented but lightly tested. Decoding is bound by many small GPU operations rather than memory bandwidth, so there is room for custom kernels.

## How it was written

From the vLLM implementation of Dory (vllm-project/vllm PR 60149), cross-checked against the llama.cpp implementation; the unit-norm (nGPT) residual updates, query/key norm after RoPE, output gate and loop re-injection follow those sources.

## License

Code: Apache-2.0 (see `LICENSE`). The model weights are NVIDIA's, under OpenMDW-1.1; see NVIDIA's model card. Credit: Paris Michalas, Partner, NetProjects (Internet Projects Ltd).
