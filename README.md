# Fixed-Point Reductions for vLLM

Drop-in plugin that makes vLLM's reductions bitwise reproducible. The trick is
boring: cast to a signed fixed-point integer up front, do the reduction in
integer space, cast back. Integer addition is associative, so it doesn't matter
how the work gets sliced across SMs, warps, or KV splits, you get the same
bits every time.

All the kernels are Triton (`@triton.jit`) in [fxpr_vllm/_triton/](fxpr_vllm/_triton/).
No C++ build — install is pure Python. Surfaced to Python through `torch.ops.fxpr.*`.

## What you get

| Op                              | File              |
| ------------------------------- | ----------------- |
| `float_to_fixed`/`fixed_to_float` casts | `fxpr_vllm/_triton/casts.py` |
| RMSNorm (with fused residual)   | `fxpr_vllm/_triton/rms_norm.py` |
| log-softmax                     | `fxpr_vllm/_triton/softmax.py` |
| fp32/fp16/bf16 GEMM (tensor cores) | `fxpr_vllm/_triton/gemm.py` |
| Unified prefill + decode attention, paged KV | `fxpr_vllm/_triton/attention.py` |

## Install

```bash
pip install git+https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM.git
```

No build step, no `TORCH_CUDA_ARCH_LIST`, no `nvcc`. Triton compiles each
kernel on first call and caches the result.

Requirements: `torch>=2.6`, `triton>=3.0`, `vllm`. GEMM needs sm_75+ (Turing)
for fp16; bf16/fp32 inputs need sm_80+ (Ampere).

## Usage

```python
from vllm import LLM
from fxpr_vllm.register import register

register()

llm = LLM(
    ...,
    quantization="fixed_point_det",
    attention_backend="CUSTOM",
)
```

### Knobs

All configured via environment variables at process start (see
[`config.py`](fxpr_vllm/config.py)).

| Var                       | Default | Allowed       | What it does |
| ------------------------- | ------- | ------------- | ------------ |
| `VLLM_FXP_FRAC_BITS`      | `16`    | `8`/`16`/`32` | Fractional bits in the Q-format. More = finer resolution, less dynamic range. |
| `VLLM_FXP_INT_BITS`       | `32`    | `16`/`32`/`64`| Integer accumulator width. |
| `VLLM_FXP_NUM_KV_SPLITS`  | `8`     | `>= 1`        | KV splits for the decode attention kernel. |

The `frac_bits` × `int_bits` pair is what fixes the dynamic range. Q16.16 in a
32-bit accumulator gets you about ±32K in original units before saturation;
bump int_bits to 64 if your activations cluster wider.

## Layout

```
fxpr_vllm/
  _lib.py             torch.library schema + CUDA impl registrations
  _triton/
    fxp.py            float <-> fixed device helpers (rint, clamp, cast)
    casts.py          float_to_fixed / fixed_to_float
    rms_norm.py       RMSNorm + residual variant
    softmax.py        log-softmax
    gemm.py           tensor-core GEMM with fxp inter-tile accumulation
    attention.py      unified prefill + decode, paged KV, split-K
  library_ops.py      register_fake meta impls for Dynamo
  register.py         vLLM plugin entry point
  monkey_patches.py   the bits vLLM doesn't expose cleanly
  quantisation_config.py
  attention_backend.py
  rms_norm.py
  sampling.py
```

## Tests

```bash
pytest tests/
```

Needs CUDA. GEMM tests for bf16/fp32 require sm_80+ (Ampere or newer); they
skip otherwise.
