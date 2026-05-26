# Fixed-Point Reductions for vLLM

Drop-in plugin that makes vLLM's reductions bitwise reproducible. The trick is
boring: cast to a signed fixed-point integer up front, do the reduction in
integer space, cast back. Integer addition is associative, so it doesn't matter
how the work gets sliced across SMs, warps, or KV splits, you get the same
bits every time.


## Install

```bash
pip install git+https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM.git
```


Requirements: `torch>=2.6`, `triton>=3.0`, `vllm`. GEMM needs sm_75+ (Turing)
for fp16.

## Usage

```python
from vllm import LLM

llm = LLM(
    ...,
    quantization="fixedpoint",
    attention_backend="CUSTOM",
)
```

Registration runs automatically via the `vllm.general_plugins` entry point. No
manual call needed.

### Knobs

All configured via environment variables at process start (see
[`config.py`](fxpr_vllm/config.py)).

| Var                       | Default | Allowed       | What it does |
| ------------------------- | ------- | ------------- | ------------ |
| `FXPR_FRAC_BITS`          | `16`    | `8`/`16`/`32` | Fractional bits in the Q-format. More = finer resolution, less dynamic range. |
| `FXPR_INT_BITS`           | `32`    | `16`/`32`/`64`| Integer accumulator width. |
| `FXPR_ENABLE_RMS_NORM`    | unset   | `1`           | Opt in to the DeterministicRMSNorm patch (off by default; leaves vLLM's stock RMSNorm in place unless set). |
| `FXPR_ENABLE_LM_HEAD`     | unset   | `1`           | Opt in to the fixed-point lm_head matmul (off by default; requires the `fixedpoint` quant config). |

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
