# Fixed-Point Reductions for vLLM

Drop-in plugin that makes vLLM's reductions bitwise reproducible. The trick is
boring: cast to a signed fixed-point integer up front, do the reduction in
integer space, cast back. Integer addition is associative, so it doesn't matter
how the work gets sliced across SMs, warps, or KV splits — you get the same
bits every time.

All the kernels are hand-written CUDA in [csrc/](csrc/). No Triton, no
CUTLASS, no cuBLAS — the whole point is to control every reduction tree.
Surfaced to Python through `torch.ops.fxpr.*`.

## What you get

| Op                              | File              |
| ------------------------------- | ----------------- |
| `float_to_fixed`/`fixed_to_float` casts | `csrc/casts.cu` |
| RMSNorm (with fused residual)   | `csrc/rms_norm.cu` |
| log-softmax                     | `csrc/softmax.cu` |
| fp32/fp16/bf16 GEMM (tensor cores) | `csrc/gemm.cu` |
| Unified prefill + decode attention, paged KV | `csrc/attention.cu` |

## Install

### Wheels

Tagged releases ship wheels for the common (Python, torch, CUDA) combinations
on the [Releases page](https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM/releases).
Filenames encode the ABI:

```
fxpr_vllm-0.1.0-cp312-cp312-linux_x86_64+torch2.6.0cu124.whl
```

Wheels cover `sm_75` through `sm_90` (Turing → Hopper). Colab example
(py3.12 / torch 2.6 / CUDA 12.4):

```bash
pip install https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM/releases/download/<TAG>/fxpr_vllm-0.1.0-cp312-cp312-linux_x86_64+torch2.6.0cu124.whl
```

### From source

If nothing matches your env, build it. Your nvcc has to match the CUDA your
torch was built against — check with:

```bash
python -c "import torch; print(torch.version.cuda)"
```

Then:

```bash
pip install --no-build-isolation git+https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM.git
```

`--no-build-isolation` is not optional: pip's isolated build venv would pull a
fresh torch with a different ABI from the one you'll load at runtime, and the
extension would fail to import.

Builds are slow because every kernel is instantiated 3 × 3 ways (frac_bits ×
int_bits). Narrow the arch list and parallelise:

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 pip install --no-build-isolation .
```

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
csrc/
  bindings.cpp        TORCH_LIBRARY registrations (Python-facing schemas)
  ops.cpp             argument validation
  ops_internal.h      detail::*_run signatures
  fixed_point.cuh     templated device-side float <-> fixed helpers
  casts.cu            float_to_fixed / fixed_to_float
  rms_norm.cu         RMSNorm + residual variant
  softmax.cu          log-softmax
  gemm.cu             tensor-core GEMM with fxp inter-tile accumulation
  attention.cu        unified prefill + decode, paged KV, split-K
fxpr_vllm/
  _cuda               built extension (.so)
  library_ops.py      Python wrappers + meta/fake impls for Dynamo
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
