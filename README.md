# Fixed-Point Reductions (FxPR) for vLLM

FxPR for vLLM allows for deterministic inference by removing the non-determinism introduced by floating-point non-associativity. Every reduction is performed as an integer sum on values that have been cast to a signed fixed-point representation up-front. Integer addition is associative, so the result is bitwise-identical regardless of how the work is split across SMs, warps, or KV-cache splits.

The kernels are written in CUDA C++ / inline PTX (no Triton, no CUTLASS, no cuBLAS). All five reductions — RMSNorm, log-softmax, fp32 GEMM, int8 GEMM with per-row/per-column scales, and unified prefill+decode attention — live in [csrc/](csrc/) and are exposed to Python via `torch.ops.fxpr.*`.

## How it works

Every reduction follows the same pattern:

1. Cast each float operand to a signed fixed-point integer via `float_to_fixed(x, frac_bits, int_dtype)` — round-to-nearest-even with saturation.
2. Do the reduction as an integer sum.
3. Cast the integer accumulator back to float via `fixed_to_float`.

Because the intermediate accumulator is an integer with a fixed scale, reordering the additions cannot change the result.

For tier-2 throughput on tensor-core hardware, the int8 GEMM uses pre-computed split-invariant per-row (activation) and per-column (weight) scales. Scales are computed before the kernel sees its K range, so any KV-split or grid-shape change leaves the final integer accumulator unchanged. See [Migrate.md](Migrate.md) section E for the plumbing rules.

## Installation

### Prebuilt wheels (recommended)

Each tagged release ships binary wheels for the most common (Python, torch, CUDA) combinations on the [Releases page](https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM/releases). Pick the wheel matching your environment — the filename encodes Python ABI, torch version, and CUDA version, e.g.

```
fxpr_vllm-0.1.0-cp312-cp312-linux_x86_64+torch2.6.0cu124.whl
   |        |    |              |             |        |
   version  py3.12 ABI          platform      torch    CUDA
```

For Colab (Python 3.12, torch 2.6+, CUDA 12.4):

```python
!pip install https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM/releases/download/<TAG>/fxpr_vllm-0.1.0-cp312-cp312-linux_x86_64+torch2.6.0cu124.whl
```

The wheel covers `sm_75 → sm_90` (Turing through Hopper); no compilation needed at install time.

### Build from source

If no prebuilt wheel matches your env, compile locally. nvcc must match the CUDA toolkit your local `torch` was built against (run `python -c "import torch; print(torch.version.cuda)"` and install a matching toolkit if necessary).

```bash
pip install --no-build-isolation git+https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM.git
```

`--no-build-isolation` is required: torch C++ extensions need to compile against the same torch ABI that the runtime will load, and pip's default isolated build venv pulls a fresh, possibly mismatched torch.

To narrow the build (single-arch is much faster), set `TORCH_CUDA_ARCH_LIST` before installing — e.g. `TORCH_CUDA_ARCH_LIST=8.9` for an L4. `MAX_JOBS=4` enables parallel compilation.

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

### Runtime configuration

See [config.py](fxpr_vllm/vllm_modules/config.py).

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_FXP_FRAC_BITS` | `16` | Number of fractional bits in the Q-format (higher = finer resolution, smaller dynamic range). |
| `VLLM_FXP_INT_BITS` | `32` | Accumulator width. One of `16`, `32`, `64`. |
| `VLLM_FXP_NUM_KV_SPLITS` | `8` | Number of KV splits for the decode attention kernel. |
| `VLLM_FXP_DET_RMSNORM` | on | Replace RMSNorm with the deterministic kernel. |
| `VLLM_FXP_DET_GEMM` | on | Register `fixed_point_det` quantisation method. |
| `VLLM_FXP_DET_ATTN` | on | Bind the deterministic attention backend. |
| `VLLM_FXP_DET_LOGPROBS` | on | Patch `Sampler.compute_logprobs`. |

## Architecture

```
csrc/
  bindings.cpp          # TORCH_LIBRARY registrations (op surface)
  fixed_point.cuh       # device float<->fixed helpers
  ptx_mma.cuh           # arch-dispatched inline-PTX mma.sync wrappers
  casts.cu              # float_to_fixed / fixed_to_float ops
  scales.cu             # per-row max-abs scale reduction
  rms_norm.cu           # RMSNorm (+ fused residual)
  softmax.cu            # log-softmax
  gemm.cu               # scalar fp32 GEMM + int8 per-row GEMM
  attention.cu          # unified prefill+decode, paged KV
fxpr_vllm/
  _cuda                 # built CUDA extension (.so)
  library_ops.py        # Python re-export shim over torch.ops.fxpr.*
  register.py           # vLLM plugin entry point
  monkey_patches.py     # patches for parts vLLM doesn't expose
  vllm_modules/         # quantisation, attention impl, RMSNorm, sampler
```
