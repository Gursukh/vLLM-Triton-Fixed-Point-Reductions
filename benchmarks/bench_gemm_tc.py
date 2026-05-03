"""Compare the fxpr GEMM against torch.matmul on vLLM-shaped problems.

Run: python -m benchmarks.bench_gemm_tc
"""

from __future__ import annotations

import statistics
from typing import Iterable

import torch

import fxpr_vllm._cuda  # noqa: F401
from tests.fixed_point_helpers import gemm_dtype_supported


H = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
FFN = 3072

SHAPES = [
    {"label": "qkv_proj",    "M": 1024, "K": H,                    "N": N_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM},
    {"label": "o_proj",      "M": 1024, "K": N_HEADS * HEAD_DIM,   "N": H},
    {"label": "gate_up_proj", "M": 1024, "K": H,                   "N": 2 * FFN},
    {"label": "down_proj",   "M": 1024, "K": FFN,                  "N": H},
    {"label": "decode_qkv",  "M": 1,    "K": H,                    "N": N_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM},
    {"label": "decode_ffn",  "M": 1,    "K": H,                    "N": 2 * FFN},
    {"label": "square_4k",   "M": 4096, "K": 4096,                 "N": 4096},
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _time(fn, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


def _bench_one(shape: dict, dtype: torch.dtype) -> dict:
    M, K, N = shape["M"], shape["K"], shape["N"]
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    fxpr = lambda: torch.ops.fxpr.gemm_fxp(a, b, None, 32, 16)
    torch_mm = lambda: torch.matmul(a, b)

    fxpr_ms = _time(fxpr)
    torch_ms = _time(torch_mm)
    flops = 2.0 * M * N * K

    return {
        "label": shape["label"],
        "shape": f"({M}, {K}) @ ({K}, {N})",
        "dtype": str(dtype).removeprefix("torch."),
        "fxpr_ms": fxpr_ms,
        "torch_ms": torch_ms,
        "fxpr_tflops": flops / fxpr_ms / 1e9,
        "torch_tflops": flops / torch_ms / 1e9,
        "speedup_vs_torch": torch_ms / fxpr_ms,
    }


def _supported_dtypes() -> Iterable[torch.dtype]:
    for d in DTYPES:
        if gemm_dtype_supported(d):
            yield d


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("benchmark needs CUDA")
    cap_major, cap_minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"# GPU: {name} (sm_{cap_major}{cap_minor})")
    print()

    header = ["label", "shape", "dtype", "fxpr_ms", "torch_ms",
              "fxpr_TF", "torch_TF", "fxpr/torch"]
    print("  ".join(f"{h:>14}" for h in header))
    for dtype in _supported_dtypes():
        for shape in SHAPES:
            r = _bench_one(shape, dtype)
            row = [
                f"{r['label']:>14}",
                f"{r['shape']:>14}",
                f"{r['dtype']:>14}",
                f"{r['fxpr_ms']:>14.4f}",
                f"{r['torch_ms']:>14.4f}",
                f"{r['fxpr_tflops']:>14.1f}",
                f"{r['torch_tflops']:>14.1f}",
                f"{r['speedup_vs_torch']:>14.2f}",
            ]
            print("  ".join(row))


if __name__ == "__main__":
    main()
