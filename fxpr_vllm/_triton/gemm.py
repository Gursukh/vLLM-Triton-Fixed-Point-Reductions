"""Tensor-core GEMM. Per-K-tile mma → fp32 partial → fxp → int accumulator.

Integer add is commutative, so permuting K-tiles gives bit-exact output, and
each (BLOCK_M, BLOCK_N) program walks the full K loop on its own, so a row's
result doesn't depend on what other rows are in the batch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K_BY_DTYPE = {
    torch.float16: 32,
    torch.bfloat16: 32,
    torch.float32: 16,
}
_SUPPORTED_DTYPES = tuple(_BLOCK_K_BY_DTYPE.keys())

_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_fp32 = fixed_to_float(fxp_acc, INV_SCALE)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        c_fp32 = c_fp32 + bias_vals[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c_fp32.to(IO_DTYPE), mask=c_mask)


def _cap() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _check_arch(dtype: torch.dtype) -> None:
    cap = _cap()
    if cap < 75:
        raise RuntimeError(
            f"gemm_fxp: requires compute capability >= 7.5 (Turing); device is "
            f"{cap // 10}.{cap % 10}"
        )
    if dtype in (torch.bfloat16, torch.float32) and cap < 80:
        name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        raise RuntimeError(
            f"gemm_fxp: {name} inputs require compute capability >= 8.0 "
            f"(Ampere); device is {cap // 10}.{cap % 10}. Use float16 inputs."
        )


def gemm_fxp_run(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    int_bits: int,
    fxp_frac_bits: int,
) -> torch.Tensor:
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("gemm_fxp: inputs must be CUDA")
    if a.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"gemm_fxp: unsupported input dtype {a.dtype}; "
            f"expected one of {_SUPPORTED_DTYPES}"
        )
    if b.dtype != a.dtype:
        raise TypeError("gemm_fxp: a and b must share dtype")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("gemm_fxp: 2D inputs required")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"gemm_fxp: shape mismatch {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if bias is not None:
        if not bias.is_cuda:
            raise RuntimeError("gemm_fxp: bias must be CUDA")
        if bias.dim() != 1:
            raise ValueError("gemm_fxp: bias must be 1-D")
        if bias.shape[0] != b.shape[1]:
            raise ValueError(
                f"gemm_fxp: bias size {bias.shape[0]} does not match N={b.shape[1]}"
            )
        if bias.dtype != a.dtype:
            raise TypeError("gemm_fxp: bias dtype must match a dtype")
    _check_arch(a.dtype)

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c

    scale, inv_scale, qmin, qmax, tl_int_dtype, _ = fxp_constants(
        int_bits, fxp_frac_bits
    )
    io_dtype = _TORCH_TO_TL_FLOAT[a.dtype]
    block_k = _BLOCK_K_BY_DTYPE[a.dtype]
    allow_tf32 = a.dtype == torch.float32

    grid = (triton.cdiv(N, _BLOCK_N), triton.cdiv(M, _BLOCK_M))
    _gemm_kernel[grid](
        a,
        b,
        # Triton wants a real tensor; only read under HAS_BIAS.
        bias if bias is not None else a,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        IO_DTYPE=io_dtype,
        HAS_BIAS=bias is not None,
        ALLOW_TF32=allow_tf32,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
    )
    return c
