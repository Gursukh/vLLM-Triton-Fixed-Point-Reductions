"""Tensor-core GEMM. Per-K-tile mma → fp32 partial → fxp → int accumulator.

Two paths:

  SPLIT_K == 1 (fast path, prefill-shaped problems)
    Persistent kernel: 1D grid of NUM_SMS programs walks the output tiles
    cyclically. Each program walks the full K range itself and writes the fp
    output directly with bias fused.

  SPLIT_K > 1 (decode-shaped / tall-skinny problems)
    Each output tile is sliced into SPLIT_K K-ranges. Programs atomic-add
    their int partial into a transient int buffer; a tiny epilogue divides
    by 2^frac, adds bias, casts to the output dtype. Integer atomic-add is
    commutative, so the result is bit-exact regardless of CTA finish order
    — this is the determinism win that pure-fp implementations can't claim.

BLOCK_K is fixed per dtype (32 fp16/bf16, 16 fp32) — that's the determinism
granularity. Split boundaries are K-tile-aligned, so permuting tiles across
splits gives the same int sum.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


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

# Shape-adaptive configs. We dispatch by problem shape rather than autotuning
# at call time, so all bit-determinism guarantees hold (same shape → same
# config → same mma sequence). Every config keeps BM, BN ≥ 16 so tl.dot
# lowers to the same mma.m16n8k16 family for fp16/bf16 (mma.m16n8k8 for
# fp32 TF32) — that's what makes per-(m, n) fp32 partials bit-identical
# across configs and preserves batch invariance even when M crosses the
# skinny / default threshold.
#
# Config tuple: (BLOCK_M, BLOCK_N, GROUP_SIZE_M, num_warps, num_stages)
_CFG_DEFAULT = (128, 128, 8, 8, 3)   # activation matmuls (M, N both medium-large)
_CFG_SKINNY_M = (16, 128, 8, 4, 3)   # tiny M, big N — LM head decode shapes
_BLOCKS_PER_SM = 2

# Epilogue tile; small enough to launch a fresh kernel cheaply.
_EPI_BLOCK_M = 64
_EPI_BLOCK_N = 64

# Soft cap on split-K. Beyond this, the atomic contention and zero-init cost
# stop paying for themselves.
_MAX_SPLIT_K = 16


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


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
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,  # actually the launched grid size = persistent stride
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        offs_m_raw = start_m + tl.arange(0, BLOCK_M)
        offs_n_raw = start_n + tl.arange(0, BLOCK_N)

        offs_am = tl.where(offs_m_raw < M, offs_m_raw, 0)
        offs_bn = tl.where(offs_n_raw < N, offs_n_raw, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

            k_mask = offs_k_for_mask < K - ki * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
            fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)

        c_fp32 = fixed_to_float(fxp_acc, INV_SCALE)
        if HAS_BIAS:
            bias_vals = tl.load(
                bias_ptr + offs_n_raw, mask=offs_n_raw < N, other=0.0
            ).to(tl.float32)
            c_fp32 = c_fp32 + bias_vals[None, :]

        c_ptrs = c_ptr + offs_m_raw[:, None] * stride_cm + offs_n_raw[None, :] * stride_cn
        c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
        tl.store(c_ptrs, c_fp32.to(IO_DTYPE), mask=c_mask)


@triton.jit
def _gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_int_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cim,
    stride_cin,
    tiles_per_split,
    num_k_tiles,
    SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles_mn = num_pid_m * num_pid_n
    num_tiles = num_tiles_mn * SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_k = tile_id // num_tiles_mn
        tile_id_mn = tile_id % num_tiles_mn
        pid_m, pid_n = _compute_pid(
            tile_id_mn, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )

        k_tile_start = pid_k * tiles_per_split
        k_tile_end = tl.minimum(k_tile_start + tiles_per_split, num_k_tiles)

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        offs_m_raw = start_m + tl.arange(0, BLOCK_M)
        offs_n_raw = start_n + tl.arange(0, BLOCK_N)

        offs_am = tl.where(offs_m_raw < M, offs_m_raw, 0)
        offs_bn = tl.where(offs_n_raw < N, offs_n_raw, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)
        for ki in range(k_tile_start, k_tile_end):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

            k_mask = offs_k_for_mask < K - ki * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
            fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)

        # Integer atomic-add: commutative, so race-free determinism.
        c_ptrs = c_int_ptr + offs_m_raw[:, None] * stride_cim + offs_n_raw[None, :] * stride_cin
        c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
        tl.atomic_add(c_ptrs, fxp_acc, mask=c_mask, sem="relaxed")


@triton.jit
def _gemm_epilogue_kernel(
    c_int_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    stride_cim,
    stride_cin,
    stride_cm,
    stride_cn,
    INV_SCALE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    ci_ptrs = c_int_ptr + offs_m[:, None] * stride_cim + offs_n[None, :] * stride_cin
    fxp = tl.load(ci_ptrs, mask=mask, other=0)
    c_fp = fixed_to_float(fxp, INV_SCALE)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        c_fp = c_fp + bias_vals[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c_fp.to(IO_DTYPE), mask=mask)


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


def _pick_split_k(
    num_tiles_mn: int, num_k_tiles: int, num_sms: int, int_bits: int
) -> int:
    # int16 atomics aren't broadly supported; widen by skipping split-K.
    if int_bits == 16:
        return 1
    if num_tiles_mn >= num_sms or num_k_tiles <= 1:
        return 1
    split = max(1, num_sms // max(1, num_tiles_mn))
    return min(_MAX_SPLIT_K, split, num_k_tiles)


def _pick_config(M: int) -> tuple[int, int, int, int, int]:
    # Skinny path: tiny batch. Activates for LM head decode shapes
    # (M=1..16) where BM=128 would waste 87%+ of every M-tile.
    if M <= 16:
        return _CFG_SKINNY_M
    return _CFG_DEFAULT


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

    scale, inv_scale, qmin, qmax, tl_int_dtype, torch_int_dtype = fxp_constants(
        int_bits, fxp_frac_bits
    )
    io_dtype = _TORCH_TO_TL_FLOAT[a.dtype]
    block_k = _BLOCK_K_BY_DTYPE[a.dtype]
    allow_tf32 = a.dtype == torch.float32

    block_m, block_n, group_size_m, num_warps, num_stages = _pick_config(M)

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    target_programs = num_sms * _BLOCKS_PER_SM
    num_tiles_mn = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    num_k_tiles = triton.cdiv(K, block_k)
    split_k = _pick_split_k(num_tiles_mn, num_k_tiles, target_programs, int_bits)

    if split_k == 1:
        grid_size = min(target_programs, num_tiles_mn)
        grid = (grid_size,)
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE_M=group_size_m,
            NUM_SMS=grid_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return c

    c_int = torch.zeros((M, N), device=a.device, dtype=torch_int_dtype)
    tiles_per_split = triton.cdiv(num_k_tiles, split_k)
    total_tiles = num_tiles_mn * split_k
    grid_size = min(target_programs, total_tiles)
    grid = (grid_size,)
    _gemm_splitk_kernel[grid](
        a,
        b,
        c_int,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_int.stride(0),
        c_int.stride(1),
        tiles_per_split,
        num_k_tiles,
        SCALE=scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        ALLOW_TF32=allow_tf32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
        SPLIT_K=split_k,
        NUM_SMS=grid_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    grid_epi = (triton.cdiv(N, _EPI_BLOCK_N), triton.cdiv(M, _EPI_BLOCK_M))
    _gemm_epilogue_kernel[grid_epi](
        c_int,
        c,
        bias if bias is not None else a,
        M,
        N,
        c_int.stride(0),
        c_int.stride(1),
        c.stride(0),
        c.stride(1),
        INV_SCALE=inv_scale,
        IO_DTYPE=io_dtype,
        HAS_BIAS=bias is not None,
        BLOCK_M=_EPI_BLOCK_M,
        BLOCK_N=_EPI_BLOCK_N,
    )
    return c
