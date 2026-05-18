"""Tensor-core fixed-point GEMM: a persistent kernel for prefill-shaped
problems, a split-K kernel for tall-skinny decode shapes. Both quantise each
per-K-tile fp32 partial to a fixed-point integer and accumulate in integers,
so the result is bit-identical however the problem is tiled or split.
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

# Configs are picked by problem shape, not autotuned, so a shape always gets
# the same mma sequence. Tuple is (BLOCK_M, BLOCK_N, GROUP_SIZE_M, num_warps,
# num_stages); BM, BN stay at least 16 so the fp32 partials are bit-identical.
_CFG_DEFAULT = (128, 128, 8, 8, 3)   # activation matmuls, M and N both large
_CFG_SKINNY_M = (16, 128, 8, 4, 3)   # tiny M, big N, e.g. lm_head decode
_BLOCKS_PER_SM = 2

# Soft cap on split-K. Beyond this, the atomic contention and zero-init cost
# stop paying for themselves.
_MAX_SPLIT_K = 16

# Persistent split-K scratch, reused across calls. c_int is bounded by
# target_programs * 128*128 ints; the epilogue self-zeroes it.
_TILE_M_MAX = 128
_TILE_N_MAX = 128
_splitk_scratch: dict[
    tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def _get_splitk_scratch(
    device: torch.device, int_dtype: torch.dtype, target_programs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent (c_int_flat, locks_flat) scratch for the split-K path.
    Sized for the worst case, zeroed once; the kernel epilogue zeroes its own
    footprint so it stays clean. Allocated in warmup before graph capture and
    fixed in size, so it never moves."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, int_dtype)
    cached = _splitk_scratch.get(key)
    if cached is None:
        c_int = torch.zeros(
            target_programs * _TILE_M_MAX * _TILE_N_MAX,
            device=device,
            dtype=int_dtype,
        )
        locks = torch.zeros(target_programs, device=device, dtype=torch.int32)
        cached = (c_int, locks)
        _splitk_scratch[key] = cached
    return cached


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# do_not_specialize M: it changes per request, and a recompile per M class
# would spike TTFT. M is only a bound/mask, so specializing it buys nothing.
@triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])
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


# do_not_specialize the shape-varying args so this compiles once. SPLIT_K is a
# runtime arg now (was a constexpr; split_k depends on M, so unenumerable).
@triton.jit(
    do_not_specialize=["M", "SPLIT_K", "tiles_per_split"],
    do_not_specialize_on_alignment=["M", "SPLIT_K", "tiles_per_split"],
)
def _gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_int_ptr,
    c_ptr,
    lock_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cim,
    stride_cin,
    stride_cm,
    stride_cn,
    tiles_per_split,
    num_k_tiles,
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
    SPLIT_K,
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
        c_int_ptrs = (
            c_int_ptr + offs_m_raw[:, None] * stride_cim + offs_n_raw[None, :] * stride_cin
        )
        c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
        tl.atomic_add(c_int_ptrs, fxp_acc, mask=c_mask, sem="relaxed")

        # Fused epilogue: the last of the SPLIT_K programs to arrive owns it.
        # The acq_rel increment makes its load of c_int see every split's
        # atomic-add, so the output is the same regardless of arrival order.
        arrived = tl.atomic_add(lock_ptr + tile_id_mn, 1, sem="acq_rel")
        if arrived == SPLIT_K - 1:
            fxp = tl.load(c_int_ptrs, mask=c_mask, other=0)
            c_fp32 = fixed_to_float(fxp, INV_SCALE)
            if HAS_BIAS:
                bias_vals = tl.load(
                    bias_ptr + offs_n_raw, mask=offs_n_raw < N, other=0.0
                ).to(tl.float32)
                c_fp32 = c_fp32 + bias_vals[None, :]
            c_ptrs = (
                c_ptr + offs_m_raw[:, None] * stride_cm + offs_n_raw[None, :] * stride_cn
            )
            tl.store(c_ptrs, c_fp32.to(IO_DTYPE), mask=c_mask)

            # Zero this tile's scratch for the next launch. All splits have
            # landed and nothing else touches it, so this is race-free.
            tl.store(
                c_int_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE), mask=c_mask
            )
            tl.store(lock_ptr + tile_id_mn, 0)


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


# Cached SM count; gemm_fxp_run runs for every linear layer.
_SM_COUNT_CACHE: dict[int, int] = {}


def _device_sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SM_COUNT_CACHE.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT_CACHE[idx] = n
    return n


def _pick_split_k(
    num_pid_m: int,
    num_tiles_mn: int,
    num_k_tiles: int,
    num_sms: int,
    int_bits: int,
) -> int:
    # int16 atomics aren't broadly supported; widen by skipping split-K.
    if int_bits == 16:
        return 1
    # Split-K only helps decode (num_pid_m == 1, idle SMs). On prefill it just
    # adds atomic contention, ~2x slower, so keep the persistent kernel.
    if num_pid_m > 1 or num_tiles_mn >= num_sms or num_k_tiles <= 1:
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

    num_sms = _device_sm_count(a.device)
    target_programs = num_sms * _BLOCKS_PER_SM
    num_pid_m = triton.cdiv(M, block_m)
    num_tiles_mn = num_pid_m * triton.cdiv(N, block_n)
    num_k_tiles = triton.cdiv(K, block_k)
    split_k = _pick_split_k(num_pid_m, num_tiles_mn, num_k_tiles, num_sms, int_bits)

    if split_k == 1:
        # Fixed grid: NUM_SMS stays one constexpr per GPU, so it compiles once.
        grid_size = target_programs
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

    # Persistent scratch (epilogue keeps it clean); the tight (M, N) view fits
    # since M * N <= num_tiles_mn * 128*128.
    c_int_flat, locks_flat = _get_splitk_scratch(
        a.device, torch_int_dtype, target_programs
    )
    c_int = c_int_flat[: M * N].view(M, N)
    locks = locks_flat[:num_tiles_mn]
    tiles_per_split = triton.cdiv(num_k_tiles, split_k)
    # Fixed persistent grid, as in the split_k == 1 path.
    grid_size = target_programs
    grid = (grid_size,)
    _gemm_splitk_kernel[grid](
        a,
        b,
        # Triton wants a real tensor; only read under HAS_BIAS.
        bias if bias is not None else a,
        c_int,
        c,
        locks,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_int.stride(0),
        c_int.stride(1),
        c.stride(0),
        c.stride(1),
        tiles_per_split,
        num_k_tiles,
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
        SPLIT_K=split_k,
        NUM_SMS=grid_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
