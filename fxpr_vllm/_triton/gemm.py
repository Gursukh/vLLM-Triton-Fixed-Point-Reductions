"""Tensor-core fixed-point GEMM: persistent kernel for prefill shapes, split-K
kernel for tall-skinny decode shapes. Both quantise each per-K-tile fp32 partial
to a fixed-point integer and accumulate in integers, so the result is
bit-identical however the problem is tiled or split.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


# BLOCK_K is the determinism granularity: each kernel quantises one fp32 tl.dot
# partial to fixed-point per BLOCK_K-chunk, and the persistent and split-K
# kernels must share it so their integer sums bit-match (the same token routes
# through either kernel by batch size). Raised 32/16 to 128/64: fewer, larger
# tl.dot calls amortise the cvt.rni.sat + add, cutting quantise sweeps ~4x. Safe
# for int32/frac16 - float_to_fixed saturates only above |partial| > 32768, and
# a BLOCK_K-wide partial is O(1..100). fp32 stays 64: its tiles are 2x the bytes
# and 128 would not fit shared memory.
_BLOCK_K_BY_DTYPE = {
    torch.float16: 128,
    torch.bfloat16: 128,
    torch.float32: 64,
}
_SUPPORTED_DTYPES = tuple(_BLOCK_K_BY_DTYPE.keys())

_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# Configs are picked by shape, not autotuned, so a shape always gets the same
# mma sequence. Tuple is (BLOCK_M, BLOCK_N, GROUP_SIZE_M, num_warps, num_stages).
# Only BLOCK_K affects numerics, so BM/BN/num_stages are tuned freely; BM, BN
# stay >= 16 to keep tl.dot on tensor cores. BN=64, num_stages=2 keep the
# BLOCK_K=128 tiles inside the ~100 KB shared-memory budget on Ada/L4 (bf16:
# 2 * (128*128 + 128*64) * 2 B = 96 KB).
_CFG_DEFAULT = (128, 64, 8, 8, 2)    # activation matmuls, M and N both large
_CFG_MID_M = (32, 128, 8, 4, 2)      # decode 17..64; BM=128 would mask 50-75%
                                     # of rows and still pay their FMAs.
_CFG_SKINNY_M = (16, 128, 8, 4, 2)   # tiny M, big N, e.g. lm_head decode

# Split-K config. Smaller (BM, BN) and fewer warps than _CFG_DEFAULT to cut
# register pressure - on Blackwell the old (128, 64, 8 warps) hit 172 reg/thread
# and 1 block/SM (16.6% occupancy). Atomic-add latency hiding wants higher
# occupancy; 4 stages let the compiler pipeline the cp.async loads.
_CFG_SPLITK = (64, 64, 8, 4, 4)

_BLOCKS_PER_SM = 2

# Cap on split-K; beyond this, atomic contention and zero-init stop paying off.
_MAX_SPLIT_K = 16

# Persistent split-K scratch, reused across calls. c_int <= target_programs *
# 128*128 ints; the epilogue self-zeroes it.
_TILE_M_MAX = 128
_TILE_N_MAX = 128
_splitk_scratch: dict[
    tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def _get_splitk_scratch(
    device: torch.device, int_dtype: torch.dtype, target_programs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent (c_int_flat, locks_flat) scratch for the split-K path. Sized
    for the worst case, zeroed once; the epilogue rezeroes its own footprint.
    Allocated in warmup before graph capture and fixed in size, so it never
    moves."""
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


# do_not_specialize M: it changes per request and a recompile per M class would
# spike TTFT; M is only a bound/mask. M_bucket is just an autotune cache key,
# never read in the body, so don't specialize on it either.
@triton.jit(
    do_not_specialize=["M", "M_bucket"],
    do_not_specialize_on_alignment=["M", "M_bucket"],
)
def _gemm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    M_bucket,  # autotune key only; ignored in body
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
    NUM_SMS: tl.constexpr,  # launched grid size = persistent stride
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

        # Build pointers once, then advance by BLOCK_K per K iter, so the inner
        # loop is an add-with-immediate instead of re-multiplying offsets.
        offs_k0 = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k0[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k0[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)
        for ki in range(k_tiles):
            k_mask = offs_k_for_mask < K - ki * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
            fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

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
# runtime arg (not constexpr): split_k depends on M, so it's unenumerable.
@triton.jit(
    do_not_specialize=["M", "SPLIT_K"],
    do_not_specialize_on_alignment=["M", "SPLIT_K"],
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

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        offs_m_raw = start_m + tl.arange(0, BLOCK_M)
        offs_n_raw = start_n + tl.arange(0, BLOCK_N)

        offs_am = tl.where(offs_m_raw < M, offs_m_raw, 0)
        offs_bn = tl.where(offs_n_raw < N, offs_n_raw, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        # Interleaved K stride: pid_k=0 takes tiles 0, SPLIT_K, 2*SPLIT_K, ...;
        # pid_k=1 takes 1, SPLIT_K+1, ... so all SPLIT_K programs touch adjacent
        # A/B K-tiles each iter, maximising L2 reuse. Integer atomic_add is
        # commutative, so this stays bit-identical to contiguous chunks. Build
        # pointers once, advance by BLOCK_K*SPLIT_K per iter.
        offs_k0 = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k0[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k0[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        adv_ak = BLOCK_K * SPLIT_K * stride_ak
        adv_bk = BLOCK_K * SPLIT_K * stride_bk

        fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)
        for ki in range(pid_k, num_k_tiles, SPLIT_K):
            k_mask = offs_k_for_mask < K - ki * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
            fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)

            a_ptrs += adv_ak
            b_ptrs += adv_bk

        # Integer atomic-add: commutative, so determinism is race-free.
        c_int_ptrs = (
            c_int_ptr + offs_m_raw[:, None] * stride_cim + offs_n_raw[None, :] * stride_cin
        )
        c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
        tl.atomic_add(c_int_ptrs, fxp_acc, mask=c_mask, sem="relaxed")

        # Fused epilogue: the last SPLIT_K program to arrive owns it. The acq_rel
        # increment makes its load of c_int see every split's atomic-add, so the
        # output is order-independent.
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

            # Zero this tile's scratch for the next launch; all splits landed
            # and nothing else touches it, so race-free.
            tl.store(
                c_int_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE), mask=c_mask
            )
            tl.store(lock_ptr + tile_id_mn, 0)


def _cap() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# Per-dtype arch check: a dtype enters the cache after its first OK call, so
# later calls are a single set-membership test.
_ARCH_CHECKED_DTYPES: set[torch.dtype] = set()


def _check_arch(dtype: torch.dtype) -> None:
    if dtype in _ARCH_CHECKED_DTYPES:
        return
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
    _ARCH_CHECKED_DTYPES.add(dtype)


# Cache SM count; gemm_fxp_run runs for every linear layer.
_SM_COUNT_CACHE: dict[int, int] = {}


def _device_sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SM_COUNT_CACHE.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT_CACHE[idx] = n
    return n


_MIN_SPLIT_K = 4


def _pick_split_k(
    num_tiles_mn: int,
    num_k_tiles: int,
    num_sms: int,
    int_bits: int,
) -> int:
    # int16 atomics aren't broadly supported; skip split-K for them.
    if int_bits == 16:
        return 1
    # Split-K helps only when (M, N) tiling doesn't saturate the device. Once
    # num_tiles_mn reaches num_sms, more splits just add atomic contention.
    # num_tiles_mn uses _CFG_SPLITK, so this gate matches the actual tiling.
    if num_tiles_mn >= num_sms or num_k_tiles <= 1:
        return 1
    split = max(1, num_sms // max(1, num_tiles_mn))
    split = min(_MAX_SPLIT_K, split, num_k_tiles)
    # 2-3 way splits don't recoup the per-tile epilogue cost (atomic-add, lock
    # increment, last-writer read-back + dequant + store, scratch zero - approx
    # 10us/tile). The persistent path's autotune-bypassed dispatch (~75us floor
    # at small M) wins. Observed: M=128 qkv/o/down sat at 95-99us via split=2-3,
    # drop to ~75us on the persistent path.
    if split < _MIN_SPLIT_K:
        return 1
    return split


# Coarse pow2-ish buckets bound the autotune cache; without them the tuner would
# re-evaluate per request-size M, spiking TTFT on misses. Spans decode (1..256)
# and prefill (up to a few k tokens).
_M_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _bucket_m(M: int) -> int:
    for b in _M_BUCKETS:
        if M <= b:
            return b
    return _M_BUCKETS[-1]


# Candidate Configs (BM, BN, GROUP_SIZE_M, num_warps, num_stages) span every
# server arch from L4 (sm_89, ~100 KB SMEM, 30 SMs) up to A100/H100/Blackwell
# (up to ~228 KB SMEM, 100+ SMs). Configs that don't fit a GPU's SMEM budget
# fail compilation and triton.autotune silently drops them, so it's safe to list
# large-tile entries only Hopper/Blackwell can run.
#
# Determinism is not in the search space: BLOCK_K is pinned per dtype by the
# caller and never enters a Config, so every Config gives the same fxp sum.
#
# SMEM per stage for bf16/fp16 (BK=128, 2 B/elem): 2 * (BM*BK + BK*BN) * 2 B.
# Reference points (fits L4 if total <= ~96 KB):
#   (BM= 16, BN= 64): per_stage=20 KB; ns=2 -> 40 KB,  ns=4 ->  80 KB
#   (BM= 16, BN=128): per_stage=36 KB; ns=2 -> 72 KB,  ns=3 -> 108 KB (A100+)
#   (BM= 64, BN=128): per_stage=48 KB; ns=2 -> 96 KB,  ns=3 -> 144 KB (A100+)
#   (BM=128, BN=128): per_stage=64 KB; ns=2 ->128 KB (A100+), ns=3 ->192 KB (H100)
#   (BM=256, BN=128): per_stage=96 KB; ns=2 ->192 KB (H100+)
_AUTOTUNE_CONFIGS = [
    # ---- Skinny-M (decode, lm_head): tiny BM ----
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=2, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # Hopper/Blackwell: wider N, fewer programs
        {"BLOCK_M": 16, "BLOCK_N": 256, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),

    # ---- Mid-M (small-batch decode, post-attn projections) ----
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # A100+: ns=3 needs >= 144 KB SMEM
        {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=3,
    ),

    # ---- Large-M (prefill, batched decode): throughput configs ----
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # A100+
        {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=3,
    ),
    triton.Config(  # A100+: 128x128 doesn't fit L4 at ns=2
        {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # alt program order for wider-N L2 behaviour
        {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # Hopper/Blackwell: deeper pipeline at 128x128
        {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=3,
    ),

    # ---- XL-M (large prefill on big-SMEM GPUs): H100/Blackwell only ----
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
]


# Autotuner injects BLOCK_M/BLOCK_N/GROUP_SIZE_M + num_warps/num_stages from the
# Config it picks for (M_bucket, N, K). The Config space is a superset of the
# hand-tuned configs, so autotune can at worst tie them. BLOCK_K is not in the
# space (still passed as constexpr), so all paths bit-match.


def _prune_low_occupancy_configs(configs, nargs, **_kwargs):
    """Drop num_warps<8 candidates at M_bucket >= 2048.

    At large M the high-warp/high-stage configs win, but the gap to a worse pick
    is in the ~1-2us autotuner noise band, so the coin flip lands wrong ~half the
    time (costs ~50us at M=4096). Pruning rather than pinning keeps the autotuner
    free to pick the right large-tile shape per (N, K). Small M_bucket
    legitimately wants fewer warps, so this only triggers for prefill shapes."""
    m_bucket = nargs.get("M_bucket", 0)
    if m_bucket >= 2048:
        filtered = [c for c in configs if c.num_warps >= 8]
        if filtered:
            return filtered
    return configs


_gemm_kernel_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M_bucket", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_low_occupancy_configs},
)(_gemm_kernel)


# After the first call for a (M_bucket, N, K, dtype) key, cache the winning
# Config and re-dispatch _gemm_kernel directly. The autotune wrapper's per-call
# dict lookup + config scan + pre_hook is ~30-80us, which dominates at M<=1024
# and is pure overhead once warmed (same Config wins every time). Safe because
# Triton's per-signature compiled-kernel cache absorbs the re-dispatch.
_PICKED_CONFIG: dict[tuple[int, int, int, torch.dtype], triton.Config] = {}


def dump_picked_configs() -> list[tuple[tuple[int, int, int, torch.dtype], triton.Config]]:
    """Every (key, picked Config) pair the autotuner has resolved, sorted by
    key. For bench/diagnostic scripts: after a warmup pass over every shape,
    this is the record of which Config won per (M_bucket, N, K, dtype)."""
    return sorted(
        _PICKED_CONFIG.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3])),
    )


def _extract_picked_config(m_bucket: int, N: int, K: int) -> triton.Config | None:
    """Find the Config triton.autotune just picked for (M_bucket, N, K).

    Triton's cache key is (M_bucket, N, K, ...str(dtype)). The dtype suffix
    length varies by version, so match the int prefix and take the first hit -
    dispatch is monomorphic in dtype per key, so the first match is right.
    Returns None if Triton restructures the cache attr (caller falls back to the
    autotune wrapper)."""
    cache = getattr(_gemm_kernel_autotuned, "cache", None)
    if not cache:
        return None
    for k, cfg in cache.items():
        if (isinstance(k, tuple) and len(k) >= 3
                and k[0] == m_bucket and k[1] == N and k[2] == K):
            return cfg
    return None


def gemm_fxp_run(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    int_bits: int,
    fxp_frac_bits: int,
) -> torch.Tensor:
    # Hot path (once per linear per forward), so skip validation: callers feed
    # known-good tensors and Triton raises on real shape/dtype problems.
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

    num_sms = _device_sm_count(a.device)
    target_programs = num_sms * _BLOCKS_PER_SM
    num_k_tiles = triton.cdiv(K, block_k)

    # Decide split-K against the split-K kernel's own tiling (_CFG_SPLITK) - the
    # BM/BN it actually runs with. The persistent path's autotune-chosen tiling
    # doesn't enter the decision.
    sk_block_m, sk_block_n, sk_group_size_m, sk_num_warps, sk_num_stages = _CFG_SPLITK
    sk_num_pid_m = triton.cdiv(M, sk_block_m)
    sk_num_tiles_mn = sk_num_pid_m * triton.cdiv(N, sk_block_n)
    split_k = _pick_split_k(sk_num_tiles_mn, num_k_tiles, num_sms, int_bits)

    if split_k == 1:
        # Fast path: NUM_SMS is constexpr per GPU so it compiles once. Once the
        # autotuner picks the Config for (M_bucket, N, K, dtype), cache it and
        # re-dispatch _gemm_kernel directly, skipping the wrapper's ~30-80us
        # per-call overhead.
        m_bucket = _bucket_m(M)
        bias_tensor = bias if bias is not None else a
        has_bias = bias is not None

        picked_key = (m_bucket, N, K, a.dtype)
        picked = _PICKED_CONFIG.get(picked_key)

        if picked is None:
            # Cold path: go through the autotuner to pick + cache the Config.
            # Use the full target_programs grid since the tile count under its
            # BLOCK_M/BLOCK_N isn't known yet.
            _gemm_kernel_autotuned[(target_programs,)](
                a, b, bias_tensor, c,
                M, N, K, m_bucket,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                SCALE=scale, INV_SCALE=inv_scale,
                QMIN=qmin, QMAX=qmax,
                INT_DTYPE=tl_int_dtype, IO_DTYPE=io_dtype,
                HAS_BIAS=has_bias, ALLOW_TF32=allow_tf32,
                BLOCK_K=block_k, NUM_SMS=target_programs,
            )
            picked = _extract_picked_config(m_bucket, N, K)
            if picked is not None:
                _PICKED_CONFIG[picked_key] = picked
            return c

        # Hot path: raw JIT kernel with the cached Config. Right-size the grid to
        # the tile count (idle programs still cost launch latency). NUM_SMS stays
        # constexpr=target_programs so the stride is fixed and we don't recompile.
        block_m = picked.kwargs["BLOCK_M"]
        block_n = picked.kwargs["BLOCK_N"]
        num_tiles_mn = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        grid_size = min(target_programs, num_tiles_mn)
        _gemm_kernel[(grid_size,)](
            a, b, bias_tensor, c,
            M, N, K, m_bucket,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            SCALE=scale, INV_SCALE=inv_scale,
            QMIN=qmin, QMAX=qmax,
            INT_DTYPE=tl_int_dtype, IO_DTYPE=io_dtype,
            HAS_BIAS=has_bias, ALLOW_TF32=allow_tf32,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE_M=picked.kwargs["GROUP_SIZE_M"],
            NUM_SMS=target_programs,
            num_warps=picked.num_warps,
            num_stages=picked.num_stages,
        )
        return c

    # Persistent scratch (epilogue keeps it clean); the (M, N) view fits since
    # M * N <= num_tiles_mn * 128*128.
    c_int_flat, locks_flat = _get_splitk_scratch(
        a.device, torch_int_dtype, target_programs
    )
    c_int = c_int_flat[: M * N].view(M, N)
    locks = locks_flat[:sk_num_tiles_mn]
    # Right-size the grid to actual tiles (idle programs still cost launch
    # latency). split-K only triggers when num_tiles_mn < num_sms, so
    # num_tiles_mn * split_k is roughly num_sms. NUM_SMS stays constexpr=target
    # so the stride is fixed (no recompile on grid changes).
    grid_size = min(target_programs, sk_num_tiles_mn * split_k)
    grid = (grid_size,)
    _gemm_splitk_kernel[grid](
        a,
        b,
        # Triton needs a real tensor; only read under HAS_BIAS.
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
        num_k_tiles,
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        IO_DTYPE=io_dtype,
        HAS_BIAS=bias is not None,
        ALLOW_TF32=allow_tf32,
        BLOCK_M=sk_block_m,
        BLOCK_N=sk_block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=sk_group_size_m,
        SPLIT_K=split_k,
        # NUM_SMS is the persistent loop stride (constexpr); keep it at
        # target_programs even when grid_size shrinks, so no per-shape recompile.
        NUM_SMS=target_programs,
        num_warps=sk_num_warps,
        num_stages=sk_num_stages,
    )
    return c
