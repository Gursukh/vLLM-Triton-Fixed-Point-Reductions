import pytest
import torch

from fxpr_vllm._triton.gemm import (
    _CFG_SPLITK,
    _M_BUCKETS,
    _PICKED_CONFIG,
    _gemm_kernel_autotuned,
    _pick_split_k,
    _reachable_split_ks,
)
from fxpr_vllm.warmup import _gemm_warmed, warmup_gemm

from .fixed_point_helpers import requires_cuda, skip_if_dtype_unsupported


def _splitk_from_M(M: int, N: int, K: int, dtype: torch.dtype, num_sms: int, int_bits: int) -> int:
    """Same split-K math gemm_fxp_run runs, but for a single M."""
    from fxpr_vllm._triton.gemm import _BLOCK_K_BY_DTYPE
    sk_block_m, sk_block_n = _CFG_SPLITK[0], _CFG_SPLITK[1]
    block_k = _BLOCK_K_BY_DTYPE[dtype]
    num_k_tiles = (K + block_k - 1) // block_k
    num_tiles_mn = ((M + sk_block_m - 1) // sk_block_m) * ((N + sk_block_n - 1) // sk_block_n)
    return _pick_split_k(num_tiles_mn, num_k_tiles, num_sms, int_bits)


def test_reachable_split_ks_round_trip():
    """Sweep every M in the bucket range and check the split_k it picks is one
    _reachable_split_ks reported. A miss here means warmup would skip a binary."""
    cases = [
        (64, 4096, torch.float16, 108, 32),    # A100-ish, tall-skinny
        (4096, 4096, torch.float16, 108, 32),  # square, prefill
        (128, 8192, torch.bfloat16, 132, 32),  # H100-ish
        (64, 2048, torch.float16, 132, 16),    # int16, forces split_k=1
    ]
    for N, K, dtype, num_sms, int_bits in cases:
        reachable = _reachable_split_ks(N, K, dtype, num_sms, int_bits)
        for M in range(1, _M_BUCKETS[-1] + 1):
            s = _splitk_from_M(M, N, K, dtype, num_sms, int_bits)
            assert s in reachable, (
                f"M={M} produced split_k={s} not in reachable set {sorted(reachable)} "
                f"for (N={N}, K={K}, dtype={dtype}, num_sms={num_sms}, int_bits={int_bits})"
            )


def test_reachable_split_ks_int16_is_only_one():
    """int_bits=16 has no atomic, so _pick_split_k always returns 1."""
    reachable = _reachable_split_ks(64, 4096, torch.float16, 108, 16)
    assert set(reachable.keys()) == {1}


@requires_cuda
def test_warmup_covers_full_M_sweep():
    """After warmup, hit every M in the bucket range with both bias polarities
    and check nothing new lands in the autotune cache or _PICKED_CONFIG. A new
    entry means the runtime hit a JIT path warmup missed."""
    dtype = torch.float16
    skip_if_dtype_unsupported(dtype)

    # small N widens the split-K range, so this shape exercises more binaries.
    K, N = 4096, 64
    # weight_native is the (K, N) transpose of an (out=N, in=K) Linear.
    w = torch.zeros(K, N, device="cuda", dtype=dtype)

    # drop any prior warmup state so we're testing this run, not a leftover.
    _gemm_warmed.discard((K, N, dtype, 32, 16))

    warmup_gemm(w, 32, 16)

    autotune_keys_after = set(_gemm_kernel_autotuned.cache.keys())
    picked_keys_after = set(_PICKED_CONFIG.keys())

    bias = torch.zeros(N, device="cuda", dtype=dtype)
    # walk every M, plus block boundaries and bucket edges to be safe.
    sk_block_m = _CFG_SPLITK[0]
    sweep = set()
    for M in range(1, _M_BUCKETS[-1] + 1):
        sweep.add(M)
        if M % sk_block_m == 0 or M % sk_block_m == 1:
            sweep.add(M)
    prev = 0
    for b in _M_BUCKETS:
        sweep.update({prev + 1, b - 1, b})
        prev = b
    sweep = sorted(s for s in sweep if 1 <= s <= _M_BUCKETS[-1])

    for M in sweep:
        a = torch.zeros(M, K, device="cuda", dtype=dtype)
        torch.ops.fxpr.gemm_fxp(a, w, None, 32, 16)
        torch.ops.fxpr.gemm_fxp(a, w, bias, 32, 16)

    new_autotune = set(_gemm_kernel_autotuned.cache.keys()) - autotune_keys_after
    new_picked = set(_PICKED_CONFIG.keys()) - picked_keys_after

    assert not new_autotune, f"warmup missed autotune keys: {new_autotune}"
    assert not new_picked, f"warmup missed _PICKED_CONFIG keys: {new_picked}"


@requires_cuda
def test_warmup_dedup():
    """Second warmup_gemm on the same shape should do nothing."""
    dtype = torch.float16
    skip_if_dtype_unsupported(dtype)

    K, N = 2048, 128
    w = torch.zeros(K, N, device="cuda", dtype=dtype)
    _gemm_warmed.discard((K, N, dtype, 32, 16))

    warmup_gemm(w, 32, 16)
    autotune_after_first = len(_gemm_kernel_autotuned.cache)
    picked_after_first = len(_PICKED_CONFIG)

    warmup_gemm(w, 32, 16)
    assert len(_gemm_kernel_autotuned.cache) == autotune_after_first
    assert len(_PICKED_CONFIG) == picked_after_first
