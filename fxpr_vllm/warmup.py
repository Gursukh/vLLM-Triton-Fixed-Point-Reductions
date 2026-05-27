"""Compile every Triton kernel at startup, before the server takes traffic.

Otherwise they JIT on first use and spike TTFT.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("fxpr_vllm")

_attn_warmed = False
_gemm_warmed: set[tuple] = set()
_rms_norm_warmed: set[tuple] = set()


def warmup_attention(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    window_size: int,
    logit_softcap: float,
    int_bits: int,
    frac_bits: int,
) -> None:
    """Compile all attention configs once, on the first eager forward."""
    global _attn_warmed
    if _attn_warmed:
        return
    from .config import get_runtime_config
    if get_runtime_config().disable_attention_warmup:
        _attn_warmed = True
        return
    # can't launch kernels during capture; try again on a later eager call.
    if torch.cuda.is_current_stream_capturing():
        return
    _attn_warmed = True
    try:
        _do_warmup_attention(
            num_heads, num_kv_heads, head_size, dtype, device,
            window_size, logit_softcap, int_bits, frac_bits,
        )
    except Exception as e:  # noqa: BLE001 - warmup must never break serving
        logger.warning("fxpr attention warmup skipped: %s", e)


def _do_warmup_attention(
    num_heads, num_kv_heads, head_size, dtype, device,
    window_size, logit_softcap, int_bits, frac_bits,
) -> None:
    from ._triton.attention import _SPLIT_OCCUPANCY, _arch_info, _next_pow2

    page_size = 16
    softmax_scale = float(head_size) ** -0.5

    def _run(max_q: int, splits: int) -> None:
        # small context, just enough to run the kernel. the binary doesn't
        # depend on sequence length.
        ctx = max(128, 2 * max_q)
        num_blocks = (ctx + page_size - 1) // page_size
        kv_cache = torch.zeros(
            num_blocks, 2, page_size, num_kv_heads, head_size,
            device=device, dtype=dtype,
        )
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(1, num_blocks)
        qsl = torch.tensor([0, max_q], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([ctx], dtype=torch.int32, device=device)
        q = torch.zeros(max_q, num_heads, head_size, device=device, dtype=dtype)
        o = torch.zeros_like(q)
        torch.ops.fxpr.unified_attention_fxp(
            q, kv_cache, o, qsl, seq_lens, block_table, max_q,
            None, True, softmax_scale,
            int(int_bits), int(frac_bits),
            float(logit_softcap), int(window_size), int(splits),
        )

    # the runtime split count is a power of two up to num_sms * 2, which can be
    # well above 16 on big GPUs. compile them all so decode never JITs one.
    _, num_sms = _arch_info(device)
    max_split = _next_pow2(num_sms * _SPLIT_OCCUPANCY)
    split_counts = tuple(1 << i for i in range(max_split.bit_length()))

    # BLOCK_M is 16 for decode (max_q 1) and 32 for prefill. cover both for
    # every split count. small max_q keeps the warmup grid small.
    n_configs = 0
    for max_q in (1, 64):
        for splits in split_counts:
            _run(max_q, splits)
            n_configs += 1
    # run the fused prefill path (split 1) at a few real lengths too.
    for max_q in (256, 512, 1024, 2048):
        _run(max_q, 1)
        n_configs += 1

    torch.cuda.synchronize(device)
    logger.info(
        "fxpr attention kernels warmed up (%d configs, max_split=%d)",
        n_configs, max_split,
    )


def warmup_rms_norm(
    hidden_size: int, dtype: torch.dtype, device: torch.device
) -> None:
    """Compile rms_norm and its residual variant for one (hidden, dtype).
    One binary covers every batch size."""
    key = (hidden_size, dtype, device.index if device.index is not None else 0)
    if key in _rms_norm_warmed:
        return
    from .config import get_runtime_config
    if get_runtime_config().disable_rms_norm_warmup:
        _rms_norm_warmed.add(key)
        return
    if torch.cuda.is_current_stream_capturing():
        return
    _rms_norm_warmed.add(key)
    try:
        x = torch.zeros(1, hidden_size, device=device, dtype=dtype)
        w = torch.zeros(hidden_size, device=device, dtype=dtype)
        r = torch.zeros_like(x)
        torch.ops.fxpr.rms_norm_fxp(x, w, 1e-6)
        torch.ops.fxpr.rms_norm_fxp_residual(x, r, w, 1e-6)
        torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001
        logger.warning("fxpr rms_norm warmup skipped: %s", e)


def warmup_gemm(
    weight_native: torch.Tensor, int_bits: int, frac_bits: int
) -> None:
    """Compile gemm_fxp for one (K, N) weight at load time. Deduped by shape.

    Two sweeps cover every binary the runtime can land on: one per autotune
    bucket (the persistent path) and one per reachable SPLIT_K. SPLIT_K is a
    constexpr so each value is its own binary; bias is too, so we run each
    probe with and without it. Bucket probes use M=b (top of the bucket) - if
    any M in the bucket lands in split_k==1, b does too; otherwise the whole
    bucket is in split-K territory and the second sweep covers it."""
    K, N = weight_native.shape
    dtype = weight_native.dtype
    device = weight_native.device
    key = (K, N, dtype, int(int_bits), int(frac_bits))
    if key in _gemm_warmed:
        return
    _gemm_warmed.add(key)
    from .config import get_runtime_config
    if get_runtime_config().disable_gemm_warmup:
        return

    from ._triton.gemm import _M_BUCKETS, _device_sm_count, _reachable_split_ks

    try:
        num_sms = _device_sm_count(device)
        bias = torch.zeros(N, device=device, dtype=dtype)

        # persistent path: top of every bucket, with and without bias. first
        # call autotunes, second compiles the path serving actually uses.
        for b in _M_BUCKETS:
            a = torch.zeros(b, K, device=device, dtype=dtype)
            for bias_arg in (None, bias):
                torch.ops.fxpr.gemm_fxp(
                    a, weight_native, bias_arg, int(int_bits), int(frac_bits)
                )
                torch.ops.fxpr.gemm_fxp(
                    a, weight_native, bias_arg, int(int_bits), int(frac_bits)
                )

        # split-K path: one M per reachable SPLIT_K. skip 1, the persistent
        # path above already hit it.
        reachable = _reachable_split_ks(N, K, dtype, num_sms, int(int_bits))
        for split, m in reachable.items():
            if split == 1:
                continue
            a = torch.zeros(m, K, device=device, dtype=dtype)
            for bias_arg in (None, bias):
                torch.ops.fxpr.gemm_fxp(
                    a, weight_native, bias_arg, int(int_bits), int(frac_bits)
                )

        torch.cuda.synchronize(device)
        logger.info(
            "fxpr gemm warmed K=%d N=%d dtype=%s (buckets=%d, splits=%d)",
            K, N, dtype, len(_M_BUCKETS), len(reachable),
        )
    except Exception as e:  # noqa: BLE001 - warmup must never break serving
        logger.warning("fxpr gemm warmup skipped for K=%d N=%d: %s", K, N, e)
