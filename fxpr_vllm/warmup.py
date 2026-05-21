"""One-time kernel warmup.

Triton kernels JIT-compile on first use, spiking TTFT. Compile every config at
start-up before the server takes traffic.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("fxpr_vllm")

_attn_warmed = False
_gemm_warmed: set[tuple] = set()
_rms_norm_warmed: set[tuple] = set()
_log_softmax_warmed: set[tuple] = set()


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
    """Compile every attention config. Runs once on the first eager forward,
    never under CUDA-graph capture."""
    global _attn_warmed
    if _attn_warmed:
        return
    # can't launch kernels mid-capture; defer to a later eager call.
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
    page_size = 16
    softmax_scale = float(head_size) ** -0.5

    # cover both BLOCK_M buckets (1=decode, >1=prefill) and realistic prefill
    # lengths; otherwise the first large request pays a compile spike.
    max_qs = (1, 64, 256, 512, 1024, 2048)
    split_counts = (1, 2, 4, 8, 16)
    n_configs = 0
    for max_q in max_qs:
        # give it a populated context (prefill reads past the query rows) to
        # cover every K/V tile load path.
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
        for splits in split_counts:
            torch.ops.fxpr.unified_attention_fxp(
                q, kv_cache, o, qsl, seq_lens, block_table, max_q,
                None, True, softmax_scale,
                int(int_bits), int(frac_bits),
                float(logit_softcap), int(window_size), int(splits),
            )
            n_configs += 1
    torch.cuda.synchronize(device)
    logger.info("fxpr attention kernels warmed up (%d configs)", n_configs)


def warmup_rms_norm(
    hidden_size: int, dtype: torch.dtype, device: torch.device
) -> None:
    """Compile rms_norm (and its residual variant) for one (hidden, dtype).
    One binary covers every batch size since hidden is a runtime arg."""
    key = (hidden_size, dtype, device.index if device.index is not None else 0)
    if key in _rms_norm_warmed:
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


def warmup_log_softmax(
    vocab_size: int, device: torch.device, int_bits: int, frac_bits: int,
) -> None:
    """Compile log_softmax. The caller always upcasts to fp32, so one binary
    covers every input dtype."""
    key = (vocab_size, int(int_bits), int(frac_bits))
    if key in _log_softmax_warmed:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    _log_softmax_warmed.add(key)
    try:
        x = torch.zeros(1, vocab_size, device=device, dtype=torch.float32)
        torch.ops.fxpr.log_softmax_fxp(x, int(int_bits), int(frac_bits))
        torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001
        logger.warning("fxpr log_softmax warmup skipped: %s", e)


def warmup_gemm(
    weight_native: torch.Tensor, int_bits: int, frac_bits: int
) -> None:
    """Compile gemm_fxp for one (K, N) weight, deduped by shape. Runs at
    weight-load time, sweeping one M per autotune bucket - autotune launches
    kernels, which is illegal once CUDA-graph capture starts."""
    K, N = weight_native.shape
    key = (K, N, weight_native.dtype, int(int_bits), int(frac_bits))
    if key in _gemm_warmed:
        return
    _gemm_warmed.add(key)
    from ._triton.gemm import _M_BUCKETS
    try:
        for m in _M_BUCKETS:
            a = torch.zeros(
                m, K, device=weight_native.device, dtype=weight_native.dtype
            )
            torch.ops.fxpr.gemm_fxp(
                a, weight_native, None, int(int_bits), int(frac_bits)
            )
        torch.cuda.synchronize(weight_native.device)
    except Exception as e:  # noqa: BLE001 - warmup must never break serving
        logger.warning("fxpr gemm warmup skipped for K=%d N=%d: %s", K, N, e)
