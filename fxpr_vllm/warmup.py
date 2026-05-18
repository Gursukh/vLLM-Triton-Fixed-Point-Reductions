"""One-time kernel warmup.

The fxpr Triton kernels JIT-compile on first use, which would spike the TTFT of
the first serving requests. This module compiles every kernel config at
start-up, before the server takes traffic.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("fxpr_vllm")

_attn_warmed = False
_gemm_warmed: set[tuple] = set()


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
    """Compile every attention kernel config. Runs once, on the first eager
    forward (vLLM's start-up profiling run), never under CUDA-graph capture."""
    global _attn_warmed
    if _attn_warmed:
        return
    # Compiling launches kernels, which is illegal mid-capture. Defer to a
    # later eager call and don't mark done.
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
    ctx = 128
    num_blocks = (ctx + page_size - 1) // page_size
    kv_cache = torch.zeros(
        num_blocks, 2, page_size, num_kv_heads, head_size,
        device=device, dtype=dtype,
    )
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(1, num_blocks)
    softmax_scale = float(head_size) ** -0.5

    # (max_query_len, num_kv_splits) pairs: both BLOCK_M sizes (decode max_q=1,
    # prefill max_q=64) crossed with every split count, so every BLOCK_S
    # constexpr is pre-compiled.
    split_counts = (1, 2, 4, 8, 16)
    configs = [(mq, s) for mq in (1, 64) for s in split_counts]
    for max_q, splits in configs:
        q = torch.zeros(max_q, num_heads, head_size, device=device, dtype=dtype)
        o = torch.zeros_like(q)
        qsl = torch.tensor([0, max_q], dtype=torch.int32, device=device)
        seq_lens = torch.tensor(
            [max(ctx, max_q)], dtype=torch.int32, device=device
        )
        torch.ops.fxpr.unified_attention_fxp(
            q, kv_cache, o, qsl, seq_lens, block_table, max_q,
            None, True, softmax_scale,
            int(int_bits), int(frac_bits),
            float(logit_softcap), int(window_size), int(splits),
        )
    torch.cuda.synchronize(device)
    logger.info("fxpr attention kernels warmed up (%d configs)", len(configs))


def warmup_gemm(
    weight_native: torch.Tensor, int_bits: int, frac_bits: int
) -> None:
    """Compile the gemm_fxp kernels for one (K, N) weight. Deduplicated, so
    layers that share a shape only compile once. Runs at weight-load time."""
    K, N = weight_native.shape
    key = (K, N, weight_native.dtype, int(int_bits), int(frac_bits))
    if key in _gemm_warmed:
        return
    _gemm_warmed.add(key)
    try:
        # M=1 exercises the skinny config + split-K kernel; M=256 the default
        # config + persistent kernel.
        for m in (1, 256):
            a = torch.zeros(
                m, K, device=weight_native.device, dtype=weight_native.dtype
            )
            torch.ops.fxpr.gemm_fxp(
                a, weight_native, None, int(int_bits), int(frac_bits)
            )
        torch.cuda.synchronize(weight_native.device)
    except Exception as e:  # noqa: BLE001 - warmup must never break serving
        logger.warning("fxpr gemm warmup skipped for K=%d N=%d: %s", K, N, e)
