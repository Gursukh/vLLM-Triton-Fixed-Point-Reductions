import pytest
import torch

import fxpr_vllm._cuda  # noqa: F401  (registers torch.ops.fxpr.*)


_INT_BITS_BY_DTYPE = {
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}

_FLOAT_BITS_BY_DTYPE = {
    torch.float16: 16,
    torch.float32: 32,
    torch.float64: 64,
}


def float_to_fixed(x: torch.Tensor, frac_bits: int, out: torch.dtype) -> torch.Tensor:
    int_bits = _INT_BITS_BY_DTYPE[out]
    return torch.ops.fxpr.float_to_fixed(x.contiguous(), int(frac_bits), int_bits)


def fixed_to_float(x: torch.Tensor, frac_bits: int, out: torch.dtype) -> torch.Tensor:
    float_bits = _FLOAT_BITS_BY_DTYPE[out]
    return torch.ops.fxpr.fixed_to_float(x.contiguous(), int(frac_bits), float_bits)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gemm_fxp_test(
    a: torch.Tensor, b: torch.Tensor, frac_bits: int = 16, fxp_int_bits: int = 32
) -> torch.Tensor:
    """Test adapter for the registered :func:`fxpr::gemm_fxp` torch op."""
    return torch.ops.fxpr.gemm_fxp(
        a.contiguous(), b.contiguous(), int(frac_bits), int(fxp_int_bits)
    )


RCP_LN2 = 1.4426950408889634


def prefill_fxp_test(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    *,
    alibi_slopes: torch.Tensor | None = None,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    frac_bits: int = 14,
) -> None:
    """Test adapter that runs prefill via the unified paged-KV attention kernel.

    Builds a synthetic page_size=1 KV cache from packed (seq, head, dim) k/v.
    Each token occupies its own physical block, so the page-table lookup
    becomes the absolute token index and the kernel's paged path collapses
    to the non-paged behaviour the prefill tests originally exercised.
    """
    # unified_attention_fxp expects pre-scaled (by RCP_LN2) alibi slopes.
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes * RCP_LN2

    total_tokens, num_kv_heads, head_dim = k.shape

    # page_size=1 KV cache: shape (total_tokens, 2, 1, num_kv_heads, head_dim).
    kv_cache = torch.empty(
        total_tokens,
        2,
        1,
        num_kv_heads,
        head_dim,
        device=k.device,
        dtype=k.dtype,
    )
    kv_cache[:, 0, 0] = k
    kv_cache[:, 1, 0] = v

    # block_table: each request maps logical block i -> absolute token index.
    num_requests = int(b_seq_len.shape[0])
    block_table = torch.zeros(
        num_requests, max_input_len, device=q.device, dtype=torch.int32
    )
    seq_lens_list = b_seq_len.tolist()
    starts_list = b_start_loc.tolist()
    for r, (start, length) in enumerate(zip(starts_list, seq_lens_list)):
        block_table[r, :length] = torch.arange(
            start, start + length, device=q.device, dtype=torch.int32
        )

    # query_start_loc is cumulative with a trailing total: shape (n+1,).
    query_start_loc = torch.empty(num_requests + 1, device=q.device, dtype=torch.int32)
    query_start_loc[0] = 0
    query_start_loc[1:] = torch.cumsum(b_seq_len, dim=0)

    torch.ops.fxpr.unified_attention_fxp(
        q,
        kv_cache,
        o,
        query_start_loc,
        b_seq_len.to(torch.int32),
        block_table,
        int(max_input_len),
        alibi_slopes,
        bool(is_causal),
        None if softmax_scale is None else float(softmax_scale),
        int(frac_bits),
        32,  # fxp_int_bits
        0.0,  # logit_softcap
        0,    # window_size
    )
