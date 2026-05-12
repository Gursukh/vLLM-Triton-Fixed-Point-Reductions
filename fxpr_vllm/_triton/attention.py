"""Unified prefill + decode attention over paged KV with split-K.

Three kernels: split_max finds per-split row max, split_dv accumulates the
denominator and weighted-V using that max, combine reduces across splits
and divides. qk dot, denom, and weighted-V are all integer reductions;
only the per-row max stays fp32. softmax_scale is pre-multiplied by
1/ln(2) so qk lives in log2 space and we use exp2.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


_RCP_LN2 = 1.4426950408889634
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _compute_chunk(key_start, key_end, num_splits, split_index):
    total = key_end - key_start
    s_lo = split_index.to(tl.int64) * total.to(tl.int64)
    s_hi = (split_index + 1).to(tl.int64) * total.to(tl.int64)
    chunk_start = key_start + (s_lo // num_splits).to(tl.int32)
    chunk_end = key_start + (s_hi // num_splits).to(tl.int32)
    return chunk_start, chunk_end


@triton.jit
def _post_process_qk(
    qk_fxp,
    softmax_scale_log2,
    softcap_log2,
    alibi_slope,
    kp,
    q_abs_pos,
    INV_SCALE: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    qk = fixed_to_float(qk_fxp, INV_SCALE)
    qk = qk * softmax_scale_log2
    if HAS_SOFTCAP:
        qk = softcap_log2 * libdevice.tanh(qk / softcap_log2)
    qk = qk + alibi_slope * (kp.to(tl.float32) - q_abs_pos.to(tl.float32))
    return qk


@triton.jit
def _attn_split_max_kernel(
    Q_ptr,
    K_ptr,
    partial_max_ptr,
    qsl_ptr,
    seq_lens_ptr,
    block_table_ptr,
    alibi_slopes_ptr,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    num_splits,
    window_size,
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_slot,
    stride_k_head,
    stride_pmax_token,
    stride_pmax_head,
    stride_pmax_split,
    stride_block_table_row,
    softmax_scale_log2,
    softcap_log2,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(axis=0)
    q_in_req = tl.program_id(axis=1)
    hs = tl.program_id(axis=2)
    head_index = hs // num_splits
    split_index = hs % num_splits
    kv_group = num_heads // num_kv_heads
    kv_head_index = head_index // kv_group

    q_start = tl.load(qsl_ptr + req)
    q_end = tl.load(qsl_ptr + req + 1)
    q_len = q_end - q_start
    if q_in_req >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + req)
    context_len = seq_len - q_len
    q_token_idx = q_start + q_in_req
    q_abs_pos = context_len + q_in_req

    if IS_CAUSAL:
        key_end = q_abs_pos + 1
    else:
        key_end = seq_len
    if HAS_WINDOW:
        key_start = tl.maximum(0, q_abs_pos - window_size + 1)
    else:
        key_start = 0

    chunk_start, chunk_end = _compute_chunk(key_start, key_end, num_splits, split_index)
    if chunk_start >= chunk_end:
        return

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes_ptr + head_index)
    else:
        alibi_slope = 0.0

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    q_row = Q_ptr + q_token_idx * stride_q_token + head_index * stride_q_head
    q_vec = tl.load(q_row + d_off, mask=d_mask, other=0.0).to(tl.float32)
    bt_row = block_table_ptr + req * stride_block_table_row

    row_max = -float("inf")
    for kp in range(chunk_start, chunk_end):
        logical_block = kp // page_size
        slot = kp % page_size
        physical_block = tl.load(bt_row + logical_block)
        k_row = (
            K_ptr
            + physical_block * stride_k_block
            + slot * stride_k_slot
            + kv_head_index * stride_k_head
        )
        k_vec = tl.load(k_row + d_off, mask=d_mask, other=0.0).to(tl.float32)
        prod = q_vec * k_vec
        qk_fxp = tl.sum(
            float_to_fixed(prod, SCALE, QMIN, QMAX, INT_DTYPE), axis=0
        )
        qk = _post_process_qk(
            qk_fxp,
            softmax_scale_log2,
            softcap_log2,
            alibi_slope,
            kp,
            q_abs_pos,
            INV_SCALE,
            HAS_SOFTCAP,
        )
        row_max = tl.maximum(row_max, qk)

    out_ptr = (
        partial_max_ptr
        + q_token_idx * stride_pmax_token
        + head_index * stride_pmax_head
        + split_index * stride_pmax_split
    )
    tl.store(out_ptr, row_max)


@triton.jit
def _attn_split_dv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    partial_max_ptr,
    partial_denom_ptr,
    partial_wv_ptr,
    qsl_ptr,
    seq_lens_ptr,
    block_table_ptr,
    alibi_slopes_ptr,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    num_splits,
    window_size,
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_slot,
    stride_k_head,
    stride_v_block,
    stride_v_slot,
    stride_v_head,
    stride_pmax_token,
    stride_pmax_head,
    stride_pmax_split,
    stride_pdenom_token,
    stride_pdenom_head,
    stride_pdenom_split,
    stride_pwv_token,
    stride_pwv_head,
    stride_pwv_split,
    stride_pwv_d,
    stride_block_table_row,
    softmax_scale_log2,
    softcap_log2,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    req = tl.program_id(axis=0)
    q_in_req = tl.program_id(axis=1)
    hs = tl.program_id(axis=2)
    head_index = hs // num_splits
    split_index = hs % num_splits
    kv_group = num_heads // num_kv_heads
    kv_head_index = head_index // kv_group

    q_start = tl.load(qsl_ptr + req)
    q_end = tl.load(qsl_ptr + req + 1)
    q_len = q_end - q_start
    if q_in_req >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + req)
    context_len = seq_len - q_len
    q_token_idx = q_start + q_in_req
    q_abs_pos = context_len + q_in_req

    if IS_CAUSAL:
        key_end = q_abs_pos + 1
    else:
        key_end = seq_len
    if HAS_WINDOW:
        key_start = tl.maximum(0, q_abs_pos - window_size + 1)
    else:
        key_start = 0

    chunk_start, chunk_end = _compute_chunk(key_start, key_end, num_splits, split_index)
    if chunk_start >= chunk_end:
        return

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes_ptr + head_index)
    else:
        alibi_slope = 0.0

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    q_row = Q_ptr + q_token_idx * stride_q_token + head_index * stride_q_head
    q_vec = tl.load(q_row + d_off, mask=d_mask, other=0.0).to(tl.float32)
    bt_row = block_table_ptr + req * stride_block_table_row

    # Fold the per-split row maxes from pass 1 into a single max for this token+head.
    pm_row = partial_max_ptr + q_token_idx * stride_pmax_token + head_index * stride_pmax_head
    s_off = tl.arange(0, BLOCK_S)
    s_mask = s_off < num_splits
    pm_vals = tl.load(pm_row + s_off * stride_pmax_split, mask=s_mask, other=-float("inf"))
    global_max = tl.max(pm_vals, axis=0)

    out_acc = tl.zeros((BLOCK_D,), dtype=INT_DTYPE)
    denom_partial = tl.zeros((), dtype=INT_DTYPE)

    for kp in range(chunk_start, chunk_end):
        logical_block = kp // page_size
        slot = kp % page_size
        physical_block = tl.load(bt_row + logical_block)
        k_row = (
            K_ptr
            + physical_block * stride_k_block
            + slot * stride_k_slot
            + kv_head_index * stride_k_head
        )
        v_row = (
            V_ptr
            + physical_block * stride_v_block
            + slot * stride_v_slot
            + kv_head_index * stride_v_head
        )
        k_vec = tl.load(k_row + d_off, mask=d_mask, other=0.0).to(tl.float32)
        v_vec = tl.load(v_row + d_off, mask=d_mask, other=0.0).to(tl.float32)

        prod = q_vec * k_vec
        qk_fxp = tl.sum(
            float_to_fixed(prod, SCALE, QMIN, QMAX, INT_DTYPE), axis=0
        )
        qk = _post_process_qk(
            qk_fxp,
            softmax_scale_log2,
            softcap_log2,
            alibi_slope,
            kp,
            q_abs_pos,
            INV_SCALE,
            HAS_SOFTCAP,
        )
        weight = libdevice.exp2(qk - global_max)
        weight_fxp = float_to_fixed(weight, SCALE, QMIN, QMAX, INT_DTYPE)
        denom_partial = denom_partial + weight_fxp

        wv_prod = weight * v_vec
        out_acc = out_acc + float_to_fixed(wv_prod, SCALE, QMIN, QMAX, INT_DTYPE)

    pdenom_ptr = (
        partial_denom_ptr
        + q_token_idx * stride_pdenom_token
        + head_index * stride_pdenom_head
        + split_index * stride_pdenom_split
    )
    tl.store(pdenom_ptr, denom_partial)

    pwv_row = (
        partial_wv_ptr
        + q_token_idx * stride_pwv_token
        + head_index * stride_pwv_head
        + split_index * stride_pwv_split
    )
    tl.store(pwv_row + d_off * stride_pwv_d, out_acc, mask=d_mask)


@triton.jit
def _attn_combine_kernel(
    partial_denom_ptr,
    partial_wv_ptr,
    O_ptr,
    qsl_ptr,
    num_heads,
    head_dim,
    num_splits,
    stride_pdenom_token,
    stride_pdenom_head,
    stride_pdenom_split,
    stride_pwv_token,
    stride_pwv_head,
    stride_pwv_split,
    stride_pwv_d,
    stride_o_token,
    stride_o_head,
    INV_SCALE: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    req = tl.program_id(axis=0)
    q_in_req = tl.program_id(axis=1)
    head_index = tl.program_id(axis=2)

    q_start = tl.load(qsl_ptr + req)
    q_end = tl.load(qsl_ptr + req + 1)
    q_len = q_end - q_start
    if q_in_req >= q_len:
        return
    q_token_idx = q_start + q_in_req

    s_off = tl.arange(0, BLOCK_S)
    s_mask = s_off < num_splits
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim

    pdenom_row = (
        partial_denom_ptr
        + q_token_idx * stride_pdenom_token
        + head_index * stride_pdenom_head
    )
    denom_parts = tl.load(pdenom_row + s_off * stride_pdenom_split, mask=s_mask, other=0)
    denom_sum = tl.sum(denom_parts, axis=0)
    denom = tl.maximum(fixed_to_float(denom_sum, INV_SCALE), 1.0e-6)

    pwv_row = (
        partial_wv_ptr
        + q_token_idx * stride_pwv_token
        + head_index * stride_pwv_head
    )
    wv_tile = tl.load(
        pwv_row + s_off[:, None] * stride_pwv_split + d_off[None, :] * stride_pwv_d,
        mask=s_mask[:, None] & d_mask[None, :],
        other=0,
    )
    wv_sum = tl.sum(wv_tile, axis=0)
    num = fixed_to_float(wv_sum, INV_SCALE)
    out = num / denom

    o_row = O_ptr + q_token_idx * stride_o_token + head_index * stride_o_head
    tl.store(o_row + d_off, out.to(IO_DTYPE), mask=d_mask)


def _pick_num_splits(
    max_query_len: int,
    requested: int,
    num_requests: int,
    num_heads: int,
    device: torch.device,
) -> int:
    if max_query_len > 1:
        return 1
    if requested >= 1:
        return requested
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    total_mblocks = max(1, num_requests * num_heads)
    return max(1, num_sms // total_mblocks)


def unified_attention_fxp_run(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    alibi_slopes: torch.Tensor | None,
    is_causal: bool,
    softmax_scale: float | None,
    fxp_int_bits: int,
    fxp_frac_bits: int,
    logit_softcap: float,
    window_size: int,
    num_kv_splits: int,
) -> None:
    if not (q.is_cuda and kv_cache.is_cuda and o.is_cuda):
        raise RuntimeError("unified_attention_fxp: tensors must be CUDA")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"unified_attention_fxp: unsupported q dtype {q.dtype}"
        )
    if kv_cache.dim() != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "kv_cache must be (num_blocks, 2, page_size, num_kv_heads, head_dim)"
        )
    if kv_cache.dtype != q.dtype or o.dtype != q.dtype:
        raise TypeError("kv_cache and o must share dtype with q")
    if query_start_loc.dtype != torch.int32:
        raise TypeError("query_start_loc must be int32")
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must be int32")
    if block_table.dtype != torch.int32:
        raise TypeError("block_table must be int32")
    if num_kv_splits < 0:
        raise ValueError(f"num_kv_splits must be >= 0, got {num_kv_splits}")
    if alibi_slopes is not None and alibi_slopes.numel() > 0:
        if alibi_slopes.dtype != torch.float32:
            raise TypeError("alibi_slopes must be float32")

    k_cache = kv_cache.select(1, 0)
    v_cache = kv_cache.select(1, 1)

    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    num_requests = seq_lens.shape[0]

    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if num_requests == 0:
        return

    ss = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    ss_log2 = float(ss) * _RCP_LN2
    softcap_log2 = float(logit_softcap) * _RCP_LN2

    has_alibi = alibi_slopes is not None and alibi_slopes.numel() > 0
    if has_alibi:
        alibi = alibi_slopes.contiguous()
    else:
        # Triton wants a real tensor; only read under HAS_ALIBI.
        alibi = q

    effective_splits = _pick_num_splits(
        int(max_query_len), int(num_kv_splits), int(num_requests), int(num_heads), q.device
    )

    scale, inv_scale, qmin, qmax, tl_int_dtype, torch_int_dtype = fxp_constants(
        int(fxp_int_bits), int(fxp_frac_bits)
    )
    io_dtype = _TORCH_TO_TL_FLOAT[q.dtype]

    T_total = q.shape[0]
    partial_max = torch.full(
        (T_total, num_heads, effective_splits),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    partial_denom = torch.zeros(
        (T_total, num_heads, effective_splits),
        device=q.device,
        dtype=torch_int_dtype,
    )
    partial_wv = torch.zeros(
        (T_total, num_heads, effective_splits, head_dim),
        device=q.device,
        dtype=torch_int_dtype,
    )

    block_d = max(_next_pow2(head_dim), 16)
    block_s = max(_next_pow2(effective_splits), 1)

    has_softcap = float(logit_softcap) > 0.0
    has_window = int(window_size) > 0

    grid_split = (num_requests, max(1, int(max_query_len)), num_heads * effective_splits)
    grid_combine = (num_requests, max(1, int(max_query_len)), num_heads)

    _attn_split_max_kernel[grid_split](
        q,
        k_cache,
        partial_max,
        query_start_loc,
        seq_lens,
        block_table,
        alibi,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        effective_splits,
        int(window_size),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        block_table.stride(0),
        ss_log2,
        softcap_log2,
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        HAS_ALIBI=has_alibi,
        HAS_SOFTCAP=has_softcap,
        IS_CAUSAL=bool(is_causal),
        HAS_WINDOW=has_window,
        BLOCK_D=block_d,
    )

    _attn_split_dv_kernel[grid_split](
        q,
        k_cache,
        v_cache,
        partial_max,
        partial_denom,
        partial_wv,
        query_start_loc,
        seq_lens,
        block_table,
        alibi,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        effective_splits,
        int(window_size),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_denom.stride(0),
        partial_denom.stride(1),
        partial_denom.stride(2),
        partial_wv.stride(0),
        partial_wv.stride(1),
        partial_wv.stride(2),
        partial_wv.stride(3),
        block_table.stride(0),
        ss_log2,
        softcap_log2,
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        HAS_ALIBI=has_alibi,
        HAS_SOFTCAP=has_softcap,
        IS_CAUSAL=bool(is_causal),
        HAS_WINDOW=has_window,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
    )

    _attn_combine_kernel[grid_combine](
        partial_denom,
        partial_wv,
        o,
        query_start_loc,
        num_heads,
        head_dim,
        effective_splits,
        partial_denom.stride(0),
        partial_denom.stride(1),
        partial_denom.stride(2),
        partial_wv.stride(0),
        partial_wv.stride(1),
        partial_wv.stride(2),
        partial_wv.stride(3),
        o.stride(0),
        o.stride(1),
        INV_SCALE=inv_scale,
        INT_DTYPE=tl_int_dtype,
        IO_DTYPE=io_dtype,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
    )
