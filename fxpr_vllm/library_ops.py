""" Wrapper functions for the custom CUDA ops so they can be used in TorchDynamo graph capture. """

from __future__ import annotations

import torch

from . import _cuda  # noqa: F401


def _int_dtype_for_bits(int_bits: int) -> torch.dtype:
    if int_bits == 16:
        return torch.int16
    if int_bits == 32:
        return torch.int32
    if int_bits == 64:
        return torch.int64
    raise ValueError(f"int_bits must be 16/32/64, got {int_bits}")


def _float_dtype_for_bits(float_bits: int) -> torch.dtype:
    if float_bits == 16:
        return torch.float16
    if float_bits == 32:
        return torch.float32
    if float_bits == 64:
        return torch.float64
    raise ValueError(f"float_bits must be 16/32/64, got {float_bits}")


@torch.library.register_fake("fxpr::float_to_fixed")
def _float_to_fixed_fake(x, int_bits, fxp_frac_bits):
    return torch.empty_like(x, dtype=_int_dtype_for_bits(int(int_bits)))


@torch.library.register_fake("fxpr::fixed_to_float")
def _fixed_to_float_fake(x, float_bits, fxp_frac_bits):
    return torch.empty_like(x, dtype=_float_dtype_for_bits(int(float_bits)))


@torch.library.register_fake("fxpr::rms_norm_fxp")
def _rms_norm_fxp_fake(x, weight_fp32, eps, fxp_int_bits, fxp_frac_bits):
    return torch.empty_like(x)


@torch.library.register_fake("fxpr::rms_norm_fxp_residual")
def _rms_norm_fxp_residual_fake(
    x, residual, weight_fp32, eps, fxp_int_bits, fxp_frac_bits
):
    return torch.empty_like(x)


@torch.library.register_fake("fxpr::log_softmax_fxp")
def _log_softmax_fxp_fake(x, fxp_int_bits, fxp_frac_bits):
    return torch.empty_like(x)


@torch.library.register_fake("fxpr::gemm_fxp")
def _gemm_fxp_fake(a, b, bias, fxp_int_bits, fxp_frac_bits):
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.register_fake("fxpr::unified_attention_fxp")
def _unified_attention_fxp_fake(
    q,
    kv_cache,
    o,
    query_start_loc,
    seq_lens,
    block_table,
    max_query_len,
    alibi_slopes,
    is_causal,
    softmax_scale,
    fxp_int_bits,
    fxp_frac_bits,
    logit_softcap,
    window_size,
    num_kv_splits,
):
    return None


gemm_fxp = torch.ops.fxpr.gemm_fxp
rms_norm_fxp = torch.ops.fxpr.rms_norm_fxp
rms_norm_fxp_residual = torch.ops.fxpr.rms_norm_fxp_residual
log_softmax_fxp = torch.ops.fxpr.log_softmax_fxp
float_to_fixed = torch.ops.fxpr.float_to_fixed
fixed_to_float = torch.ops.fxpr.fixed_to_float
