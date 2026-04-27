"""Public op surface — thin shim over the CUDA extension.

After the CUDA migration, every operation runs through `fxpr_vllm._cuda`,
which registers `torch.ops.fxpr.*` at import time. This module re-exports
those ops under their Python names and provides a small Python helper
for the tier-2 int8 GEMM that bundles per-row scale computation +
quantisation + MMA into one call.
"""

from __future__ import annotations

import torch

from . import _cuda  # noqa: F401  (registers torch.ops.fxpr.*)


# Re-exports of the registered torch.ops.fxpr.* ops.
gemm_fxp = torch.ops.fxpr.gemm_fxp
gemm_fxp_int8 = torch.ops.fxpr.gemm_fxp_int8
rms_norm_fxp = torch.ops.fxpr.rms_norm_fxp
rms_norm_fxp_residual = torch.ops.fxpr.rms_norm_fxp_residual
log_softmax_fxp = torch.ops.fxpr.log_softmax_fxp
compute_per_row_scale = torch.ops.fxpr.compute_per_row_scale
float_to_fixed = torch.ops.fxpr.float_to_fixed
fixed_to_float = torch.ops.fxpr.fixed_to_float


def quantise_to_int8(
    x: torch.Tensor, scale_fp16: torch.Tensor
) -> torch.Tensor:
    """Per-row int8 quantisation: x_int8[i, k] = clamp(round(x[i,k] / s[i]), [-127, 127]).

    Both x and scale_fp16 are expected on the same device.
    The scale tensor must be precomputed via :func:`compute_per_row_scale`
    (or any equivalently split-invariant procedure) so the result stays
    bit-identical regardless of how rows are partitioned across launches.
    """
    assert x.is_cuda and scale_fp16.is_cuda
    assert x.dim() == 2
    assert scale_fp16.shape == (x.shape[0],)
    s = scale_fp16.to(torch.float32).unsqueeze(-1)
    q = torch.round(x / s).clamp_(-127, 127).to(torch.int8)
    return q


def launch_gemm_fxp_mma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    q_frac_bits: int = 8,
    frac_bits: int = 16,
    fxp_int_bits: int = 32,
    block_k: int | None = None,
) -> torch.Tensor:
    """Tier-2 int8 GEMM with per-row/per-col scales.

    The original Triton-era signature took q_frac_bits to drive
    static int8 scaling. The deterministic CUDA path computes
    split-invariant per-row scales instead; q_frac_bits and
    block_k are accepted for backwards compatibility but ignored.
    The K-accumulator is integer over int8 products, so the result is
    bit-identical for any K-block schedule.
    """
    del q_frac_bits, block_k

    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]

    a_c = a.contiguous()
    b_t = b.t().contiguous()  # per-col scales of B = per-row scales of B.T

    a_scale = compute_per_row_scale(a_c, 1e-8)
    b_scale = compute_per_row_scale(b_t, 1e-8)

    a_int8 = quantise_to_int8(a_c, a_scale)
    b_t_int8 = quantise_to_int8(b_t, b_scale)
    b_int8 = b_t_int8.t().contiguous()

    return gemm_fxp_int8(a_int8, a_scale, b_int8, b_scale, frac_bits, fxp_int_bits)
