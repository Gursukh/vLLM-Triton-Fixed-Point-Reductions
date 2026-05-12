"""Float <-> signed fixed-point helpers shared by every kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


_INT_DTYPE_BY_BITS = {
    16: tl.int16,
    32: tl.int32,
    64: tl.int64,
}

_TORCH_INT_DTYPE_BY_BITS = {
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}


def fxp_constants(int_bits: int, frac_bits: int):
    if int_bits not in _INT_DTYPE_BY_BITS:
        raise ValueError(f"int_bits must be 16/32/64, got {int_bits}")
    if not (0 <= frac_bits < 64):
        raise ValueError(f"frac_bits must satisfy 0 <= N < 64, got {frac_bits}")
    scale = float(1 << frac_bits)
    inv_scale = 1.0 / scale
    qmax_i = (1 << (int_bits - 1)) - 1
    qmin_i = -(1 << (int_bits - 1))
    return (
        scale,
        inv_scale,
        float(qmin_i),
        float(qmax_i),
        _INT_DTYPE_BY_BITS[int_bits],
        _TORCH_INT_DTYPE_BY_BITS[int_bits],
    )


@triton.jit
def float_to_fixed(
    x,
    SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
):
    # libdevice.rint is round-half-to-even. tl.math.round rounds half-away-from-zero,
    # which breaks parity on .5 boundaries.
    scaled = x.to(tl.float32) * SCALE
    rounded = libdevice.rint(scaled)
    clamped = tl.minimum(tl.maximum(rounded, QMIN), QMAX)
    return clamped.to(INT_DTYPE)


@triton.jit
def fixed_to_float(x, INV_SCALE: tl.constexpr):
    return x.to(tl.float32) * INV_SCALE
