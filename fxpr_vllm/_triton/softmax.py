"""log-softmax. Row max in fp32, exp-sum in fxp, then (x - max) - log(sum)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


_BLOCK_N = 1024
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _log_softmax_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    N,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y

    row_max = -float("inf")
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        xi = tl.load(x_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(xi, axis=0))

    sum_fxp = tl.zeros((), dtype=INT_DTYPE)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        # -inf so tail lanes contribute exp(-inf) = 0 to the sum.
        xi = tl.load(x_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        ei = libdevice.exp(xi - row_max)
        sum_fxp += tl.sum(float_to_fixed(ei, SCALE, QMIN, QMAX, INT_DTYPE), axis=0)

    log_sum = libdevice.log(fixed_to_float(sum_fxp, INV_SCALE))

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        xi = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        yi = (xi - row_max) - log_sum
        tl.store(y_row + cols, yi.to(IO_DTYPE), mask=mask)


_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def log_softmax_fxp_run(
    x: torch.Tensor, int_bits: int, fxp_frac_bits: int
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("log_softmax_fxp: input must be CUDA")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"log_softmax_fxp: unsupported input dtype {x.dtype}; "
            f"expected one of {_SUPPORTED_DTYPES}"
        )

    x_c = x.contiguous()
    x_2d = x_c.reshape(-1, x_c.shape[-1]).contiguous()
    y_2d = torch.empty_like(x_2d)

    rows, N = x_2d.shape
    if rows == 0 or N == 0:
        return y_2d.view(x_c.shape)

    scale, inv_scale, qmin, qmax, tl_int_dtype, _ = fxp_constants(
        int_bits, fxp_frac_bits
    )
    io_dtype = _TORCH_TO_TL_FLOAT[x.dtype]

    grid = (rows,)
    _log_softmax_kernel[grid](
        x_2d,
        y_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        N,
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        IO_DTYPE=io_dtype,
        BLOCK_N=_BLOCK_N,
    )
    return y_2d.view(x_c.shape)
