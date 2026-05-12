"""RMSNorm with a fixed-point sum-of-squares.

One program per row, chunked over hidden so it works at any width.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


_BLOCK_N = 1024
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    r_ptr,
    stride_x,
    hidden,
    eps,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_x
    if HAS_RESIDUAL:
        r_row = r_ptr + row * stride_x

    acc = tl.zeros((), dtype=INT_DTYPE)
    for off in range(0, hidden, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < hidden
        xi = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            ri = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
            xi = xi + ri
            tl.store(r_row + cols, xi.to(IO_DTYPE), mask=mask)
        sq = xi * xi
        acc += tl.sum(float_to_fixed(sq, SCALE, QMIN, QMAX, INT_DTYPE), axis=0)

    sum_f = fixed_to_float(acc, INV_SCALE)
    mean_sq = tl.maximum(sum_f / hidden.to(tl.float32), 0.0)
    rrms = libdevice.rsqrt(mean_sq + eps)

    for off in range(0, hidden, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < hidden
        if HAS_RESIDUAL:
            xi = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            xi = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        wi = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        yi = xi * wi * rrms
        tl.store(y_row + cols, yi.to(IO_DTYPE), mask=mask)


_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _common_launch(
    x: torch.Tensor,
    w: torch.Tensor,
    residual: torch.Tensor | None,
    eps: float,
    int_bits: int,
    fxp_frac_bits: int,
) -> torch.Tensor:
    if not x.is_cuda or not w.is_cuda:
        raise RuntimeError("rms_norm: x and w must be CUDA")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"rms_norm: unsupported input dtype {x.dtype}; "
            f"expected one of {_SUPPORTED_DTYPES}"
        )
    if w.dtype != x.dtype:
        raise TypeError("rms_norm: weight dtype must match input dtype")
    if residual is not None:
        if not residual.is_cuda:
            raise RuntimeError("rms_norm: residual must be CUDA")
        if residual.dtype != x.dtype:
            raise TypeError("rms_norm: residual dtype must match input dtype")
        if residual.shape != x.shape:
            raise ValueError("rms_norm: residual shape must match x")

    x_2d = _as_2d(x).contiguous()
    y_2d = torch.empty_like(x_2d)
    if residual is not None:
        r_2d = _as_2d(residual)
        r_storage = r_2d if r_2d.is_contiguous() else r_2d.contiguous()
    else:
        r_storage = None

    batch, hidden = x_2d.shape
    if batch == 0 or hidden == 0:
        if residual is not None and r_storage.data_ptr() != residual.data_ptr():
            residual.copy_(r_storage.view_as(residual))
        return y_2d.view_as(x)

    scale, inv_scale, qmin, qmax, tl_int_dtype, _ = fxp_constants(
        int_bits, fxp_frac_bits
    )
    io_dtype = _TORCH_TO_TL_FLOAT[x.dtype]

    grid = (batch,)
    _rms_norm_kernel[grid](
        x_2d,
        w,
        y_2d,
        # Triton needs a real tensor; the kernel only reads it under HAS_RESIDUAL.
        r_storage if r_storage is not None else x_2d,
        x_2d.stride(0),
        hidden,
        float(eps),
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        IO_DTYPE=io_dtype,
        HAS_RESIDUAL=residual is not None,
        BLOCK_N=_BLOCK_N,
    )

    if residual is not None and r_storage.data_ptr() != residual.data_ptr():
        residual.copy_(r_storage.view_as(residual))
    return y_2d.view_as(x)


def rms_norm_fxp_run(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    int_bits: int,
    fxp_frac_bits: int,
) -> torch.Tensor:
    return _common_launch(x, w, None, eps, int_bits, fxp_frac_bits)


def rms_norm_fxp_residual_run(
    x: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    int_bits: int,
    fxp_frac_bits: int,
) -> torch.Tensor:
    return _common_launch(x, w, residual, eps, int_bits, fxp_frac_bits)
