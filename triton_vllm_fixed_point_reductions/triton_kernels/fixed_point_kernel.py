import triton
import triton.language as tl


@triton.jit
def float_to_fixed(x: tl.tensor, frac_bits: tl.constexpr, dtype: tl.constexpr):

    tl.static_assert(
        dtype == tl.int16 or dtype == tl.int32 or dtype == tl.int64,
        "Fixed-point conversion must use a signed integer dtype",
    )
    tl.static_assert(
        x.dtype == tl.float16 or x.dtype == tl.float32 or x.dtype == tl.float64,
        "x must be of a floating-point type",
    )

    bits: tl.constexpr = dtype.primitive_bitwidth
    mantissa_bits: tl.constexpr = x.dtype.fp_mantissa_width + 1

    safe_shift: tl.constexpr = 0 if bits <= mantissa_bits else (bits - mantissa_bits)
    qmax_f: tl.constexpr = float((1 << (bits - 1)) - (1 << safe_shift))
    qmin_f: tl.constexpr = float(-(1 << (bits - 1)))

    scale = 2.0**frac_bits
    scaled = x * scale
    clamped = tl.minimum(tl.maximum(scaled, qmin_f), qmax_f)

    rounded = tl.extra.libdevice.rint(clamped)
    return rounded.to(dtype)


@triton.jit
def fixed_to_float(x: tl.tensor, frac_bits: tl.constexpr, dtype: tl.constexpr):

    tl.static_assert(
        dtype == tl.float16 or dtype == tl.float32 or dtype == tl.float64,
        "Fixed-point conversion must use a float dtype",
    )

    inv_scale = 2.0**-frac_bits
    return x.to(dtype) * inv_scale
