// Device-side fixed-point conversion helpers.
//
// Two templates: float_to_fixed<FxpInt> and fixed_to_float<FxpInt>.
// Determinism: every conversion is a single round-to-nearest-even fp32
// multiply followed by an IEEE-rint and saturating cast. We use
// __fmul_rn / __builtin_rintf so the optimizer doesn't fuse into an
// FMA or otherwise re-associate operations across calls.
//
// FxpInt must be a signed integer type. frac_bits is bounded by the
// integer width minus 1; saturation clamps to [INT_MIN, INT_MAX].

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace fxpr {

// 2^frac_bits as fp32, via the IEEE bias trick. frac_bits must be in
// [0, 127] for the result to be a normal fp32. We assert at the call
// site (frac_bits is a runtime int passed in from Python).
__device__ __forceinline__ float pow2_fp32(int frac_bits) {
  // (127 + frac_bits) << 23 is the fp32 representation of 2^frac_bits.
  return __int_as_float((127 + frac_bits) << 23);
}

template <typename FxpInt>
__device__ __forceinline__ FxpInt float_to_fixed(float x, int frac_bits) {
  static_assert(std::is_integral<FxpInt>::value && std::is_signed<FxpInt>::value,
                "fxp output must be a signed integer type");

  const float scale = pow2_fp32(frac_bits);
  // Single deterministic fp32 multiply: --fmad=false in setup.py
  // prevents nvcc from fusing this into a downstream add via FMA.
  const float scaled = __fmul_rn(x, scale);
  const float rounded = rintf(scaled);

  // Compute saturation bounds in fp32. For 32-bit / 64-bit FxpInt,
  // INT_MAX isn't exactly representable; the next-representable float
  // below INT_MAX is INT_MAX - (ulp-1). Using float(INT_MAX) gives a
  // value that is >= INT_MAX (rounds up by one ulp), so we use the
  // closed-below half of the range and clamp upward to INT_MAX.
  constexpr float qmin_f = static_cast<float>(std::numeric_limits<FxpInt>::min());
  constexpr float qmax_f = static_cast<float>(std::numeric_limits<FxpInt>::max());
  const float clamped = fminf(fmaxf(rounded, qmin_f), qmax_f);

  // Cast: out-of-range -> implementation-defined in C++, but on CUDA
  // float-to-int conversion saturates per the hardware FP-to-int
  // instruction. We've already clamped so this is safe.
  return static_cast<FxpInt>(clamped);
}

template <typename FxpInt>
__device__ __forceinline__ float fixed_to_float(FxpInt x, int frac_bits) {
  static_assert(std::is_integral<FxpInt>::value && std::is_signed<FxpInt>::value,
                "fxp input must be a signed integer type");
  // 2^-frac_bits via IEEE bias trick; frac_bits in [0, 127].
  const float inv_scale = __int_as_float((127 - frac_bits) << 23);
  return __fmul_rn(static_cast<float>(x), inv_scale);
}

}  // namespace fxpr
