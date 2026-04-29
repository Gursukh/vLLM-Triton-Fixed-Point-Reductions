// Device-side fixed-point conversion. Single __fmul_rn (blocks FMA
// fusion) + IEEE-rint + saturating cast. FxpInt must be signed; clamps
// to [INT_MIN, INT_MAX].

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace fxpr {

// 2^frac_bits as fp32 via IEEE bias trick; frac_bits must be in [0, 127].
__device__ __forceinline__ float pow2_fp32(int frac_bits) {
  return __int_as_float((127 + frac_bits) << 23);
}

template <typename FxpInt>
__device__ __forceinline__ FxpInt float_to_fixed(float x, int frac_bits) {
  static_assert(std::is_integral<FxpInt>::value && std::is_signed<FxpInt>::value,
                "fxp output must be a signed integer type");

  const float scale = pow2_fp32(frac_bits);
  const float scaled = __fmul_rn(x, scale);
  const float rounded = rintf(scaled);

  // float(INT_MAX) for 32/64-bit FxpInt rounds up by one ulp, so the
  // upward fminf clamp produces a value safely castable to FxpInt.
  constexpr float qmin_f = static_cast<float>(std::numeric_limits<FxpInt>::min());
  constexpr float qmax_f = static_cast<float>(std::numeric_limits<FxpInt>::max());
  const float clamped = fminf(fmaxf(rounded, qmin_f), qmax_f);

  return static_cast<FxpInt>(clamped);
}

template <typename FxpInt>
__device__ __forceinline__ float fixed_to_float(FxpInt x, int frac_bits) {
  static_assert(std::is_integral<FxpInt>::value && std::is_signed<FxpInt>::value,
                "fxp input must be a signed integer type");
  const float inv_scale = __int_as_float((127 - frac_bits) << 23);
  return __fmul_rn(static_cast<float>(x), inv_scale);
}

}  // namespace fxpr
