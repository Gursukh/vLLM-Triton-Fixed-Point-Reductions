#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef FXPR_FRAC_BITS
#define FXPR_FRAC_BITS 16
#endif

namespace fxpr {

// Override at build time with -DFXPR_FRAC_BITS=N.
constexpr int kFracBits = FXPR_FRAC_BITS;
static_assert(kFracBits >= 0 && kFracBits < 127,
              "FXPR_FRAC_BITS must be in [0, 127)");

__device__ __forceinline__ constexpr float pow2_fp32() {
  return static_cast<float>(1ULL << kFracBits);
}

template <typename FxpInt>
__device__ __forceinline__ FxpInt float_to_fixed(float x) {
  static_assert(std::is_integral<FxpInt>::value && std::is_signed<FxpInt>::value,
                "fxp output must be a signed integer type");

  constexpr float scale = pow2_fp32();
  const float scaled = x * scale;
  const float rounded = rintf(scaled);

  // Clamp in float space; float(INT_MAX) rounds up one ulp for 32/64-bit.
  constexpr float qmin_f = static_cast<float>(std::numeric_limits<FxpInt>::min());
  constexpr float qmax_f = static_cast<float>(std::numeric_limits<FxpInt>::max());
  const float clamped = fminf(fmaxf(rounded, qmin_f), qmax_f);

  return static_cast<FxpInt>(clamped);
}

template <typename FxpInt>
__device__ __forceinline__ FxpInt half_to_fixed(__half x) {
  return float_to_fixed<FxpInt>(__half2float(x));
}

template <typename FxpInt>
__device__ __forceinline__ float fixed_to_float(FxpInt x) {
  static_assert(std::is_integral<FxpInt>::value && std::is_signed<FxpInt>::value,
                "fxp input must be a signed integer type");
  constexpr float inv_scale = 1.0f / pow2_fp32();
  return static_cast<float>(x) * inv_scale;
}

}  // namespace fxpr
