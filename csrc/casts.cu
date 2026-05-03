#include "fixed_point.cuh"
#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace fxpr {
namespace detail {

namespace {

// 4 elements per thread => LDG.128 / STG.128.
constexpr int kVec = 4;

template <typename FloatT, typename FxpInt, int kFracBits>
__global__ void float_to_fixed_kernel(
    const FloatT* __restrict__ x,
    FxpInt* __restrict__ y,
    int64_t n) {
  const int64_t base =
      (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) * kVec;
  if (base + kVec <= n) {
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float xf = static_cast<float>(x[base + i]);
      y[base + i] = float_to_fixed<FxpInt, kFracBits>(xf);
    }
  } else {
    for (int64_t idx = base; idx < n; ++idx) {
      const float xf = static_cast<float>(x[idx]);
      y[idx] = float_to_fixed<FxpInt, kFracBits>(xf);
    }
  }
}

template <typename FxpInt, typename FloatT, int kFracBits>
__global__ void fixed_to_float_kernel(
    const FxpInt* __restrict__ x,
    FloatT* __restrict__ y,
    int64_t n) {
  const int64_t base =
      (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) * kVec;
  if (base + kVec <= n) {
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float f = fixed_to_float<FxpInt, kFracBits>(x[base + i]);
      y[base + i] = static_cast<FloatT>(f);
    }
  } else {
    for (int64_t idx = base; idx < n; ++idx) {
      const float f = fixed_to_float<FxpInt, kFracBits>(x[idx]);
      y[idx] = static_cast<FloatT>(f);
    }
  }
}

template <typename FloatT, typename FxpInt, int kFracBits>
void launch_f2x(const at::Tensor& x, at::Tensor& y) {
  const int64_t n = x.numel();
  if (n == 0) return;
  constexpr int kBlock = 256;
  const int64_t blocks = (n + kBlock * kVec - 1) / (kBlock * kVec);
  auto stream = at::cuda::getCurrentCUDAStream();
  float_to_fixed_kernel<FloatT, FxpInt, kFracBits><<<blocks, kBlock, 0, stream>>>(
      x.data_ptr<FloatT>(), y.data_ptr<FxpInt>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt, typename FloatT, int kFracBits>
void launch_x2f(const at::Tensor& x, at::Tensor& y) {
  const int64_t n = x.numel();
  if (n == 0) return;
  constexpr int kBlock = 256;
  const int64_t blocks = (n + kBlock * kVec - 1) / (kBlock * kVec);
  auto stream = at::cuda::getCurrentCUDAStream();
  fixed_to_float_kernel<FxpInt, FloatT, kFracBits><<<blocks, kBlock, 0, stream>>>(
      x.data_ptr<FxpInt>(), y.data_ptr<FloatT>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kFracBits>
void float_to_fixed_dispatch_dtype(const at::Tensor& x_c, at::Tensor& y,
                                   int64_t int_bits) {
  switch (x_c.scalar_type()) {
    case at::kFloat:
      switch (int_bits) {
        case 16: launch_f2x<float, int16_t, kFracBits>(x_c, y); break;
        case 32: launch_f2x<float, int32_t, kFracBits>(x_c, y); break;
        case 64: launch_f2x<float, int64_t, kFracBits>(x_c, y); break;
      }
      break;
    case at::kHalf:
      switch (int_bits) {
        case 16: launch_f2x<at::Half, int16_t, kFracBits>(x_c, y); break;
        case 32: launch_f2x<at::Half, int32_t, kFracBits>(x_c, y); break;
        case 64: launch_f2x<at::Half, int64_t, kFracBits>(x_c, y); break;
      }
      break;
    case at::kDouble:
      switch (int_bits) {
        case 16: launch_f2x<double, int16_t, kFracBits>(x_c, y); break;
        case 32: launch_f2x<double, int32_t, kFracBits>(x_c, y); break;
        case 64: launch_f2x<double, int64_t, kFracBits>(x_c, y); break;
      }
      break;
    default:
      TORCH_CHECK(false, "float_to_fixed: unsupported input dtype ",
                  x_c.scalar_type());
  }
}

template <int kFracBits>
void fixed_to_float_dispatch_dtype(const at::Tensor& x_c, at::Tensor& y,
                                   int64_t float_bits) {
  switch (x_c.scalar_type()) {
    case at::kShort:
      switch (float_bits) {
        case 16: launch_x2f<int16_t, at::Half, kFracBits>(x_c, y); break;
        case 32: launch_x2f<int16_t, float, kFracBits>(x_c, y); break;
        case 64: launch_x2f<int16_t, double, kFracBits>(x_c, y); break;
      }
      break;
    case at::kInt:
      switch (float_bits) {
        case 16: launch_x2f<int32_t, at::Half, kFracBits>(x_c, y); break;
        case 32: launch_x2f<int32_t, float, kFracBits>(x_c, y); break;
        case 64: launch_x2f<int32_t, double, kFracBits>(x_c, y); break;
      }
      break;
    case at::kLong:
      switch (float_bits) {
        case 16: launch_x2f<int64_t, at::Half, kFracBits>(x_c, y); break;
        case 32: launch_x2f<int64_t, float, kFracBits>(x_c, y); break;
        case 64: launch_x2f<int64_t, double, kFracBits>(x_c, y); break;
      }
      break;
    default:
      TORCH_CHECK(false, "fixed_to_float: unsupported input dtype ",
                  x_c.scalar_type());
  }
}

}  // namespace

at::Tensor float_to_fixed_run(
    at::Tensor x,
    int64_t int_bits,
    int64_t fxp_frac_bits) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_c = x.contiguous();

  c10::ScalarType out_dtype;
  switch (int_bits) {
    case 16: out_dtype = at::kShort; break;
    case 32: out_dtype = at::kInt;   break;
    case 64: out_dtype = at::kLong;  break;
    default: TORCH_CHECK(false, "unreachable");
  }
  auto y = at::empty_like(x_c, x_c.options().dtype(out_dtype));

  switch (fxp_frac_bits) {
    case 8:  float_to_fixed_dispatch_dtype<8>(x_c, y, int_bits);  break;
    case 16: float_to_fixed_dispatch_dtype<16>(x_c, y, int_bits); break;
    case 32: float_to_fixed_dispatch_dtype<32>(x_c, y, int_bits); break;
    default: TORCH_CHECK(false, "unreachable");
  }
  return y;
}

at::Tensor fixed_to_float_run(
    at::Tensor x,
    int64_t float_bits,
    int64_t fxp_frac_bits) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_c = x.contiguous();

  c10::ScalarType out_dtype;
  switch (float_bits) {
    case 16: out_dtype = at::kHalf;   break;
    case 32: out_dtype = at::kFloat;  break;
    case 64: out_dtype = at::kDouble; break;
    default: TORCH_CHECK(false, "unreachable");
  }
  auto y = at::empty_like(x_c, x_c.options().dtype(out_dtype));

  switch (fxp_frac_bits) {
    case 8:  fixed_to_float_dispatch_dtype<8>(x_c, y, float_bits);  break;
    case 16: fixed_to_float_dispatch_dtype<16>(x_c, y, float_bits); break;
    case 32: fixed_to_float_dispatch_dtype<32>(x_c, y, float_bits); break;
    default: TORCH_CHECK(false, "unreachable");
  }
  return y;
}

}  // namespace detail
}  // namespace fxpr
