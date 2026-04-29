// Element-wise float<->fixed conversion ops, thin wrappers over fixed_point.cuh.

#include "fixed_point.cuh"
#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace fxpr {
namespace detail {

namespace {

// 4 elements per thread → one LDG.128 + one STG.128 per thread instead of
// 4 scalar transactions, roughly tripling bandwidth on memory-bound casts.
constexpr int kVec = 4;

template <typename FloatT, typename FxpInt>
__global__ void float_to_fixed_kernel(
    const FloatT* __restrict__ x,
    FxpInt* __restrict__ y,
    int64_t n,
    int frac_bits) {
  const int64_t base =
      (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) * kVec;
  if (base + kVec <= n) {
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float xf = static_cast<float>(x[base + i]);
      y[base + i] = float_to_fixed<FxpInt>(xf, frac_bits);
    }
  } else {
    for (int64_t idx = base; idx < n; ++idx) {
      const float xf = static_cast<float>(x[idx]);
      y[idx] = float_to_fixed<FxpInt>(xf, frac_bits);
    }
  }
}

template <typename FxpInt, typename FloatT>
__global__ void fixed_to_float_kernel(
    const FxpInt* __restrict__ x,
    FloatT* __restrict__ y,
    int64_t n,
    int frac_bits) {
  const int64_t base =
      (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) * kVec;
  if (base + kVec <= n) {
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float f = fixed_to_float<FxpInt>(x[base + i], frac_bits);
      y[base + i] = static_cast<FloatT>(f);
    }
  } else {
    for (int64_t idx = base; idx < n; ++idx) {
      const float f = fixed_to_float<FxpInt>(x[idx], frac_bits);
      y[idx] = static_cast<FloatT>(f);
    }
  }
}

template <typename FloatT, typename FxpInt>
void launch_f2x(const at::Tensor& x, at::Tensor& y, int frac_bits) {
  const int64_t n = x.numel();
  if (n == 0) return;
  constexpr int kBlock = 256;
  const int64_t blocks = (n + kBlock * kVec - 1) / (kBlock * kVec);
  auto stream = at::cuda::getCurrentCUDAStream();
  float_to_fixed_kernel<FloatT, FxpInt><<<blocks, kBlock, 0, stream>>>(
      x.data_ptr<FloatT>(), y.data_ptr<FxpInt>(), n, frac_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt, typename FloatT>
void launch_x2f(const at::Tensor& x, at::Tensor& y, int frac_bits) {
  const int64_t n = x.numel();
  if (n == 0) return;
  constexpr int kBlock = 256;
  const int64_t blocks = (n + kBlock * kVec - 1) / (kBlock * kVec);
  auto stream = at::cuda::getCurrentCUDAStream();
  fixed_to_float_kernel<FxpInt, FloatT><<<blocks, kBlock, 0, stream>>>(
      x.data_ptr<FxpInt>(), y.data_ptr<FloatT>(), n, frac_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor float_to_fixed_run(
    at::Tensor x,
    int64_t frac_bits,
    int64_t int_bits) {
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

  const int fb = static_cast<int>(frac_bits);
  switch (x_c.scalar_type()) {
    case at::kFloat:
      switch (int_bits) {
        case 16: launch_f2x<float, int16_t>(x_c, y, fb); break;
        case 32: launch_f2x<float, int32_t>(x_c, y, fb); break;
        case 64: launch_f2x<float, int64_t>(x_c, y, fb); break;
      }
      break;
    case at::kHalf:
      switch (int_bits) {
        case 16: launch_f2x<at::Half, int16_t>(x_c, y, fb); break;
        case 32: launch_f2x<at::Half, int32_t>(x_c, y, fb); break;
        case 64: launch_f2x<at::Half, int64_t>(x_c, y, fb); break;
      }
      break;
    case at::kDouble:
      switch (int_bits) {
        case 16: launch_f2x<double, int16_t>(x_c, y, fb); break;
        case 32: launch_f2x<double, int32_t>(x_c, y, fb); break;
        case 64: launch_f2x<double, int64_t>(x_c, y, fb); break;
      }
      break;
    default:
      TORCH_CHECK(false, "float_to_fixed: unsupported input dtype ",
                  x_c.scalar_type());
  }
  return y;
}

at::Tensor fixed_to_float_run(
    at::Tensor x,
    int64_t frac_bits,
    int64_t float_bits) {
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

  const int fb = static_cast<int>(frac_bits);
  switch (x_c.scalar_type()) {
    case at::kShort:
      switch (float_bits) {
        case 16: launch_x2f<int16_t, at::Half>(x_c, y, fb); break;
        case 32: launch_x2f<int16_t, float>(x_c, y, fb); break;
        case 64: launch_x2f<int16_t, double>(x_c, y, fb); break;
      }
      break;
    case at::kInt:
      switch (float_bits) {
        case 16: launch_x2f<int32_t, at::Half>(x_c, y, fb); break;
        case 32: launch_x2f<int32_t, float>(x_c, y, fb); break;
        case 64: launch_x2f<int32_t, double>(x_c, y, fb); break;
      }
      break;
    case at::kLong:
      switch (float_bits) {
        case 16: launch_x2f<int64_t, at::Half>(x_c, y, fb); break;
        case 32: launch_x2f<int64_t, float>(x_c, y, fb); break;
        case 64: launch_x2f<int64_t, double>(x_c, y, fb); break;
      }
      break;
    default:
      TORCH_CHECK(false, "fixed_to_float: unsupported input dtype ",
                  x_c.scalar_type());
  }
  return y;
}

}  // namespace detail
}  // namespace fxpr
