// One CTA per row; warp-shuffle + smem integer reduction.
// Residual variant adds x into residual in fp32, in place.

#include "fixed_point.cuh"
#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace fxpr {
namespace detail {

namespace {

constexpr int kMaxThreads = 1024;
constexpr int kWarpSize = 32;

// Stays inside the 48KB per-CTA smem budget on every supported arch.
constexpr int kSmemCacheElems = 8192;

template <typename FxpInt>
__device__ __forceinline__ FxpInt warp_reduce_sum(FxpInt v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

// 16B load -> floats. Requires p 16B-aligned and VEC*sizeof(T) == 16.
template <typename T, int VEC>
__device__ __forceinline__ void vec_load_to_float(const T* __restrict__ p,
                                                  float (&out)[VEC]) {
  const int4 raw = *reinterpret_cast<const int4*>(p);
  const T* elems = reinterpret_cast<const T*>(&raw);
  #pragma unroll
  for (int e = 0; e < VEC; ++e) out[e] = static_cast<float>(elems[e]);
}

// 16B store of VEC floats downcast to T.
template <typename T, int VEC>
__device__ __forceinline__ void vec_store_from_float(T* __restrict__ p,
                                                     const float (&in)[VEC]) {
  int4 raw;
  T* elems = reinterpret_cast<T*>(&raw);
  #pragma unroll
  for (int e = 0; e < VEC; ++e) elems[e] = static_cast<T>(in[e]);
  *reinterpret_cast<int4*>(p) = raw;
}

template <typename FxpInt, bool kHasResidual, typename IOFloat, int kFracBits>
__global__ void rms_norm_kernel(
    const IOFloat* __restrict__ x,
    const IOFloat* __restrict__ w,
    IOFloat* __restrict__ y,
    IOFloat* __restrict__ residual,
    int64_t stride_x,
    int hidden_size,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const IOFloat* x_row = x + row * stride_x;
  IOFloat* y_row = y + row * stride_x;
  IOFloat* r_row = kHasResidual ? residual + row * stride_x : nullptr;

  extern __shared__ float x_cache[];
  const bool use_cache = (hidden_size <= kSmemCacheElems);

  // Vectorise when hidden_size divides the 16B vec width. Bit-exact: integer
  // add is associative, only the reduction tree shape changes.
  constexpr int kVec = 16 / static_cast<int>(sizeof(IOFloat));
  const bool use_vec = (hidden_size % kVec == 0);

  FxpInt partial = 0;
  if (use_vec) {
    const int n_vec = hidden_size / kVec;
    for (int v = tid; v < n_vec; v += nthreads) {
      const int base = v * kVec;
      float xv[kVec];
      vec_load_to_float<IOFloat, kVec>(x_row + base, xv);
      if constexpr (kHasResidual) {
        float rv[kVec];
        vec_load_to_float<IOFloat, kVec>(r_row + base, rv);
        #pragma unroll
        for (int e = 0; e < kVec; ++e) xv[e] += rv[e];
        vec_store_from_float<IOFloat, kVec>(r_row + base, xv);
      }
      if (use_cache) {
        #pragma unroll
        for (int e = 0; e < kVec; ++e) x_cache[base + e] = xv[e];
      }
      #pragma unroll
      for (int e = 0; e < kVec; ++e) {
        partial += float_to_fixed<FxpInt, kFracBits>(xv[e] * xv[e]);
      }
    }
  } else {
    for (int i = tid; i < hidden_size; i += nthreads) {
      float xi = static_cast<float>(x_row[i]);
      if constexpr (kHasResidual) {
        const float ri = static_cast<float>(r_row[i]);
        xi = xi + ri;
        r_row[i] = static_cast<IOFloat>(xi);
      }
      if (use_cache) x_cache[i] = xi;
      partial += float_to_fixed<FxpInt, kFracBits>(xi * xi);
    }
  }

  partial = warp_reduce_sum<FxpInt>(partial);

  __shared__ FxpInt warp_sums[kMaxThreads / kWarpSize];
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const int num_warps = (nthreads + kWarpSize - 1) / kWarpSize;

  if (lane == 0) {
    warp_sums[warp_id] = partial;
  }
  __syncthreads();

  FxpInt total;
  if (warp_id == 0) {
    FxpInt v = lane < num_warps ? warp_sums[lane] : FxpInt(0);
    v = warp_reduce_sum<FxpInt>(v);
    if (lane == 0) {
      warp_sums[0] = v;
    }
  }
  __syncthreads();
  total = warp_sums[0];

  const float sum_f = fixed_to_float<FxpInt, kFracBits>(total);
  const float mean_sq = fmaxf(sum_f / static_cast<float>(hidden_size), 0.0f);
  const float rrms = rsqrtf(mean_sq + eps);

  if (use_vec) {
    const int n_vec = hidden_size / kVec;
    for (int v = tid; v < n_vec; v += nthreads) {
      const int base = v * kVec;
      float wv[kVec];
      vec_load_to_float<IOFloat, kVec>(w + base, wv);
      float xv[kVec];
      if (use_cache) {
        #pragma unroll
        for (int e = 0; e < kVec; ++e) xv[e] = x_cache[base + e];
      } else if constexpr (kHasResidual) {
        vec_load_to_float<IOFloat, kVec>(r_row + base, xv);
      } else {
        vec_load_to_float<IOFloat, kVec>(x_row + base, xv);
      }
      float yv[kVec];
      #pragma unroll
      for (int e = 0; e < kVec; ++e) yv[e] = xv[e] * wv[e] * rrms;
      vec_store_from_float<IOFloat, kVec>(y_row + base, yv);
    }
  } else {
    for (int i = tid; i < hidden_size; i += nthreads) {
      const float xi = use_cache
          ? x_cache[i]
          : static_cast<float>(kHasResidual ? r_row[i] : x_row[i]);
      const float wi = static_cast<float>(w[i]);
      y_row[i] = static_cast<IOFloat>(xi * wi * rrms);
    }
  }
}

template <typename FxpInt, bool kHasResidual, typename IOFloat, int kFracBits>
void launch_typed(
    const at::Tensor& x_2d,
    const at::Tensor& w,
    at::Tensor& y_2d,
    at::Tensor* residual_2d_ptr,
    double eps) {
  const int batch = x_2d.size(0);
  const int hidden = x_2d.size(1);
  if (batch == 0 || hidden == 0) return;

  int block = 1;
  while (block < hidden && block < kMaxThreads) block <<= 1;
  if (block > kMaxThreads) block = kMaxThreads;

  const size_t smem_bytes =
      (hidden <= kSmemCacheElems) ? hidden * sizeof(float) : 0;

  auto stream = at::cuda::getCurrentCUDAStream();
  rms_norm_kernel<FxpInt, kHasResidual, IOFloat, kFracBits>
      <<<batch, block, smem_bytes, stream>>>(
      x_2d.data_ptr<IOFloat>(),
      w.data_ptr<IOFloat>(),
      y_2d.data_ptr<IOFloat>(),
      kHasResidual ? residual_2d_ptr->data_ptr<IOFloat>() : nullptr,
      x_2d.stride(0),
      hidden,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt, bool kHasResidual, int kFracBits>
void launch(
    const at::Tensor& x_2d,
    const at::Tensor& w,
    at::Tensor& y_2d,
    at::Tensor* residual_2d_ptr,
    double eps) {
  // Tying weight dtype to input keeps the dispatch table small.
  TORCH_CHECK(w.scalar_type() == x_2d.scalar_type(),
              "rms_norm: weight dtype must match input dtype (got w=",
              w.scalar_type(), ", x=", x_2d.scalar_type(), ")");
  switch (x_2d.scalar_type()) {
    case at::kFloat:
      launch_typed<FxpInt, kHasResidual, float, kFracBits>(
          x_2d, w, y_2d, residual_2d_ptr, eps);
      break;
    case at::kHalf:
      launch_typed<FxpInt, kHasResidual, at::Half, kFracBits>(
          x_2d, w, y_2d, residual_2d_ptr, eps);
      break;
    case at::kBFloat16:
      launch_typed<FxpInt, kHasResidual, at::BFloat16, kFracBits>(
          x_2d, w, y_2d, residual_2d_ptr, eps);
      break;
    default:
      TORCH_CHECK(false, "rms_norm: unsupported input dtype ",
                  x_2d.scalar_type());
  }
}

at::Tensor as_2d(const at::Tensor& x) {
  return x.reshape({-1, x.size(-1)});
}

template <bool kHasResidual, int kFracBits>
void rms_norm_dispatch_int(
    const at::Tensor& x_2d,
    const at::Tensor& w,
    at::Tensor& y_2d,
    at::Tensor* r_2d_ptr,
    double eps,
    int64_t int_bits) {
  switch (int_bits) {
    case 16:
      launch<int16_t, kHasResidual, kFracBits>(x_2d, w, y_2d, r_2d_ptr, eps);
      break;
    case 32:
      launch<int32_t, kHasResidual, kFracBits>(x_2d, w, y_2d, r_2d_ptr, eps);
      break;
    case 64:
      launch<int64_t, kHasResidual, kFracBits>(x_2d, w, y_2d, r_2d_ptr, eps);
      break;
  }
}

template <bool kHasResidual>
at::Tensor rms_norm_dispatch(
    at::Tensor x,
    at::Tensor* residual,
    at::Tensor w,
    double eps,
    int64_t int_bits,
    int64_t fxp_frac_bits) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_2d = as_2d(x).contiguous();
  auto y_2d = at::empty_like(x_2d);
  at::Tensor r_2d_storage;
  at::Tensor* r_2d_ptr = nullptr;
  if constexpr (kHasResidual) {
    r_2d_storage = as_2d(*residual);
    if (!r_2d_storage.is_contiguous()) {
      r_2d_storage = r_2d_storage.contiguous();
    }
    r_2d_ptr = &r_2d_storage;
  }

  switch (fxp_frac_bits) {
    case 8:
      rms_norm_dispatch_int<kHasResidual, 8>(x_2d, w, y_2d, r_2d_ptr, eps,
                                             int_bits);
      break;
    case 16:
      rms_norm_dispatch_int<kHasResidual, 16>(x_2d, w, y_2d, r_2d_ptr, eps,
                                              int_bits);
      break;
    case 32:
      rms_norm_dispatch_int<kHasResidual, 32>(x_2d, w, y_2d, r_2d_ptr, eps,
                                              int_bits);
      break;
  }

  if constexpr (kHasResidual) {
    if (r_2d_ptr->data_ptr() != residual->data_ptr()) {
      residual->copy_(r_2d_ptr->view_as(*residual));
    }
  }

  return y_2d.view_as(x);
}

}  // namespace

at::Tensor rms_norm_fxp_run(
    at::Tensor x,
    at::Tensor w,
    double eps,
    int64_t int_bits,
    int64_t fxp_frac_bits) {
  return rms_norm_dispatch<false>(
      std::move(x), nullptr, std::move(w), eps, int_bits, fxp_frac_bits);
}

at::Tensor rms_norm_fxp_residual_run(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor w,
    double eps,
    int64_t int_bits,
    int64_t fxp_frac_bits) {
  return rms_norm_dispatch<true>(
      std::move(x), &residual, std::move(w), eps, int_bits, fxp_frac_bits);
}

}  // namespace detail
}  // namespace fxpr
