// RMSNorm with deterministic fixed-point sum of squares. One CTA per
// row; warp-shuffle + shared-memory integer reduction (associative).
// Fused-residual variant does residual += x in fp32 in place.

#include "fixed_point.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace fxpr {

namespace {

constexpr int kMaxThreads = 1024;
constexpr int kWarpSize = 32;

// Cap on the pass-1 → pass-2 smem cache. Sized to stay under the 48KB
// per-CTA budget on every supported arch.
constexpr int kSmemCacheElems = 8192;

template <typename FxpInt>
__device__ __forceinline__ FxpInt warp_reduce_sum(FxpInt v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

template <typename FxpInt, bool kHasResidual, typename WeightT>
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const WeightT* __restrict__ w,
    float* __restrict__ y,
    float* __restrict__ residual,
    int64_t stride_x,
    int hidden_size,
    float eps,
    int frac_bits) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const float* x_row = x + row * stride_x;
  float* y_row = y + row * stride_x;
  float* r_row = kHasResidual ? residual + row * stride_x : nullptr;

  // Stage x (post-residual when fused) into smem when it fits, so pass 2
  // can read from there instead of HBM. Bit-identical fallback otherwise.
  extern __shared__ float x_cache[];
  const bool use_cache = (hidden_size <= kSmemCacheElems);

  // Pass 1: per-thread fixed-point partial sum of squares. Residual
  // fusion writes the new residual once here, so x flows through global
  // memory twice or once when use_cache is true.
  FxpInt partial = 0;
  for (int i = tid; i < hidden_size; i += nthreads) {
    float xi = x_row[i];
    if constexpr (kHasResidual) {
      const float ri = r_row[i];
      // The __fmul_rn in the square below blocks FMA contraction with this add.
      xi = xi + ri;
      r_row[i] = xi;
    }
    if (use_cache) x_cache[i] = xi;
    const float xi2 = __fmul_rn(xi, xi);
    partial += float_to_fixed<FxpInt>(xi2, frac_bits);
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

  const float sum_f = fixed_to_float<FxpInt>(total, frac_bits);
  const float mean_sq = fmaxf(sum_f / static_cast<float>(hidden_size), 0.0f);
  const float rrms = rsqrtf(mean_sq + eps);

  // Pass 2: y = x * w * rrms.
  for (int i = tid; i < hidden_size; i += nthreads) {
    const float xi = use_cache
        ? x_cache[i]
        : (kHasResidual ? r_row[i] : x_row[i]);
    const float wi = static_cast<float>(w[i]);
    y_row[i] = xi * wi * rrms;
  }
}

template <typename FxpInt, bool kHasResidual, typename WeightT>
void launch_typed(
    const at::Tensor& x_2d,
    const at::Tensor& w,
    at::Tensor& y_2d,
    at::Tensor* residual_2d_ptr,
    double eps,
    int64_t frac_bits) {
  const int batch = x_2d.size(0);
  const int hidden = x_2d.size(1);
  if (batch == 0 || hidden == 0) return;

  int block = 1;
  while (block < hidden && block < kMaxThreads) block <<= 1;
  if (block > kMaxThreads) block = kMaxThreads;

  const size_t smem_bytes =
      (hidden <= kSmemCacheElems) ? hidden * sizeof(float) : 0;

  auto stream = at::cuda::getCurrentCUDAStream();
  rms_norm_kernel<FxpInt, kHasResidual, WeightT><<<batch, block, smem_bytes, stream>>>(
      x_2d.data_ptr<float>(),
      w.data_ptr<WeightT>(),
      y_2d.data_ptr<float>(),
      kHasResidual ? residual_2d_ptr->data_ptr<float>() : nullptr,
      x_2d.stride(0),
      hidden,
      static_cast<float>(eps),
      static_cast<int>(frac_bits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt, bool kHasResidual>
void launch(
    const at::Tensor& x_2d,
    const at::Tensor& w,
    at::Tensor& y_2d,
    at::Tensor* residual_2d_ptr,
    double eps,
    int64_t frac_bits) {
  switch (w.scalar_type()) {
    case at::kFloat:
      launch_typed<FxpInt, kHasResidual, float>(
          x_2d, w, y_2d, residual_2d_ptr, eps, frac_bits);
      break;
    case at::kHalf:
      launch_typed<FxpInt, kHasResidual, at::Half>(
          x_2d, w, y_2d, residual_2d_ptr, eps, frac_bits);
      break;
    case at::kBFloat16:
      launch_typed<FxpInt, kHasResidual, at::BFloat16>(
          x_2d, w, y_2d, residual_2d_ptr, eps, frac_bits);
      break;
    default:
      TORCH_CHECK(false, "rms_norm: unsupported weight dtype ",
                  w.scalar_type());
  }
}

at::Tensor as_2d(const at::Tensor& x) {
  return x.reshape({-1, x.size(-1)});
}

template <bool kHasResidual>
at::Tensor rms_norm_dispatch(
    at::Tensor x,
    at::Tensor* residual,
    at::Tensor w,
    double eps,
    int64_t frac_bits,
    int64_t int_bits) {
  TORCH_CHECK(x.is_cuda(), "rms_norm: x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "rms_norm: w must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "rms_norm: x must be float32");
  TORCH_CHECK(
      w.scalar_type() == at::kFloat || w.scalar_type() == at::kHalf
          || w.scalar_type() == at::kBFloat16,
      "rms_norm: w must be float32 / float16 / bfloat16");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  if constexpr (kHasResidual) {
    TORCH_CHECK(residual != nullptr && residual->is_cuda(),
                "rms_norm: residual must be CUDA");
    TORCH_CHECK(residual->scalar_type() == at::kFloat,
                "rms_norm: residual must be float32");
    TORCH_CHECK(residual->sizes() == x.sizes(),
                "rms_norm: residual must have the same shape as x");
  }

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

  switch (int_bits) {
    case 16:
      launch<int16_t, kHasResidual>(x_2d, w, y_2d, r_2d_ptr, eps, frac_bits);
      break;
    case 32:
      launch<int32_t, kHasResidual>(x_2d, w, y_2d, r_2d_ptr, eps, frac_bits);
      break;
    case 64:
      launch<int64_t, kHasResidual>(x_2d, w, y_2d, r_2d_ptr, eps, frac_bits);
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

torch::Tensor rms_norm_fxp_op(
    torch::Tensor x,
    torch::Tensor w,
    double eps,
    int64_t frac_bits,
    int64_t int_bits) {
  return rms_norm_dispatch<false>(
      std::move(x), nullptr, std::move(w), eps, frac_bits, int_bits);
}

torch::Tensor rms_norm_fxp_residual_op(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w,
    double eps,
    int64_t frac_bits,
    int64_t int_bits) {
  return rms_norm_dispatch<true>(
      std::move(x), &residual, std::move(w), eps, frac_bits, int_bits);
}

}  // namespace fxpr
