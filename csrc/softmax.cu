// Per-row deterministic log-softmax via fixed-point exp sum. One CTA
// per row; three passes: row_max -> integer-sum exp(x - row_max) ->
// y = (x - row_max) - log(sum). Only summation is integer; max/log
// are single ops per output.

#include "fixed_point.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

namespace fxpr {

namespace {

constexpr int kMaxThreads = 1024;
constexpr int kWarpSize = 32;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  return v;
}

template <typename FxpInt>
__global__ void log_softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t stride_x,
    int64_t stride_y,
    int N,
    int frac_bits) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const float* x_row = x + row * stride_x;
  float* y_row = y + row * stride_y;

  // Pass 1: per-thread max -> warp -> block.
  float partial_max = -INFINITY;
  for (int i = tid; i < N; i += nthreads) {
    partial_max = fmaxf(partial_max, x_row[i]);
  }
  partial_max = warp_reduce_max(partial_max);

  __shared__ float warp_max[kMaxThreads / kWarpSize];
  __shared__ FxpInt warp_sum[kMaxThreads / kWarpSize];
  __shared__ float row_max_s;
  __shared__ float log_sum_s;

  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const int num_warps = (nthreads + kWarpSize - 1) / kWarpSize;

  if (lane == 0) warp_max[warp_id] = partial_max;
  __syncthreads();

  if (warp_id == 0) {
    float v = lane < num_warps ? warp_max[lane] : -INFINITY;
    v = warp_reduce_max(v);
    if (lane == 0) row_max_s = v;
  }
  __syncthreads();

  const float row_max = row_max_s;

  // Pass 2: per-thread exp(x - row_max), quantise, integer block sum.
  FxpInt partial_sum = 0;
  for (int i = tid; i < N; i += nthreads) {
    const float xi = x_row[i];
    const float ei = expf(xi - row_max);
    partial_sum += float_to_fixed<FxpInt>(ei, frac_bits);
  }
  partial_sum = warp_reduce_sum<FxpInt>(partial_sum);

  if (lane == 0) warp_sum[warp_id] = partial_sum;
  __syncthreads();

  if (warp_id == 0) {
    FxpInt v = lane < num_warps ? warp_sum[lane] : FxpInt(0);
    v = warp_reduce_sum<FxpInt>(v);
    if (lane == 0) {
      const float sum_f = fixed_to_float<FxpInt>(v, frac_bits);
      log_sum_s = logf(sum_f);
    }
  }
  __syncthreads();

  const float log_sum = log_sum_s;

  // Pass 3: y = (x - row_max) - log_sum.
  for (int i = tid; i < N; i += nthreads) {
    const float xi = x_row[i];
    y_row[i] = (xi - row_max) - log_sum;
  }
}

template <typename FxpInt>
void launch(
    const at::Tensor& x_2d,
    at::Tensor& y_2d,
    int64_t frac_bits) {
  const int rows = x_2d.size(0);
  const int N = x_2d.size(1);
  if (rows == 0 || N == 0) return;

  int block = 1;
  while (block < N && block < kMaxThreads) block <<= 1;
  if (block > kMaxThreads) block = kMaxThreads;

  auto stream = at::cuda::getCurrentCUDAStream();
  log_softmax_kernel<FxpInt><<<rows, block, 0, stream>>>(
      x_2d.data_ptr<float>(),
      y_2d.data_ptr<float>(),
      x_2d.stride(0),
      y_2d.stride(0),
      N,
      static_cast<int>(frac_bits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor log_softmax_fxp_op(
    torch::Tensor x,
    int64_t frac_bits,
    int64_t int_bits) {
  TORCH_CHECK(x.is_cuda(), "log_softmax_fxp: input must be CUDA");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const auto orig_dtype = x.scalar_type();
  auto x_f32 = x.to(at::kFloat).contiguous();
  auto x_2d = x_f32.reshape({-1, x_f32.size(-1)}).contiguous();
  auto y_2d = at::empty_like(x_2d);

  switch (int_bits) {
    case 16: launch<int16_t>(x_2d, y_2d, frac_bits); break;
    case 32: launch<int32_t>(x_2d, y_2d, frac_bits); break;
    case 64: launch<int64_t>(x_2d, y_2d, frac_bits); break;
  }

  auto y = y_2d.view(x_f32.sizes());
  if (orig_dtype != at::kFloat) y = y.to(orig_dtype);
  return y;
}

}  // namespace fxpr
