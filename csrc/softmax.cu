// Three passes per row: row_max, fxp sum of exp(x - row_max), then
// y = (x - row_max) - log(sum). Only the sum is in fixed point.

#include "fixed_point.cuh"
#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <limits>

namespace fxpr {
namespace detail {

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

template <typename FxpInt, typename IOFloat>
__global__ void log_softmax_kernel(
    const IOFloat* __restrict__ x,
    IOFloat* __restrict__ y,
    int64_t stride_x,
    int64_t stride_y,
    int N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const IOFloat* x_row = x + row * stride_x;
  IOFloat* y_row = y + row * stride_y;

  float partial_max = -INFINITY;
  for (int i = tid; i < N; i += nthreads) {
    partial_max = fmaxf(partial_max, static_cast<float>(x_row[i]));
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

  FxpInt partial_sum = 0;
  for (int i = tid; i < N; i += nthreads) {
    const float xi = static_cast<float>(x_row[i]);
    const float ei = expf(xi - row_max);
    partial_sum += float_to_fixed<FxpInt>(ei);
  }
  partial_sum = warp_reduce_sum<FxpInt>(partial_sum);

  if (lane == 0) warp_sum[warp_id] = partial_sum;
  __syncthreads();

  if (warp_id == 0) {
    FxpInt v = lane < num_warps ? warp_sum[lane] : FxpInt(0);
    v = warp_reduce_sum<FxpInt>(v);
    if (lane == 0) {
      const float sum_f = fixed_to_float<FxpInt>(v);
      log_sum_s = logf(sum_f);
    }
  }
  __syncthreads();

  const float log_sum = log_sum_s;

  for (int i = tid; i < N; i += nthreads) {
    const float xi = static_cast<float>(x_row[i]);
    y_row[i] = static_cast<IOFloat>((xi - row_max) - log_sum);
  }
}

template <typename FxpInt, typename IOFloat>
void launch_typed(
    const at::Tensor& x_2d,
    at::Tensor& y_2d) {
  const int rows = x_2d.size(0);
  const int N = x_2d.size(1);
  if (rows == 0 || N == 0) return;

  int block = 1;
  while (block < N && block < kMaxThreads) block <<= 1;
  if (block > kMaxThreads) block = kMaxThreads;

  auto stream = at::cuda::getCurrentCUDAStream();
  log_softmax_kernel<FxpInt, IOFloat><<<rows, block, 0, stream>>>(
      x_2d.data_ptr<IOFloat>(),
      y_2d.data_ptr<IOFloat>(),
      x_2d.stride(0),
      y_2d.stride(0),
      N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt>
void launch(
    const at::Tensor& x_2d,
    at::Tensor& y_2d) {
  switch (x_2d.scalar_type()) {
    case at::kFloat:
      launch_typed<FxpInt, float>(x_2d, y_2d);
      break;
    case at::kHalf:
      launch_typed<FxpInt, at::Half>(x_2d, y_2d);
      break;
    case at::kBFloat16:
      launch_typed<FxpInt, at::BFloat16>(x_2d, y_2d);
      break;
    default:
      TORCH_CHECK(false, "log_softmax: unsupported input dtype ",
                  x_2d.scalar_type());
  }
}

}  // namespace

at::Tensor log_softmax_fxp_run(
    at::Tensor x,
    int64_t int_bits) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_c = x.contiguous();
  auto x_2d = x_c.reshape({-1, x_c.size(-1)}).contiguous();
  auto y_2d = at::empty_like(x_2d);

  switch (int_bits) {
    case 16: launch<int16_t>(x_2d, y_2d); break;
    case 32: launch<int32_t>(x_2d, y_2d); break;
    case 64: launch<int64_t>(x_2d, y_2d); break;
  }

  return y_2d.view(x_c.sizes());
}

}  // namespace detail
}  // namespace fxpr
