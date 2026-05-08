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

// Slice [0, total) into num_splits chunks (matches attention's compute_chunk).
__device__ __forceinline__ void compute_chunk(
    int total, int num_splits, int split,
    int* out_start, int* out_end) {
  const int64_t lo = (int64_t)split * total;
  const int64_t hi = (int64_t)(split + 1) * total;
  *out_start = (int)(lo / num_splits);
  *out_end = (int)(hi / num_splits);
}

template <typename FxpInt>
__device__ __forceinline__ FxpInt block_reduce_sum_fxp(
    FxpInt v, FxpInt* shmem) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  v = warp_reduce_sum<FxpInt>(v);
  if (lane == 0) shmem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    FxpInt x = lane < num_warps ? shmem[lane] : FxpInt(0);
    x = warp_reduce_sum<FxpInt>(x);
    if (lane == 0) shmem[0] = x;
  }
  __syncthreads();
  return shmem[0];
}

__device__ __forceinline__ float block_reduce_max(float v, float* shmem) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  v = warp_reduce_max(v);
  if (lane == 0) shmem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float x = lane < num_warps ? shmem[lane] : -INFINITY;
    x = warp_reduce_max(x);
    if (lane == 0) shmem[0] = x;
  }
  __syncthreads();
  return shmem[0];
}

template <typename FxpInt, typename IOFloat, int kFracBits>
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
    partial_sum += float_to_fixed<FxpInt, kFracBits>(ei);
  }
  partial_sum = warp_reduce_sum<FxpInt>(partial_sum);

  if (lane == 0) warp_sum[warp_id] = partial_sum;
  __syncthreads();

  if (warp_id == 0) {
    FxpInt v = lane < num_warps ? warp_sum[lane] : FxpInt(0);
    v = warp_reduce_sum<FxpInt>(v);
    if (lane == 0) {
      const float sum_f = fixed_to_float<FxpInt, kFracBits>(v);
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

// Split-V variant for small batch / large N (e.g. decode with rows=1). All
// splits share global_max, and we sum partials in split-index order, so the
// result is bit-exact vs the single-CTA path.

template <typename IOFloat>
__global__ void log_softmax_split_max_kernel(
    const IOFloat* __restrict__ x,
    float* __restrict__ partial_max,
    int64_t stride_x, int N, int num_splits) {
  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int tid = threadIdx.x;

  int chunk_start, chunk_end;
  compute_chunk(N, num_splits, split, &chunk_start, &chunk_end);

  const IOFloat* x_row = x + row * stride_x;
  float local = -INFINITY;
  for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
    local = fmaxf(local, static_cast<float>(x_row[i]));
  }

  __shared__ float shm[kMaxThreads / kWarpSize];
  const float row_split_max = block_reduce_max(local, shm);
  if (tid == 0) {
    partial_max[row * num_splits + split] = row_split_max;
  }
}

template <typename FxpInt, typename IOFloat, int kFracBits>
__global__ void log_softmax_split_sum_kernel(
    const IOFloat* __restrict__ x,
    const float* __restrict__ partial_max,
    FxpInt* __restrict__ partial_sum,
    int64_t stride_x, int N, int num_splits) {
  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float s_global_max;
  if (tid == 0) {
    float m = -INFINITY;
    const float* pm = partial_max + row * num_splits;
    for (int s = 0; s < num_splits; ++s) m = fmaxf(m, pm[s]);
    s_global_max = m;
  }
  __syncthreads();
  const float global_max = s_global_max;

  int chunk_start, chunk_end;
  compute_chunk(N, num_splits, split, &chunk_start, &chunk_end);

  const IOFloat* x_row = x + row * stride_x;
  FxpInt local = 0;
  for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
    const float xi = static_cast<float>(x_row[i]);
    local += float_to_fixed<FxpInt, kFracBits>(expf(xi - global_max));
  }

  __shared__ FxpInt shm[kMaxThreads / kWarpSize];
  const FxpInt total_split = block_reduce_sum_fxp<FxpInt>(local, shm);
  if (tid == 0) {
    partial_sum[row * num_splits + split] = total_split;
  }
}

template <typename FxpInt, typename IOFloat, int kFracBits>
__global__ void log_softmax_split_writeback_kernel(
    const IOFloat* __restrict__ x,
    IOFloat* __restrict__ y,
    const float* __restrict__ partial_max,
    const FxpInt* __restrict__ partial_sum,
    int64_t stride_x, int64_t stride_y,
    int N, int num_splits) {
  const int row = blockIdx.x;
  const int split = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float s_global_max;
  __shared__ float s_log_sum;
  if (tid == 0) {
    float m = -INFINITY;
    const float* pm = partial_max + row * num_splits;
    for (int s = 0; s < num_splits; ++s) m = fmaxf(m, pm[s]);
    FxpInt total = 0;
    const FxpInt* ps = partial_sum + row * num_splits;
    for (int s = 0; s < num_splits; ++s) total += ps[s];
    s_global_max = m;
    s_log_sum = logf(fixed_to_float<FxpInt, kFracBits>(total));
  }
  __syncthreads();
  const float global_max = s_global_max;
  const float log_sum = s_log_sum;

  int chunk_start, chunk_end;
  compute_chunk(N, num_splits, split, &chunk_start, &chunk_end);

  const IOFloat* x_row = x + row * stride_x;
  IOFloat* y_row = y + row * stride_y;
  for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
    const float xi = static_cast<float>(x_row[i]);
    y_row[i] = static_cast<IOFloat>((xi - global_max) - log_sum);
  }
}

template <typename FxpInt, typename IOFloat, int kFracBits>
void launch_typed(
    const at::Tensor& x_2d,
    at::Tensor& y_2d) {
  const int rows = x_2d.size(0);
  const int N = x_2d.size(1);
  if (rows == 0 || N == 0) return;

  auto stream = at::cuda::getCurrentCUDAStream();

  // Split V only when rows wouldn't fill the SMs and N is big enough to pay
  // for the extra launches. Capped at 16 splits.
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int sm_count = props->multiProcessorCount;
  int num_splits = 1;
  if (rows < sm_count && N >= 4096) {
    num_splits = std::min(16, std::max(1, sm_count / std::max(1, rows)));
    if (num_splits < 2) num_splits = 1;
  }

  if (num_splits == 1) {
    int block = 1;
    while (block < N && block < kMaxThreads) block <<= 1;
    if (block > kMaxThreads) block = kMaxThreads;

    log_softmax_kernel<FxpInt, IOFloat, kFracBits><<<rows, block, 0, stream>>>(
        x_2d.data_ptr<IOFloat>(),
        y_2d.data_ptr<IOFloat>(),
        x_2d.stride(0),
        y_2d.stride(0),
        N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  const int max_chunk = (N + num_splits - 1) / num_splits;
  int block = 1;
  while (block < max_chunk && block < kMaxThreads) block <<= 1;
  if (block > kMaxThreads) block = kMaxThreads;
  if (block < kWarpSize) block = kWarpSize;

  const auto fxp_dtype = c10::CppTypeToScalarType<FxpInt>::value;
  auto partial_max = at::full(
      {rows, num_splits},
      -std::numeric_limits<float>::infinity(),
      x_2d.options().dtype(at::kFloat));
  auto partial_sum = at::zeros(
      {rows, num_splits}, x_2d.options().dtype(fxp_dtype));

  dim3 grid(rows, num_splits);

  log_softmax_split_max_kernel<IOFloat><<<grid, block, 0, stream>>>(
      x_2d.data_ptr<IOFloat>(),
      partial_max.data_ptr<float>(),
      x_2d.stride(0), N, num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  log_softmax_split_sum_kernel<FxpInt, IOFloat, kFracBits>
      <<<grid, block, 0, stream>>>(
      x_2d.data_ptr<IOFloat>(),
      partial_max.data_ptr<float>(),
      partial_sum.template data_ptr<FxpInt>(),
      x_2d.stride(0), N, num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  log_softmax_split_writeback_kernel<FxpInt, IOFloat, kFracBits>
      <<<grid, block, 0, stream>>>(
      x_2d.data_ptr<IOFloat>(),
      y_2d.data_ptr<IOFloat>(),
      partial_max.data_ptr<float>(),
      partial_sum.template data_ptr<FxpInt>(),
      x_2d.stride(0), y_2d.stride(0), N, num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt, int kFracBits>
void launch(
    const at::Tensor& x_2d,
    at::Tensor& y_2d) {
  switch (x_2d.scalar_type()) {
    case at::kFloat:
      launch_typed<FxpInt, float, kFracBits>(x_2d, y_2d);
      break;
    case at::kHalf:
      launch_typed<FxpInt, at::Half, kFracBits>(x_2d, y_2d);
      break;
    case at::kBFloat16:
      launch_typed<FxpInt, at::BFloat16, kFracBits>(x_2d, y_2d);
      break;
    default:
      TORCH_CHECK(false, "log_softmax: unsupported input dtype ",
                  x_2d.scalar_type());
  }
}

template <int kFracBits>
void log_softmax_dispatch_int(const at::Tensor& x_2d, at::Tensor& y_2d,
                              int64_t int_bits) {
  switch (int_bits) {
    case 16: launch<int16_t, kFracBits>(x_2d, y_2d); break;
    case 32: launch<int32_t, kFracBits>(x_2d, y_2d); break;
    case 64: launch<int64_t, kFracBits>(x_2d, y_2d); break;
  }
}

}  // namespace

at::Tensor log_softmax_fxp_run(
    at::Tensor x,
    int64_t int_bits,
    int64_t fxp_frac_bits) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_c = x.contiguous();
  auto x_2d = x_c.reshape({-1, x_c.size(-1)}).contiguous();
  auto y_2d = at::empty_like(x_2d);

  switch (fxp_frac_bits) {
    case 8:  log_softmax_dispatch_int<8>(x_2d, y_2d, int_bits);  break;
    case 16: log_softmax_dispatch_int<16>(x_2d, y_2d, int_bits); break;
    case 32: log_softmax_dispatch_int<32>(x_2d, y_2d, int_bits); break;
  }

  return y_2d.view(x_c.sizes());
}

}  // namespace detail
}  // namespace fxpr
