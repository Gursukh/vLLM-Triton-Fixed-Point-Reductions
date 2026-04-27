// Per-row max-abs scale reductions.
//
// One CTA per row: each thread reduces a strided slice via fp32
// max-abs (max is order-invariant in fp32), warps reduce, then a
// single warp finishes the reduction across warp partials. The result
// is fmax(max_abs / 127, eps) -> fp16 stored as a (M,) tensor.
//
// Determinism: max-abs over fp32 is order-invariant; the reduction
// tree shape doesn't change the result. Per-row scaling means the
// scale is fixed *before* any K-partition kernel sees its slice, so
// the int8 MMA path stays split-invariant.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace fxpr {

namespace {

constexpr int kMaxThreads = 1024;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  return v;
}

template <typename FloatT>
__global__ void per_row_scale_kernel(
    const FloatT* __restrict__ x,
    __half* __restrict__ scale_out,
    int64_t stride_x,
    int K,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const FloatT* x_row = x + row * stride_x;

  float partial = 0.0f;
  for (int i = tid; i < K; i += nthreads) {
    const float xi = static_cast<float>(x_row[i]);
    partial = fmaxf(partial, fabsf(xi));
  }
  partial = warp_reduce_max(partial);

  __shared__ float warp_max[kMaxThreads / kWarpSize];
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const int num_warps = (nthreads + kWarpSize - 1) / kWarpSize;

  if (lane == 0) warp_max[warp_id] = partial;
  __syncthreads();

  if (warp_id == 0) {
    float v = lane < num_warps ? warp_max[lane] : 0.0f;
    v = warp_reduce_max(v);
    if (lane == 0) {
      // Scale = max_abs / 127, with an eps floor so all-zero rows
      // don't divide by zero on the int8 cast.
      const float s = fmaxf(v / 127.0f, eps);
      scale_out[row] = __float2half(s);
    }
  }
}

template <typename FloatT>
void launch(const at::Tensor& x_2d, at::Tensor& scale, double eps) {
  const int rows = x_2d.size(0);
  const int K = x_2d.size(1);
  if (rows == 0) return;

  int block = 1;
  while (block < K && block < kMaxThreads) block <<= 1;
  if (block > kMaxThreads) block = kMaxThreads;

  auto stream = at::cuda::getCurrentCUDAStream();
  per_row_scale_kernel<FloatT><<<rows, block, 0, stream>>>(
      x_2d.data_ptr<FloatT>(),
      reinterpret_cast<__half*>(scale.data_ptr<at::Half>()),
      x_2d.stride(0),
      K,
      static_cast<float>(eps));
}

}  // namespace

torch::Tensor compute_per_row_scale_op(
    torch::Tensor x,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "compute_per_row_scale: x must be CUDA");
  TORCH_CHECK(x.dim() >= 1, "compute_per_row_scale: x must have at least 1 dim");

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_2d = x.reshape({-1, x.size(-1)}).contiguous();
  const int rows = x_2d.size(0);

  auto scale = at::empty({rows}, x_2d.options().dtype(at::kHalf));

  switch (x_2d.scalar_type()) {
    case at::kFloat:  launch<float>(x_2d, scale, eps); break;
    case at::kHalf:   launch<at::Half>(x_2d, scale, eps); break;
    case at::kBFloat16: launch<at::BFloat16>(x_2d, scale, eps); break;
    default:
      TORCH_CHECK(false, "compute_per_row_scale: unsupported dtype ",
                  x_2d.scalar_type());
  }

  // Reshape scale to match all but the last dim of x.
  auto out_shape = x.sizes().vec();
  out_shape.pop_back();
  return scale.view(out_shape);
}

}  // namespace fxpr
