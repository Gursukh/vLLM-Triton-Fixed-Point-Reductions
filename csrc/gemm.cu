// gemm_fxp_scalar       — fallback GEMM, deterministic per-element
//                         fixed-point quantisation before any add.
// gemm_fxp_int8_per_row — int8 GEMM with per-row scales; integer
//                         K-accumulator (split-invariant).
//
// Determinism (scalar path): one CTA per (M_block, N_block) tile.
// Each thread owns one output cell. The K-loop multiplies a[m,k] *
// b[k,n] in fp32, immediately quantises to int, and accumulates in
// integer. Because integer addition is associative and the
// per-element product is computed identically regardless of how the
// grid is shaped, the output is bit-identical for any block tiling.
//
// Determinism (int8 path): a_int8 / b_int8 are pre-quantised against
// per-row / per-col scales that are computed before the kernel sees
// its K range (see csrc/scales.cu). The K-accumulator is int32 over
// int8*int8 partial products — exact and associative — so any
// partition of K (block size, KV split, grid shape) yields the same
// integer accumulator and therefore the same fp32 output.

#include "fixed_point.cuh"
#include "ptx_mma.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace fxpr {

namespace {

constexpr int kBlockM = 16;
constexpr int kBlockN = 16;

template <typename FxpInt>
__global__ void gemm_fxp_scalar_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int frac_bits) {
  const int m = blockIdx.y * kBlockM + threadIdx.y;
  const int n = blockIdx.x * kBlockN + threadIdx.x;
  if (m >= M || n >= N) return;

  FxpInt acc = 0;
  for (int k = 0; k < K; ++k) {
    const float av = a[m * stride_am + k * stride_ak];
    const float bv = b[k * stride_bk + n * stride_bn];
    // Single fp32 multiply; --fmad=false stops nvcc from fusing this
    // with the implicit cast-to-int that follows.
    const float prod = __fmul_rn(av, bv);
    acc += float_to_fixed<FxpInt>(prod, frac_bits);
  }
  c[m * stride_cm + n * stride_cn] = fixed_to_float<FxpInt>(acc, frac_bits);
}

template <typename FxpInt>
void launch_scalar(
    const at::Tensor& a, const at::Tensor& b, at::Tensor& c,
    int frac_bits) {
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  if (M == 0 || N == 0) return;

  dim3 block(kBlockN, kBlockM);
  dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);

  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_fxp_scalar_kernel<FxpInt><<<grid, block, 0, stream>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      M, N, K,
      a.stride(0), a.stride(1),
      b.stride(0), b.stride(1),
      c.stride(0), c.stride(1),
      frac_bits);
}

}  // namespace

torch::Tensor gemm_fxp_op(
    torch::Tensor a,
    torch::Tensor b,
    int64_t frac_bits,
    int64_t int_bits) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "gemm_fxp: inputs must be CUDA");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "gemm_fxp: a must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "gemm_fxp: b must be float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm_fxp: 2D inputs required");
  TORCH_CHECK(a.size(1) == b.size(0),
              "gemm_fxp: shape mismatch ", a.sizes(), " @ ", b.sizes());
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");

  const c10::cuda::CUDAGuard device_guard(a.device());
  const int M = a.size(0);
  const int N = b.size(1);

  auto c = at::empty({M, N}, a.options());

  switch (int_bits) {
    case 16: launch_scalar<int16_t>(a, b, c, frac_bits); break;
    case 32: launch_scalar<int32_t>(a, b, c, frac_bits); break;
    case 64: launch_scalar<int64_t>(a, b, c, frac_bits); break;
  }
  return c;
}

// ---------- int8 per-row GEMM ----------

namespace {

constexpr int kI8BlockM = 16;
constexpr int kI8BlockN = 16;

template <typename FxpInt>
__global__ void gemm_fxp_int8_kernel(
    const int8_t* __restrict__ a,         // (M, K) int8
    const at::Half* __restrict__ a_scale, // (M,) fp16
    const int8_t* __restrict__ b,         // (K, N) int8
    const at::Half* __restrict__ b_scale, // (N,) fp16
    float* __restrict__ c,                // (M, N) fp32
    int M, int N, int K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int frac_bits) {
  const int m = blockIdx.y * kI8BlockM + threadIdx.y;
  const int n = blockIdx.x * kI8BlockN + threadIdx.x;
  if (m >= M || n >= N) return;

  // Integer K-accumulator: 4 product magnitudes ~127^2 ~ 16k, K up to
  // a few thousand stays well inside int32. Use int32 unconditionally;
  // the FxpInt template parameter only governs the *output* fxp grid.
  int32_t acc = 0;
  for (int k = 0; k < K; ++k) {
    const int32_t av = static_cast<int32_t>(a[m * stride_am + k * stride_ak]);
    const int32_t bv = static_cast<int32_t>(b[k * stride_bk + n * stride_bn]);
    acc += av * bv;
  }

  // Single fp32 reconstruction per output cell. Two multiplies, both
  // single-rounded (fmad disabled).
  const float as = static_cast<float>(__half2float(
      reinterpret_cast<const __half&>(a_scale[m])));
  const float bs = static_cast<float>(__half2float(
      reinterpret_cast<const __half&>(b_scale[n])));
  const float partial_fp = __fmul_rn(__fmul_rn(static_cast<float>(acc), as), bs);

  // Quantise->dequantise to land on the fxp grid (matches the rest of
  // the project's invariant: every reported fp32 result is a value
  // representable in Q<int_bits-frac_bits>.<frac_bits>).
  const FxpInt out_int = float_to_fixed<FxpInt>(partial_fp, frac_bits);
  c[m * stride_cm + n * stride_cn] = fixed_to_float<FxpInt>(out_int, frac_bits);
}

template <typename FxpInt>
void launch_int8(
    const at::Tensor& a, const at::Tensor& a_scale,
    const at::Tensor& b, const at::Tensor& b_scale,
    at::Tensor& c, int frac_bits) {
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  if (M == 0 || N == 0) return;

  dim3 block(kI8BlockN, kI8BlockM);
  dim3 grid((N + kI8BlockN - 1) / kI8BlockN, (M + kI8BlockM - 1) / kI8BlockM);

  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_fxp_int8_kernel<FxpInt><<<grid, block, 0, stream>>>(
      a.data_ptr<int8_t>(),
      reinterpret_cast<const at::Half*>(a_scale.data_ptr<at::Half>()),
      b.data_ptr<int8_t>(),
      reinterpret_cast<const at::Half*>(b_scale.data_ptr<at::Half>()),
      c.data_ptr<float>(),
      M, N, K,
      a.stride(0), a.stride(1),
      b.stride(0), b.stride(1),
      c.stride(0), c.stride(1),
      frac_bits);
}

}  // namespace

torch::Tensor gemm_fxp_int8_op(
    torch::Tensor a_int8,
    torch::Tensor a_scale,
    torch::Tensor b_int8,
    torch::Tensor b_scale,
    int64_t frac_bits,
    int64_t int_bits) {
  TORCH_CHECK(a_int8.is_cuda() && b_int8.is_cuda(),
              "gemm_fxp_int8: inputs must be CUDA");
  TORCH_CHECK(a_int8.scalar_type() == at::kChar,
              "gemm_fxp_int8: a must be int8");
  TORCH_CHECK(b_int8.scalar_type() == at::kChar,
              "gemm_fxp_int8: b must be int8");
  TORCH_CHECK(a_scale.scalar_type() == at::kHalf,
              "gemm_fxp_int8: a_scale must be float16");
  TORCH_CHECK(b_scale.scalar_type() == at::kHalf,
              "gemm_fxp_int8: b_scale must be float16");
  TORCH_CHECK(a_int8.dim() == 2 && b_int8.dim() == 2,
              "gemm_fxp_int8: 2D inputs required");
  TORCH_CHECK(a_int8.size(1) == b_int8.size(0),
              "gemm_fxp_int8: shape mismatch ", a_int8.sizes(),
              " @ ", b_int8.sizes());
  TORCH_CHECK(a_scale.numel() == a_int8.size(0),
              "a_scale must have shape (M,)");
  TORCH_CHECK(b_scale.numel() == b_int8.size(1),
              "b_scale must have shape (N,)");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");

  const c10::cuda::CUDAGuard device_guard(a_int8.device());
  const int M = a_int8.size(0);
  const int N = b_int8.size(1);

  auto c = at::empty({M, N}, a_int8.options().dtype(at::kFloat));

  switch (int_bits) {
    case 16: launch_int8<int16_t>(a_int8, a_scale, b_int8, b_scale, c, frac_bits); break;
    case 32: launch_int8<int32_t>(a_int8, a_scale, b_int8, b_scale, c, frac_bits); break;
    case 64: launch_int8<int64_t>(a_int8, a_scale, b_int8, b_scale, c, frac_bits); break;
  }
  return c;
}

}  // namespace fxpr
