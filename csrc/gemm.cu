// gemm_fxp, deterministic fp32 GEMM with per-product fixed-point
// quantisation before each accumulator add.
//
// Determinism: each c[m,n] sees products in k = 0..K-1 order. Tile
// shape (BM x BN, TM x TN, BK) only affects parallel layout, not the
// per-cell K order. float4 staging changes load instructions, not math.

#include "fixed_point.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace fxpr {

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kThreadsX = kBN / kTN;       // 16
constexpr int kThreadsY = kBM / kTM;       // 16
constexpr int kThreads  = kThreadsX * kThreadsY;  // 256

static_assert((kBM * kBK) % kThreads == 0, "A staging must divide evenly");
static_assert((kBK * kBN) % kThreads == 0, "B staging must divide evenly");
static_assert((kBM * kBK) % (kThreads * 4) == 0,
              "float4 A staging requires BM*BK divisible by 4*threads");
static_assert((kBK * kBN) % (kThreads * 4) == 0,
              "float4 B staging requires BK*BN divisible by 4*threads");
static_assert(kBK % 4 == 0,
              "float4 A staging requires BK divisible by 4 so a thread's "
              "4-element strip lies within a single tile row");
static_assert(kBN % 4 == 0,
              "float4 B staging requires BN divisible by 4");

template <typename FxpInt, bool kVec>
__global__ void gemm_fxp_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int frac_bits) {
  __shared__ float A_smem[kBM][kBK];
  __shared__ float B_smem[kBK][kBN];

  const int tid = threadIdx.y * kThreadsX + threadIdx.x;
  const int m_block = blockIdx.y * kBM;
  const int n_block = blockIdx.x * kBN;

  // Per-thread integer accumulators for a TM x TN tile.
  FxpInt acc[kTM][kTN];
  #pragma unroll
  for (int i = 0; i < kTM; ++i) {
    #pragma unroll
    for (int j = 0; j < kTN; ++j) {
      acc[i][j] = 0;
    }
  }

  // Per-thread A/B staging counts. With kBM*kBK == kThreads*kAStage, each
  // thread loads kAStage elements of A; same for B. For the float4 path
  // each thread loads one float4 = 4 contiguous elements.
  constexpr int kAStage = (kBM * kBK) / kThreads;
  constexpr int kBStage = (kBK * kBN) / kThreads;

  for (int k_block = 0; k_block < K; k_block += kBK) {
    if constexpr (kVec) {
      // float4 stage of A: one float4 per thread = 4 contiguous elements
      // along the K dimension within a single tile row. The dispatch in
      // launch_tiled has already verified stride_ak == 1, stride_am % 4 == 0,
      // and 16-byte base alignment, so the float4 read is well-defined.
      const int idx_base = tid * 4;
      const int sm_row   = idx_base / kBK;
      const int sm_col   = idx_base % kBK;          // multiple of 4 by construction
      const int g_row    = m_block + sm_row;
      const int g_col    = k_block + sm_col;
      float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (g_row < M && g_col < K) {
        v4 = *reinterpret_cast<const float4*>(&a[g_row * stride_am + g_col]);
        // K-tail mask: zero out any lanes that fall past K. This makes
        // kVec correct for K not divisible by 4 without restricting the
        // dispatch to K%4==0.
        if (g_col + 1 >= K) v4.y = 0.0f;
        if (g_col + 2 >= K) v4.z = 0.0f;
        if (g_col + 3 >= K) v4.w = 0.0f;
      }
      *reinterpret_cast<float4*>(&A_smem[sm_row][sm_col]) = v4;
    } else {
      // Stage A[m_block:m_block+BM, k_block:k_block+BK] into smem. Threads
      // step linearly through the (BM*BK) elements; with row-major A and
      // contiguous K stride this gives 32-thread coalesced loads.
      #pragma unroll
      for (int e = 0; e < kAStage; ++e) {
        const int idx    = e * kThreads + tid;
        const int sm_row = idx / kBK;
        const int sm_col = idx % kBK;
        const int g_row  = m_block + sm_row;
        const int g_col  = k_block + sm_col;
        float v = 0.0f;
        if (g_row < M && g_col < K) {
          v = a[g_row * stride_am + g_col * stride_ak];
        }
        A_smem[sm_row][sm_col] = v;
      }
    }

    if constexpr (kVec) {
      // float4 stage of B: one float4 per thread along the N dimension
      // within a single tile row. Dispatch guarantees stride_bn == 1 and
      // stride_bk % 4 == 0.
      const int idx_base = tid * 4;
      const int sm_row   = idx_base / kBN;
      const int sm_col   = idx_base % kBN;
      const int g_row    = k_block + sm_row;
      const int g_col    = n_block + sm_col;
      float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (g_row < K && g_col < N) {
        v4 = *reinterpret_cast<const float4*>(&b[g_row * stride_bk + g_col]);
        if (g_col + 1 >= N) v4.y = 0.0f;
        if (g_col + 2 >= N) v4.z = 0.0f;
        if (g_col + 3 >= N) v4.w = 0.0f;
      }
      *reinterpret_cast<float4*>(&B_smem[sm_row][sm_col]) = v4;
    } else {
      // Stage B[k_block:k_block+BK, n_block:n_block+BN] into smem.
      #pragma unroll
      for (int e = 0; e < kBStage; ++e) {
        const int idx    = e * kThreads + tid;
        const int sm_row = idx / kBN;
        const int sm_col = idx % kBN;
        const int g_row  = k_block + sm_row;
        const int g_col  = n_block + sm_col;
        float v = 0.0f;
        if (g_row < K && g_col < N) {
          v = b[g_row * stride_bk + g_col * stride_bn];
        }
        B_smem[sm_row][sm_col] = v;
      }
    }
    __syncthreads();

    // Inner-product over the BK stripe. Per-cell K-order is preserved
    // by the kk and k_block loops, matching the scalar reference.
    #pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      float a_reg[kTM];
      #pragma unroll
      for (int i = 0; i < kTM; ++i) {
        a_reg[i] = A_smem[threadIdx.y * kTM + i][kk];
      }
      float b_reg[kTN];
      #pragma unroll
      for (int j = 0; j < kTN; ++j) {
        b_reg[j] = B_smem[kk][threadIdx.x * kTN + j];
      }
      #pragma unroll
      for (int i = 0; i < kTM; ++i) {
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
          // Single discrete multiply -> quantise -> integer add.
          const float prod = __fmul_rn(a_reg[i], b_reg[j]);
          acc[i][j] += float_to_fixed<FxpInt>(prod, frac_bits);
        }
      }
    }
    __syncthreads();
  }

  const int m_thread = m_block + threadIdx.y * kTM;
  const int n_thread = n_block + threadIdx.x * kTN;
  #pragma unroll
  for (int i = 0; i < kTM; ++i) {
    const int gm = m_thread + i;
    if (gm >= M) continue;
    #pragma unroll
    for (int j = 0; j < kTN; ++j) {
      const int gn = n_thread + j;
      if (gn >= N) continue;
      c[gm * stride_cm + gn * stride_cn] =
          fixed_to_float<FxpInt>(acc[i][j], frac_bits);
    }
  }
}

template <typename FxpInt>
void launch_tiled(
    const at::Tensor& a, const at::Tensor& b, at::Tensor& c,
    int frac_bits) {
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  if (M == 0 || N == 0) return;

  dim3 block(kThreadsX, kThreadsY);
  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);

  // float4 staging requires unit inner stride and 16-byte-aligned rows.
  // PyTorch's allocator already gives ≥256-byte base alignment.
  const auto a_ptr = reinterpret_cast<uintptr_t>(a.data_ptr<float>());
  const auto b_ptr = reinterpret_cast<uintptr_t>(b.data_ptr<float>());
  const bool can_vec =
      a.stride(1) == 1 && b.stride(1) == 1 &&
      a.stride(0) % 4 == 0 && b.stride(0) % 4 == 0 &&
      (a_ptr % 16 == 0) && (b_ptr % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  if (can_vec) {
    gemm_fxp_tiled_kernel<FxpInt, /*kVec=*/true><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        frac_bits);
  } else {
    gemm_fxp_tiled_kernel<FxpInt, /*kVec=*/false><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        frac_bits);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    case 16: launch_tiled<int16_t>(a, b, c, frac_bits); break;
    case 32: launch_tiled<int32_t>(a, b, c, frac_bits); break;
    case 64: launch_tiled<int64_t>(a, b, c, frac_bits); break;
  }
  return c;
}

}  // namespace fxpr
