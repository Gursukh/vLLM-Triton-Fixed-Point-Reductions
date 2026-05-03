// Tensor-core GEMM with fixed-point inter-tile accumulation.
// Each K-tile mma result is converted to fxp once and added into a
// per-thread integer accumulator, so the K-tile reduction is integer.
//
// kBK: 32 for half/bf16, 16 for float (fits 48KB smem with double buffer).
// Arch floor: fp16 needs sm_75+, bf16/fp32 need sm_80+. Gated in ops.cpp.

#include "fixed_point.cuh"
#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <type_traits>

namespace fxpr
{
  namespace detail
  {

    namespace wmma = nvcuda::wmma;

    namespace
    {

      // ----- Block / warp / fragment layout -------------------------------------

      constexpr int kBM = 128;
      constexpr int kBN = 128;
      constexpr int kWarpsM = 4;
      constexpr int kWarpsN = 2;
      constexpr int kWarps = kWarpsM * kWarpsN; // 8
      constexpr int kThreads = kWarps * 32;     // 256

      constexpr int kWarpM = kBM / kWarpsM; // 32
      constexpr int kWarpN = kBN / kWarpsN; // 64

      // K-tile width per dtype. Picked to fit double-buffered smem in 48KB.
      template <typename T>
      struct BKConfig
      {
        static constexpr int value = 16;
      };
      template <>
      struct BKConfig<__half>
      {
        static constexpr int value = 32;
      };
      template <>
      struct BKConfig<__nv_bfloat16>
      {
        static constexpr int value = 32;
      };

      // Fragment shape: m16n16k16 for fp16/bf16, m16n16k8 for TF32.
      template <typename T>
      struct FragShape
      {
        static constexpr int M = 16, N = 16, K = 16;
      };
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
      template <>
      struct FragShape<float>
      {
        static constexpr int M = 16, N = 16, K = 8;
      };
#endif

      // fp16 in -> fp16 acc; bf16/TF32 in -> fp32 acc.
      template <typename T>
      struct AccumT
      {
        using type = float;
      };
      template <>
      struct AccumT<__half>
      {
        using type = __half;
      };

      // fp32 inputs go through the tf32 mma precision tag (sm_80+ only).
      template <typename T>
      struct MmaInT
      {
        using type = T;
      };
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
      template <>
      struct MmaInT<float>
      {
        using type = wmma::precision::tf32;
      };
#endif

      constexpr int kFragsM = kWarpM / 16; // 2
      constexpr int kFragsN = kWarpN / 16; // 4

      // 8 elements per thread for the m16n16 fragments.
      constexpr int kFragElems = 8;

      // ----- Per-dtype fixed-point conversion -----------------------------------

      template <typename FxpInt, typename Acc>
      __device__ __forceinline__ FxpInt acc_to_fixed(Acc x)
      {
        if constexpr (std::is_same_v<Acc, __half>)
        {
          return half_to_fixed<FxpInt>(x);
        }
        else
        {
          return float_to_fixed<FxpInt>(x);
        }
      }

      template <typename Acc>
      __device__ __forceinline__ Acc float_to_acc(float v)
      {
        if constexpr (std::is_same_v<Acc, __half>)
        {
          return __float2half(v);
        }
        else
        {
          return v;
        }
      }

      template <typename Acc>
      __device__ __forceinline__ float acc_to_float(Acc v)
      {
        if constexpr (std::is_same_v<Acc, __half>)
        {
          return __half2float(v);
        }
        else
        {
          return v;
        }
      }

      // 128-bit vectorised load when the layout/alignment allows it, scalar
      // fallback otherwise. Tails zero-fill so non-multiple K just works.

      template <typename T, int BK>
      __device__ __forceinline__ void load_a_to_smem(
          T (*A_smem)[BK],
          const T *__restrict__ a, int M, int K,
          int64_t stride_am, int64_t stride_ak,
          int m_block, int k_block, int tid)
      {
        constexpr int kBytesPerVec = 16;
        constexpr int kElemsPerVec = kBytesPerVec / sizeof(T);
        constexpr int kVecsPerThread =
            (kBM * BK) / (kThreads * kElemsPerVec);
        static_assert(BK % kElemsPerVec == 0, "BK must be a multiple of vec width");
        static_assert(kVecsPerThread * kThreads * kElemsPerVec == kBM * BK,
                      "load layout mismatch");

        const bool fast = (stride_ak == 1) && ((stride_am * (int64_t)sizeof(T)) % kBytesPerVec == 0) && ((reinterpret_cast<uintptr_t>(a) % kBytesPerVec) == 0);

#pragma unroll
        for (int e = 0; e < kVecsPerThread; ++e)
        {
          const int v = e * kThreads + tid;
          const int sm_row = (v * kElemsPerVec) / BK;
          const int sm_col = (v * kElemsPerVec) % BK;
          const int g_row = m_block + sm_row;
          const int g_col = k_block + sm_col;

          if (fast && g_row < M && g_col + kElemsPerVec <= K)
          {
            const int4 vec = *reinterpret_cast<const int4 *>(
                &a[g_row * stride_am + g_col]);
            *reinterpret_cast<int4 *>(&A_smem[sm_row][sm_col]) = vec;
          }
          else
          {
#pragma unroll
            for (int j = 0; j < kElemsPerVec; ++j)
            {
              T val = T(0);
              if (g_row < M && (g_col + j) < K)
              {
                val = a[g_row * stride_am + (g_col + j) * stride_ak];
              }
              A_smem[sm_row][sm_col + j] = val;
            }
          }
        }
      }

      template <typename T, int BK>
      __device__ __forceinline__ void load_b_to_smem(
          T (*B_smem)[kBN],
          const T *__restrict__ b, int K, int N,
          int64_t stride_bk, int64_t stride_bn,
          int k_block, int n_block, int tid)
      {
        constexpr int kBytesPerVec = 16;
        constexpr int kElemsPerVec = kBytesPerVec / sizeof(T);
        constexpr int kVecsPerThread =
            (BK * kBN) / (kThreads * kElemsPerVec);
        static_assert(kBN % kElemsPerVec == 0, "kBN must be a multiple of vec width");
        static_assert(kVecsPerThread * kThreads * kElemsPerVec == BK * kBN,
                      "load layout mismatch");

        const bool fast = (stride_bn == 1) && ((stride_bk * (int64_t)sizeof(T)) % kBytesPerVec == 0) && ((reinterpret_cast<uintptr_t>(b) % kBytesPerVec) == 0);

#pragma unroll
        for (int e = 0; e < kVecsPerThread; ++e)
        {
          const int v = e * kThreads + tid;
          const int sm_row = (v * kElemsPerVec) / kBN;
          const int sm_col = (v * kElemsPerVec) % kBN;
          const int g_row = k_block + sm_row;
          const int g_col = n_block + sm_col;

          if (fast && g_row < K && g_col + kElemsPerVec <= N)
          {
            const int4 vec = *reinterpret_cast<const int4 *>(
                &b[g_row * stride_bk + g_col]);
            *reinterpret_cast<int4 *>(&B_smem[sm_row][sm_col]) = vec;
          }
          else
          {
#pragma unroll
            for (int j = 0; j < kElemsPerVec; ++j)
            {
              T val = T(0);
              if (g_row < K && (g_col + j) < N)
              {
                val = b[g_row * stride_bk + (g_col + j) * stride_bn];
              }
              B_smem[sm_row][sm_col + j] = val;
            }
          }
        }
      }

      // Body is templated on (FxpInt, IOFloat). The sm_75 path discards the
      // bf16/TF32 branches via `if constexpr` so they're never instantiated.

      template <typename FxpInt, typename IOFloat>
      __device__ void gemm_body(
          const IOFloat *__restrict__ a,
          const IOFloat *__restrict__ b,
          const IOFloat *__restrict__ bias,
          IOFloat *__restrict__ c,
          int M, int N, int K,
          int64_t stride_am, int64_t stride_ak,
          int64_t stride_bk, int64_t stride_bn,
          int64_t stride_cm, int64_t stride_cn)
      {
        using Acc = typename AccumT<IOFloat>::type;
        using MmaIn = typename MmaInT<IOFloat>::type;
        constexpr int FM = FragShape<IOFloat>::M;     // 16
        constexpr int FN = FragShape<IOFloat>::N;     // 16
        constexpr int FK = FragShape<IOFloat>::K;     // 16 (fp16/bf16) or 8 (TF32)
        constexpr int kBK = BKConfig<IOFloat>::value; // 32 (half/bf16) or 16 (float)
        constexpr int K_ITERS = kBK / FK;

        // Double-buffered A/B smem so HBM loads overlap mma compute.
        __shared__ __align__(16) IOFloat A_smem[2][kBM][kBK];
        __shared__ __align__(16) IOFloat B_smem[2][kBK][kBN];
        __shared__ __align__(16) Acc C_buf[kWarps][FM * FN];

        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane = tid & 31;
        const int warp_m = warp_id / kWarpsN; // 0..3
        const int warp_n = warp_id % kWarpsN; // 0..1
        const int m_block = blockIdx.y * kBM;
        const int n_block = blockIdx.x * kBN;

        FxpInt acc_fxp[kFragsM][kFragsN][kFragElems];
#pragma unroll
        for (int wm = 0; wm < kFragsM; ++wm)
#pragma unroll
          for (int wn = 0; wn < kFragsN; ++wn)
#pragma unroll
            for (int i = 0; i < kFragElems; ++i)
              acc_fxp[wm][wn][i] = FxpInt(0);

        const int num_tiles = (K + kBK - 1) / kBK;

        if (num_tiles > 0)
        {
          load_a_to_smem<IOFloat, kBK>(A_smem[0], a, M, K, stride_am, stride_ak,
                                       m_block, 0, tid);
          load_b_to_smem<IOFloat, kBK>(B_smem[0], b, K, N, stride_bk, stride_bn,
                                       0, n_block, tid);
        }
        __syncthreads();

        for (int it = 0; it < num_tiles; ++it)
        {
          const int cur = it & 1;
          const int nxt = cur ^ 1;
          const int next_k_block = (it + 1) * kBK;

          // Prefetch tile it+1 into the inactive buffer.
          if (next_k_block < K)
          {
            load_a_to_smem<IOFloat, kBK>(A_smem[nxt], a, M, K, stride_am, stride_ak,
                                         m_block, next_k_block, tid);
            load_b_to_smem<IOFloat, kBK>(B_smem[nxt], b, K, N, stride_bk, stride_bn,
                                         next_k_block, n_block, tid);
          }

          wmma::fragment<wmma::matrix_a, FM, FN, FK, MmaIn, wmma::row_major> a_frag[kFragsM];
          wmma::fragment<wmma::matrix_b, FM, FN, FK, MmaIn, wmma::row_major> b_frag[kFragsN];
          wmma::fragment<wmma::accumulator, FM, FN, FK, Acc> d_frag[kFragsM][kFragsN];

#pragma unroll
          for (int wm = 0; wm < kFragsM; ++wm)
#pragma unroll
            for (int wn = 0; wn < kFragsN; ++wn)
              wmma::fill_fragment(d_frag[wm][wn], float_to_acc<Acc>(0.f));

#pragma unroll
          for (int kit = 0; kit < K_ITERS; ++kit)
          {
            const int k_off = kit * FK;

#pragma unroll
            for (int wm = 0; wm < kFragsM; ++wm)
            {
              const int row = warp_m * kWarpM + wm * FM;
              wmma::load_matrix_sync(a_frag[wm],
                                     reinterpret_cast<const IOFloat *>(&A_smem[cur][row][k_off]),
                                     kBK);
            }
#pragma unroll
            for (int wn = 0; wn < kFragsN; ++wn)
            {
              const int col = warp_n * kWarpN + wn * FN;
              wmma::load_matrix_sync(b_frag[wn],
                                     reinterpret_cast<const IOFloat *>(&B_smem[cur][k_off][col]),
                                     kBN);
            }

            // mma.sync.tf32 drops the low 13 bits in hardware.

#pragma unroll
            for (int wm = 0; wm < kFragsM; ++wm)
            {
#pragma unroll
              for (int wn = 0; wn < kFragsN; ++wn)
              {
                wmma::mma_sync(d_frag[wm][wn], a_frag[wm], b_frag[wn], d_frag[wm][wn]);
              }
            }
          }

// Tile boundary: fold the mma partial into the fxp accumulator.
#pragma unroll
          for (int wm = 0; wm < kFragsM; ++wm)
          {
#pragma unroll
            for (int wn = 0; wn < kFragsN; ++wn)
            {
#pragma unroll
              for (int i = 0; i < kFragElems; ++i)
              {
                acc_fxp[wm][wn][i] += acc_to_fixed<FxpInt, Acc>(d_frag[wm][wn].x[i]);
              }
            }
          }
          __syncthreads();
        }

        // Writeback via smem, with bias added on the fly.
#pragma unroll
        for (int wm = 0; wm < kFragsM; ++wm)
        {
#pragma unroll
          for (int wn = 0; wn < kFragsN; ++wn)
          {
            wmma::fragment<wmma::accumulator, FM, FN, FK, Acc> out_frag;
#pragma unroll
            for (int i = 0; i < kFragElems; ++i)
            {
              const float v = fixed_to_float<FxpInt>(acc_fxp[wm][wn][i]);
              out_frag.x[i] = float_to_acc<Acc>(v);
            }
            wmma::store_matrix_sync(&C_buf[warp_id][0], out_frag, FN,
                                    wmma::mem_row_major);
            __syncwarp();

            const int m_off = m_block + warp_m * kWarpM + wm * FM;
            const int n_off = n_block + warp_n * kWarpN + wn * FN;
#pragma unroll
            for (int e = lane; e < FM * FN; e += 32)
            {
              const int lm = e / FN;
              const int ln = e % FN;
              const int gm = m_off + lm;
              const int gn = n_off + ln;
              if (gm < M && gn < N)
              {
                float val = acc_to_float<Acc>(C_buf[warp_id][lm * FN + ln]);
                if (bias != nullptr)
                {
                  val += static_cast<float>(bias[gn]);
                }
                c[gm * stride_cm + gn * stride_cn] = static_cast<IOFloat>(val);
              }
            }
            __syncwarp();
          }
        }
      }

      template <typename FxpInt, typename IOFloat>
      __global__ void gemm_kernel_tc(
          const IOFloat *__restrict__ a,
          const IOFloat *__restrict__ b,
          const IOFloat *__restrict__ bias,
          IOFloat *__restrict__ c,
          int M, int N, int K,
          int64_t stride_am, int64_t stride_ak,
          int64_t stride_bk, int64_t stride_bn,
          int64_t stride_cm, int64_t stride_cn)
      {
        [[maybe_unused]] constexpr bool kIsFp16 = std::is_same_v<IOFloat, __half>;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
        if constexpr (kIsFp16)
        {
          gemm_body<FxpInt, IOFloat>(a, b, bias, c, M, N, K,
                                     stride_am, stride_ak, stride_bk, stride_bn,
                                     stride_cm, stride_cn);
        }
        else
        {
          // ops.cpp blocks this branch on sm < 80.
          __trap();
        }
#else
        gemm_body<FxpInt, IOFloat>(a, b, bias, c, M, N, K,
                                   stride_am, stride_ak, stride_bk, stride_bn,
                                   stride_cm, stride_cn);
#endif
      }

      // PyTorch storage types are layout-compatible with the cuda native
      // dtypes; reinterpret_cast at the boundary.

      template <typename IOFloat>
      struct AtenDtype
      {
        using type = IOFloat;
      };
      template <>
      struct AtenDtype<__half>
      {
        using type = at::Half;
      };
      template <>
      struct AtenDtype<__nv_bfloat16>
      {
        using type = at::BFloat16;
      };

      template <typename IOFloat>
      inline IOFloat *native_ptr(at::Tensor &t)
      {
        using Aten = typename AtenDtype<IOFloat>::type;
        return reinterpret_cast<IOFloat *>(t.data_ptr<Aten>());
      }

      template <typename IOFloat>
      inline const IOFloat *native_ptr(const at::Tensor &t)
      {
        using Aten = typename AtenDtype<IOFloat>::type;
        return reinterpret_cast<const IOFloat *>(t.data_ptr<Aten>());
      }

      template <typename FxpInt, typename IOFloat>
      void launch_typed(
          const at::Tensor &a, const at::Tensor &b,
          const c10::optional<at::Tensor> &bias, at::Tensor &c)
      {
        const int M = a.size(0);
        const int K = a.size(1);
        const int N = b.size(1);
        if (M == 0 || N == 0)
          return;

        dim3 block(kThreads);
        dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);

        const IOFloat *bias_ptr = bias.has_value()
                                      ? native_ptr<IOFloat>(*bias)
                                      : nullptr;

        auto stream = at::cuda::getCurrentCUDAStream();
        gemm_kernel_tc<FxpInt, IOFloat><<<grid, block, 0, stream>>>(
            native_ptr<IOFloat>(a), native_ptr<IOFloat>(b),
            bias_ptr, native_ptr<IOFloat>(c),
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }

      template <typename FxpInt>
      void launch_tiled(
          const at::Tensor &a, const at::Tensor &b,
          const c10::optional<at::Tensor> &bias, at::Tensor &c)
      {
        switch (a.scalar_type())
        {
        case at::kFloat:
          launch_typed<FxpInt, float>(a, b, bias, c);
          break;
        case at::kHalf:
          launch_typed<FxpInt, __half>(a, b, bias, c);
          break;
        case at::kBFloat16:
          launch_typed<FxpInt, __nv_bfloat16>(a, b, bias, c);
          break;
        default:
          TORCH_CHECK(false, "gemm_fxp: unsupported input dtype ", a.scalar_type());
        }
      }

    } // namespace

    at::Tensor gemm_fxp_run(
        at::Tensor a,
        at::Tensor b,
        c10::optional<at::Tensor> bias,
        int64_t int_bits)
    {
      const c10::cuda::CUDAGuard device_guard(a.device());
      const int M = a.size(0);
      const int N = b.size(1);

      auto c = at::empty({M, N}, a.options());

      switch (int_bits)
      {
      case 16:
        launch_tiled<int16_t>(a, b, bias, c);
        break;
      case 32:
        launch_tiled<int32_t>(a, b, bias, c);
        break;
      case 64:
        launch_tiled<int64_t>(a, b, bias, c);
        break;
      }
      return c;
    }

  } // namespace detail
} // namespace fxpr
