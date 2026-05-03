// Unified prefill + decode attention, paged KV, split-K.
// All reductions are integer (qk dot, denom, wV) or fmaxf (row_max).
// softmax_scale is pre-scaled by 1/ln(2) so qk lives in log2 space.

#include "fixed_point.cuh"
#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace fxpr {
namespace detail {

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 256;
constexpr int kMaxPerThread = 8;
constexpr float kRcpLn2 = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

template <typename FxpInt>
__device__ FxpInt block_reduce_sum(FxpInt v, FxpInt* shmem) {
  // Single-warp fast path skips the smem stage and two __syncthreads.
  if (blockDim.x <= kWarpSize) {
    return warp_reduce_sum<FxpInt>(v);
  }

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

template <typename T>
__device__ __forceinline__ const T* paged_kv_row(
    const T* kv_base,
    const int* block_table_row,
    int key_pos,
    int page_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_head) {
  const int logical_block = key_pos / page_size;
  const int slot = key_pos % page_size;
  const int physical_block = block_table_row[logical_block];
  return kv_base
      + physical_block * stride_block
      + slot * stride_slot
      + kv_head * stride_head;
}

// Slice [key_start, key_end) into num_splits chunks (some may be empty).
__device__ __forceinline__ void compute_chunk(
    int key_start, int key_end, int num_splits, int split_index,
    int* chunk_start_out, int* chunk_end_out) {
  const int total = key_end - key_start;
  // 64-bit to avoid overflow on long contexts.
  const int64_t s_lo = (int64_t)split_index * total;
  const int64_t s_hi = (int64_t)(split_index + 1) * total;
  *chunk_start_out = key_start + (int)(s_lo / num_splits);
  *chunk_end_out   = key_start + (int)(s_hi / num_splits);
}

// Shared between both passes so qk_post matches per key.
template <typename FxpInt, int kFracBits>
__device__ __forceinline__ float post_process_qk(
    FxpInt qk_fxp,
    float softmax_scale_log2, float logit_softcap,
    float alibi_slope, int kp, int q_abs_pos) {
  float qk = fixed_to_float<FxpInt, kFracBits>(qk_fxp);
  qk = qk * softmax_scale_log2;
  if (logit_softcap > 0.0f) {
    const float softcap_log2 = logit_softcap * kRcpLn2;
    qk = softcap_log2 * tanhf(qk / softcap_log2);
  }
  qk += alibi_slope * (static_cast<float>(kp) - static_cast<float>(q_abs_pos));
  return qk;
}

template <typename FxpInt, typename IOFloat, int kFracBits>
__global__ void attn_split_max_kernel(
    const IOFloat* __restrict__ Q,                  // (T, H, D)
    const IOFloat* __restrict__ K_cache,            // (B, page_size, H_kv, D)
    float*         __restrict__ partial_max,        // (T, H, S) fp32
    const int*   __restrict__ query_start_loc,      // (R+1,)
    const int*   __restrict__ seq_lens,             // (R,)
    const int*   __restrict__ block_table,          // (R, max_logical_blocks)
    const float* __restrict__ alibi_slopes,         // (H,) or nullptr; pre * RCP_LN2
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   page_size,
    int   num_splits,
    int64_t stride_q_token,
    int64_t stride_q_head,
    int64_t stride_k_block,
    int64_t stride_k_slot,
    int64_t stride_k_head,
    int64_t stride_pmax_token,
    int64_t stride_pmax_head,
    int64_t stride_pmax_split,
    int64_t stride_block_table_row,
    float softmax_scale_log2,
    float logit_softcap,
    int   window_size,
    bool  is_causal) {
  const int request_index = blockIdx.x;
  const int q_in_request = blockIdx.y;
  const int hs = blockIdx.z;
  const int head_index = hs / num_splits;
  const int split_index = hs % num_splits;
  const int kv_head_index = head_index / (num_heads / num_kv_heads);

  const int q_start = query_start_loc[request_index];
  const int q_end = query_start_loc[request_index + 1];
  const int q_len = q_end - q_start;
  if (q_in_request >= q_len) return;

  const int seq_len = seq_lens[request_index];
  const int context_len = seq_len - q_len;
  const int q_token_idx = q_start + q_in_request;
  const int q_abs_pos = context_len + q_in_request;

  const int key_end = is_causal ? (q_abs_pos + 1) : seq_len;
  const int key_start = window_size > 0 ? max(0, q_abs_pos - window_size + 1) : 0;

  int chunk_start, chunk_end;
  compute_chunk(key_start, key_end, num_splits, split_index, &chunk_start, &chunk_end);

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // partial_max is host-init -INFINITY (fmaxf identity).
  if (chunk_start >= chunk_end) return;

  const float alibi_slope =
      alibi_slopes != nullptr ? alibi_slopes[head_index] : 0.0f;

  const IOFloat* q_row = Q + q_token_idx * stride_q_token + head_index * stride_q_head;
  const int*     bt_row = block_table + request_index * stride_block_table_row;

  __shared__ FxpInt shm_int[kMaxThreads / kWarpSize];

  extern __shared__ unsigned char dyn_smem[];
  IOFloat* q_smem = reinterpret_cast<IOFloat*>(dyn_smem);

  for (int d = tid; d < head_dim; d += nthreads) {
    q_smem[d] = q_row[d];
  }
  __syncthreads();

  float row_max = -INFINITY;
  for (int kp = chunk_start; kp < chunk_end; ++kp) {
    const IOFloat* k_row = paged_kv_row<IOFloat>(
        K_cache, bt_row, kp, page_size, kv_head_index,
        stride_k_block, stride_k_slot, stride_k_head);

    FxpInt partial = 0;
    for (int d = tid; d < head_dim; d += nthreads) {
      const float qd = static_cast<float>(q_smem[d]);
      const float kd = static_cast<float>(k_row[d]);
      partial += float_to_fixed<FxpInt, kFracBits>(qd * kd);
    }
    partial = block_reduce_sum<FxpInt>(partial, shm_int);

    const float qk = post_process_qk<FxpInt, kFracBits>(
        partial, softmax_scale_log2, logit_softcap,
        alibi_slope, kp, q_abs_pos);
    row_max = fmaxf(row_max, qk);
  }

  if (tid == 0) {
    partial_max[q_token_idx * stride_pmax_token
                + head_index * stride_pmax_head
                + split_index * stride_pmax_split] = row_max;
  }
}

template <typename FxpInt, typename IOFloat, int kFracBits>
__global__ void attn_split_dv_kernel(
    const IOFloat* __restrict__ Q,                  // (T, H, D)
    const IOFloat* __restrict__ K_cache,            // (B, page_size, H_kv, D)
    const IOFloat* __restrict__ V_cache,            // (B, page_size, H_kv, D)
    const float*   __restrict__ partial_max,        // (T, H, S) fp32
    FxpInt*        __restrict__ partial_denom,      // (T, H, S)
    FxpInt*        __restrict__ partial_wv,         // (T, H, S, D)
    const int*   __restrict__ query_start_loc,      // (R+1,)
    const int*   __restrict__ seq_lens,             // (R,)
    const int*   __restrict__ block_table,          // (R, max_logical_blocks)
    const float* __restrict__ alibi_slopes,         // (H,) or nullptr
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   page_size,
    int   num_splits,
    int64_t stride_q_token,
    int64_t stride_q_head,
    int64_t stride_k_block,
    int64_t stride_k_slot,
    int64_t stride_k_head,
    int64_t stride_v_block,
    int64_t stride_v_slot,
    int64_t stride_v_head,
    int64_t stride_pmax_token,
    int64_t stride_pmax_head,
    int64_t stride_pmax_split,
    int64_t stride_pdenom_token,
    int64_t stride_pdenom_head,
    int64_t stride_pdenom_split,
    int64_t stride_pwv_token,
    int64_t stride_pwv_head,
    int64_t stride_pwv_split,
    int64_t stride_pwv_d,
    int64_t stride_block_table_row,
    float softmax_scale_log2,
    float logit_softcap,
    int   window_size,
    bool  is_causal) {
  const int request_index = blockIdx.x;
  const int q_in_request = blockIdx.y;
  const int hs = blockIdx.z;
  const int head_index = hs / num_splits;
  const int split_index = hs % num_splits;
  const int kv_head_index = head_index / (num_heads / num_kv_heads);

  const int q_start = query_start_loc[request_index];
  const int q_end = query_start_loc[request_index + 1];
  const int q_len = q_end - q_start;
  if (q_in_request >= q_len) return;

  const int seq_len = seq_lens[request_index];
  const int context_len = seq_len - q_len;
  const int q_token_idx = q_start + q_in_request;
  const int q_abs_pos = context_len + q_in_request;

  const int key_end = is_causal ? (q_abs_pos + 1) : seq_len;
  const int key_start = window_size > 0 ? max(0, q_abs_pos - window_size + 1) : 0;

  int chunk_start, chunk_end;
  compute_chunk(key_start, key_end, num_splits, split_index, &chunk_start, &chunk_end);

  // partial_denom / partial_wv are host-init 0.
  if (chunk_start >= chunk_end) return;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const float alibi_slope =
      alibi_slopes != nullptr ? alibi_slopes[head_index] : 0.0f;

  const IOFloat* q_row = Q + q_token_idx * stride_q_token + head_index * stride_q_head;
  const int*     bt_row = block_table + request_index * stride_block_table_row;

  __shared__ FxpInt shm_int[kMaxThreads / kWarpSize];

  extern __shared__ unsigned char dyn_smem[];
  IOFloat* q_smem = reinterpret_cast<IOFloat*>(dyn_smem);
  for (int d = tid; d < head_dim; d += nthreads) {
    q_smem[d] = q_row[d];
  }
  __syncthreads();

  __shared__ float s_global_max;
  if (tid == 0) {
    float m = -INFINITY;
    const float* pm_row = partial_max
        + q_token_idx * stride_pmax_token
        + head_index * stride_pmax_head;
    for (int s = 0; s < num_splits; ++s) {
      m = fmaxf(m, pm_row[s * stride_pmax_split]);
    }
    s_global_max = m;
  }
  __syncthreads();
  const float global_max = s_global_max;

  FxpInt out_acc[kMaxPerThread];
  #pragma unroll
  for (int i = 0; i < kMaxPerThread; ++i) out_acc[i] = 0;

  FxpInt denom_partial = 0;

  for (int kp = chunk_start; kp < chunk_end; ++kp) {
    const IOFloat* k_row = paged_kv_row<IOFloat>(
        K_cache, bt_row, kp, page_size, kv_head_index,
        stride_k_block, stride_k_slot, stride_k_head);
    const IOFloat* v_row = paged_kv_row<IOFloat>(
        V_cache, bt_row, kp, page_size, kv_head_index,
        stride_v_block, stride_v_slot, stride_v_head);

    // Cheaper to recompute qk than to stage it through HBM.
    FxpInt qk_fxp = 0;
    for (int d = tid; d < head_dim; d += nthreads) {
      const float qd = static_cast<float>(q_smem[d]);
      const float kd = static_cast<float>(k_row[d]);
      qk_fxp += float_to_fixed<FxpInt, kFracBits>(qd * kd);
    }
    qk_fxp = block_reduce_sum<FxpInt>(qk_fxp, shm_int);

    const float qk = post_process_qk<FxpInt, kFracBits>(
        qk_fxp, softmax_scale_log2, logit_softcap,
        alibi_slope, kp, q_abs_pos);
    const float weight = exp2f(qk - global_max);
    const FxpInt weight_fxp = float_to_fixed<FxpInt, kFracBits>(weight);

    if (tid == 0) denom_partial += weight_fxp;

    int local_idx = 0;
    for (int d = tid; d < head_dim; d += nthreads) {
      const float vd = static_cast<float>(v_row[d]);
      const float prod = weight * vd;
      out_acc[local_idx] += float_to_fixed<FxpInt, kFracBits>(prod);
      ++local_idx;
    }
  }

  if (tid == 0) {
    partial_denom[q_token_idx * stride_pdenom_token
                  + head_index * stride_pdenom_head
                  + split_index * stride_pdenom_split] = denom_partial;
  }

  FxpInt* pwv_row = partial_wv
      + q_token_idx * stride_pwv_token
      + head_index * stride_pwv_head
      + split_index * stride_pwv_split;
  int local_idx = 0;
  for (int d = tid; d < head_dim; d += nthreads) {
    pwv_row[d * stride_pwv_d] = out_acc[local_idx];
    ++local_idx;
  }
}

template <typename FxpInt, typename IOFloat, int kFracBits>
__global__ void attn_combine_kernel(
    const FxpInt* __restrict__ partial_denom,    // (T, H, S)
    const FxpInt* __restrict__ partial_wv,       // (T, H, S, D)
    IOFloat*      __restrict__ O,                // (T, H, D)
    const int* __restrict__ query_start_loc,
    int num_heads, int head_dim, int num_splits,
    int64_t stride_pdenom_token,
    int64_t stride_pdenom_head,
    int64_t stride_pdenom_split,
    int64_t stride_pwv_token,
    int64_t stride_pwv_head,
    int64_t stride_pwv_split,
    int64_t stride_pwv_d,
    int64_t stride_o_token,
    int64_t stride_o_head) {
  const int request_index = blockIdx.x;
  const int q_in_request = blockIdx.y;
  const int head_index = blockIdx.z;

  const int q_start = query_start_loc[request_index];
  const int q_end = query_start_loc[request_index + 1];
  const int q_len = q_end - q_start;
  if (q_in_request >= q_len) return;
  const int q_token_idx = q_start + q_in_request;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  __shared__ float s_denom;
  if (tid == 0) {
    FxpInt acc = 0;
    const FxpInt* pdenom_row = partial_denom
        + q_token_idx * stride_pdenom_token
        + head_index * stride_pdenom_head;
    for (int s = 0; s < num_splits; ++s) {
      acc += pdenom_row[s * stride_pdenom_split];
    }
    const float d = fixed_to_float<FxpInt, kFracBits>(acc);
    s_denom = fmaxf(d, 1.0e-6f);
  }
  __syncthreads();
  const float denom = s_denom;

  const FxpInt* pwv_row = partial_wv
      + q_token_idx * stride_pwv_token
      + head_index * stride_pwv_head;
  IOFloat* o_row = O + q_token_idx * stride_o_token + head_index * stride_o_head;
  for (int d = tid; d < head_dim; d += nthreads) {
    FxpInt acc = 0;
    for (int s = 0; s < num_splits; ++s) {
      acc += pwv_row[s * stride_pwv_split + d * stride_pwv_d];
    }
    const float num = fixed_to_float<FxpInt, kFracBits>(acc);
    o_row[d] = static_cast<IOFloat>(num / denom);
  }
}

template <typename FxpInt, typename IOFloat, int kFracBits>
void launch_attention_typed(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& o,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    const at::Tensor& block_table,
    const at::Tensor* alibi_slopes,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    float softmax_scale_log2,
    float logit_softcap,
    int window_size,
    bool is_causal,
    int max_query_len,
    int num_requests,
    int num_splits) {
  if (num_requests == 0) return;

  // threads = next_pow2(head_dim), capped at kMaxThreads. More warps
  // hide LDG latency in the per-key inner loop; single-warp was ~2.5x
  // slower on the A100 profile.
  int threads = 32;
  while (threads < head_dim && threads < kMaxThreads) threads <<= 1;
  if (threads > kMaxThreads) threads = kMaxThreads;

  TORCH_CHECK(head_dim <= kMaxThreads * kMaxPerThread,
              "unified_attention_fxp: head_dim=", head_dim,
              " exceeds kMaxThreads*kMaxPerThread=",
              kMaxThreads * kMaxPerThread);
  TORCH_CHECK((head_dim + threads - 1) / threads <= kMaxPerThread,
              "unified_attention_fxp: ceil(head_dim/threads)=",
              (head_dim + threads - 1) / threads,
              " exceeds kMaxPerThread=", kMaxPerThread,
              " (head_dim=", head_dim, ", threads=", threads, ")");
  TORCH_CHECK(num_splits >= 0, "num_splits must be >= 0");

  // Prefill already has enough CTAs to fill the GPU, so force splits=1.
  // Decode honours num_splits when >= 1; pass 0 to auto-pick.
  int effective_splits;
  if (max_query_len > 1) {
    effective_splits = 1;
  } else if (num_splits >= 1) {
    effective_splits = num_splits;
  } else {
    const auto& props = *at::cuda::getDeviceProperties(q.device().index());
    const int num_sms = props.multiProcessorCount;
    const int total_mblocks = std::max(1, num_requests * num_heads);
    effective_splits = std::max(1, num_sms / total_mblocks);
  }

  const auto fxp_dtype = c10::CppTypeToScalarType<FxpInt>::value;
  const int64_t T_total = q.size(0);

  auto partial_max = at::full(
      {T_total, num_heads, effective_splits},
      -std::numeric_limits<float>::infinity(),
      q.options().dtype(at::kFloat));
  auto partial_denom = at::zeros(
      {T_total, num_heads, effective_splits},
      q.options().dtype(fxp_dtype));
  auto partial_wv = at::zeros(
      {T_total, num_heads, effective_splits, head_dim},
      q.options().dtype(fxp_dtype));

  const size_t dyn_smem_bytes = head_dim * sizeof(IOFloat);

  dim3 grid_split(num_requests, max_query_len, num_heads * effective_splits);
  dim3 grid_combine(num_requests, max_query_len, num_heads);

  auto stream = at::cuda::getCurrentCUDAStream();

  attn_split_max_kernel<FxpInt, IOFloat, kFracBits><<<grid_split, threads, dyn_smem_bytes, stream>>>(
      q.data_ptr<IOFloat>(),
      k_cache.data_ptr<IOFloat>(),
      partial_max.data_ptr<float>(),
      query_start_loc.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      block_table.data_ptr<int>(),
      alibi_slopes ? alibi_slopes->data_ptr<float>() : nullptr,
      num_heads, num_kv_heads, head_dim, page_size, effective_splits,
      q.stride(0), q.stride(1),
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
      partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
      block_table.stride(0),
      softmax_scale_log2,
      logit_softcap,
      window_size,
      is_causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  attn_split_dv_kernel<FxpInt, IOFloat, kFracBits><<<grid_split, threads, dyn_smem_bytes, stream>>>(
      q.data_ptr<IOFloat>(),
      k_cache.data_ptr<IOFloat>(),
      v_cache.data_ptr<IOFloat>(),
      partial_max.data_ptr<float>(),
      partial_denom.template data_ptr<FxpInt>(),
      partial_wv.template data_ptr<FxpInt>(),
      query_start_loc.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      block_table.data_ptr<int>(),
      alibi_slopes ? alibi_slopes->data_ptr<float>() : nullptr,
      num_heads, num_kv_heads, head_dim, page_size, effective_splits,
      q.stride(0), q.stride(1),
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
      partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
      partial_denom.stride(0), partial_denom.stride(1), partial_denom.stride(2),
      partial_wv.stride(0), partial_wv.stride(1), partial_wv.stride(2), partial_wv.stride(3),
      block_table.stride(0),
      softmax_scale_log2,
      logit_softcap,
      window_size,
      is_causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int combine_threads = 32;
  while (combine_threads < head_dim && combine_threads < kMaxThreads) combine_threads <<= 1;
  if (combine_threads > kMaxThreads) combine_threads = kMaxThreads;

  attn_combine_kernel<FxpInt, IOFloat, kFracBits><<<grid_combine, combine_threads, 0, stream>>>(
      partial_denom.template data_ptr<FxpInt>(),
      partial_wv.template data_ptr<FxpInt>(),
      o.data_ptr<IOFloat>(),
      query_start_loc.data_ptr<int>(),
      num_heads, head_dim, effective_splits,
      partial_denom.stride(0), partial_denom.stride(1), partial_denom.stride(2),
      partial_wv.stride(0), partial_wv.stride(1), partial_wv.stride(2), partial_wv.stride(3),
      o.stride(0), o.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FxpInt, int kFracBits>
void launch_attention(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& o,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    const at::Tensor& block_table,
    const at::Tensor* alibi_slopes,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    float softmax_scale_log2,
    float logit_softcap,
    int window_size,
    bool is_causal,
    int max_query_len,
    int num_requests,
    int num_splits) {
  switch (q.scalar_type()) {
    case at::kFloat:
      launch_attention_typed<FxpInt, float, kFracBits>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_slopes, num_heads, num_kv_heads, head_dim, page_size,
          softmax_scale_log2, logit_softcap, window_size, is_causal,
          max_query_len, num_requests, num_splits);
      break;
    case at::kHalf:
      launch_attention_typed<FxpInt, at::Half, kFracBits>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_slopes, num_heads, num_kv_heads, head_dim, page_size,
          softmax_scale_log2, logit_softcap, window_size, is_causal,
          max_query_len, num_requests, num_splits);
      break;
    case at::kBFloat16:
      launch_attention_typed<FxpInt, at::BFloat16, kFracBits>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_slopes, num_heads, num_kv_heads, head_dim, page_size,
          softmax_scale_log2, logit_softcap, window_size, is_causal,
          max_query_len, num_requests, num_splits);
      break;
    default:
      TORCH_CHECK(false, "unified_attention_fxp: unsupported q dtype ",
                  q.scalar_type());
  }
}

template <int kFracBits>
void attention_dispatch_int(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& o,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    const at::Tensor& block_table,
    const at::Tensor* alibi_ptr,
    int num_heads, int num_kv_heads, int head_dim, int page_size,
    float ss_log2, float logit_softcap_f, int window_size_i, bool is_causal,
    int max_query_len_i, int num_requests, int num_kv_splits_i,
    int64_t fxp_int_bits) {
  switch (fxp_int_bits) {
    case 16:
      launch_attention<int16_t, kFracBits>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_ptr, num_heads, num_kv_heads, head_dim, page_size, ss_log2,
          logit_softcap_f, window_size_i, is_causal, max_query_len_i,
          num_requests, num_kv_splits_i);
      break;
    case 32:
      launch_attention<int32_t, kFracBits>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_ptr, num_heads, num_kv_heads, head_dim, page_size, ss_log2,
          logit_softcap_f, window_size_i, is_causal, max_query_len_i,
          num_requests, num_kv_splits_i);
      break;
    case 64:
      launch_attention<int64_t, kFracBits>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_ptr, num_heads, num_kv_heads, head_dim, page_size, ss_log2,
          logit_softcap_f, window_size_i, is_causal, max_query_len_i,
          num_requests, num_kv_splits_i);
      break;
  }
}

}  // namespace

void unified_attention_fxp_run(
    at::Tensor q,
    at::Tensor kv_cache,
    at::Tensor o,
    at::Tensor query_start_loc,
    at::Tensor seq_lens,
    at::Tensor block_table,
    int64_t max_query_len,
    c10::optional<at::Tensor> alibi_slopes,
    bool is_causal,
    c10::optional<double> softmax_scale,
    int64_t fxp_int_bits,
    int64_t fxp_frac_bits,
    double logit_softcap,
    int64_t window_size,
    int64_t num_kv_splits) {
  const c10::cuda::CUDAGuard device_guard(q.device());

  // Strided views; calling .contiguous() here would double HBM usage.
  auto k_cache = kv_cache.select(1, 0);
  auto v_cache = kv_cache.select(1, 1);

  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int num_kv_heads = k_cache.size(2);
  const int page_size = k_cache.size(1);
  const int num_requests = seq_lens.size(0);

  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");

  const float ss = softmax_scale.has_value()
      ? static_cast<float>(softmax_scale.value())
      : 1.0f / sqrtf(static_cast<float>(head_dim));
  const float ss_log2 = ss * 1.4426950408889634f;

  const at::Tensor* alibi_ptr = nullptr;
  at::Tensor alibi_holder;
  if (alibi_slopes.has_value() && alibi_slopes->numel() > 0) {
    alibi_holder = alibi_slopes->contiguous();
    alibi_ptr = &alibi_holder;
  }

  const float logit_softcap_f = static_cast<float>(logit_softcap);
  const int window_size_i = static_cast<int>(window_size);
  const int max_query_len_i = static_cast<int>(max_query_len);
  const int num_kv_splits_i = static_cast<int>(num_kv_splits);

  switch (fxp_frac_bits) {
    case 8:
      attention_dispatch_int<8>(q, k_cache, v_cache, o, query_start_loc,
                                seq_lens, block_table, alibi_ptr,
                                num_heads, num_kv_heads, head_dim, page_size,
                                ss_log2, logit_softcap_f, window_size_i,
                                is_causal, max_query_len_i, num_requests,
                                num_kv_splits_i, fxp_int_bits);
      break;
    case 16:
      attention_dispatch_int<16>(q, k_cache, v_cache, o, query_start_loc,
                                 seq_lens, block_table, alibi_ptr,
                                 num_heads, num_kv_heads, head_dim, page_size,
                                 ss_log2, logit_softcap_f, window_size_i,
                                 is_causal, max_query_len_i, num_requests,
                                 num_kv_splits_i, fxp_int_bits);
      break;
    case 32:
      attention_dispatch_int<32>(q, k_cache, v_cache, o, query_start_loc,
                                 seq_lens, block_table, alibi_ptr,
                                 num_heads, num_kv_heads, head_dim, page_size,
                                 ss_log2, logit_softcap_f, window_size_i,
                                 is_causal, max_query_len_i, num_requests,
                                 num_kv_splits_i, fxp_int_bits);
      break;
  }
}

}  // namespace detail
}  // namespace fxpr
