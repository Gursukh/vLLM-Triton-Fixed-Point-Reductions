// TORCH_CHECK shims live here so edits don't trigger a kernel recompile.

#include "ops_internal.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>

namespace fxpr {

namespace {
inline void check_frac_bits(int64_t fxp_frac_bits) {
  TORCH_CHECK(fxp_frac_bits == 8 || fxp_frac_bits == 16 || fxp_frac_bits == 32,
              "fxp_frac_bits must be 8/16/32, got ", fxp_frac_bits);
}
inline bool is_supported_float_dtype(at::ScalarType t) {
  return t == at::kFloat || t == at::kHalf || t == at::kBFloat16;
}
}  // namespace

torch::Tensor float_to_fixed_op(torch::Tensor x, int64_t int_bits,
                                int64_t fxp_frac_bits) {
  TORCH_CHECK(x.is_cuda(), "float_to_fixed: input must be CUDA");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "int_bits must be 16, 32, or 64");
  check_frac_bits(fxp_frac_bits);
  return detail::float_to_fixed_run(std::move(x), int_bits, fxp_frac_bits);
}

torch::Tensor fixed_to_float_op(torch::Tensor x, int64_t float_bits,
                                int64_t fxp_frac_bits) {
  TORCH_CHECK(x.is_cuda(), "fixed_to_float: input must be CUDA");
  TORCH_CHECK(float_bits == 16 || float_bits == 32 || float_bits == 64,
              "float_bits must be 16, 32, or 64");
  check_frac_bits(fxp_frac_bits);
  return detail::fixed_to_float_run(std::move(x), float_bits, fxp_frac_bits);
}

torch::Tensor rms_norm_fxp_op(torch::Tensor x, torch::Tensor w, double eps,
                              int64_t int_bits, int64_t fxp_frac_bits) {
  TORCH_CHECK(x.is_cuda(), "rms_norm: x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "rms_norm: w must be CUDA");
  TORCH_CHECK(is_supported_float_dtype(x.scalar_type()),
              "rms_norm: x must be float32 / float16 / bfloat16");
  TORCH_CHECK(w.scalar_type() == x.scalar_type(),
              "rms_norm: weight dtype must match x dtype");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  check_frac_bits(fxp_frac_bits);
  return detail::rms_norm_fxp_run(std::move(x), std::move(w), eps, int_bits,
                                  fxp_frac_bits);
}

torch::Tensor rms_norm_fxp_residual_op(torch::Tensor x, torch::Tensor residual,
                                       torch::Tensor w, double eps,
                                       int64_t int_bits,
                                       int64_t fxp_frac_bits) {
  TORCH_CHECK(x.is_cuda(), "rms_norm: x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "rms_norm: w must be CUDA");
  TORCH_CHECK(is_supported_float_dtype(x.scalar_type()),
              "rms_norm: x must be float32 / float16 / bfloat16");
  TORCH_CHECK(w.scalar_type() == x.scalar_type(),
              "rms_norm: weight dtype must match x dtype");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  check_frac_bits(fxp_frac_bits);
  TORCH_CHECK(residual.is_cuda(), "rms_norm: residual must be CUDA");
  TORCH_CHECK(residual.scalar_type() == x.scalar_type(),
              "rms_norm: residual dtype must match x dtype");
  TORCH_CHECK(residual.sizes() == x.sizes(),
              "rms_norm: residual must have the same shape as x");
  return detail::rms_norm_fxp_residual_run(std::move(x), std::move(residual),
                                           std::move(w), eps, int_bits,
                                           fxp_frac_bits);
}

torch::Tensor log_softmax_fxp_op(torch::Tensor x, int64_t int_bits,
                                 int64_t fxp_frac_bits) {
  TORCH_CHECK(x.is_cuda(), "log_softmax_fxp: input must be CUDA");
  TORCH_CHECK(is_supported_float_dtype(x.scalar_type()),
              "log_softmax_fxp: x must be float32 / float16 / bfloat16");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  check_frac_bits(fxp_frac_bits);
  return detail::log_softmax_fxp_run(std::move(x), int_bits, fxp_frac_bits);
}

torch::Tensor gemm_fxp_op(torch::Tensor a, torch::Tensor b,
                          c10::optional<torch::Tensor> bias,
                          int64_t int_bits, int64_t fxp_frac_bits) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "gemm_fxp: inputs must be CUDA");
  TORCH_CHECK(is_supported_float_dtype(a.scalar_type()),
              "gemm_fxp: a must be float32 / float16 / bfloat16");
  TORCH_CHECK(b.scalar_type() == a.scalar_type(),
              "gemm_fxp: a and b must share dtype");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm_fxp: 2D inputs required");
  TORCH_CHECK(a.size(1) == b.size(0),
              "gemm_fxp: shape mismatch ", a.sizes(), " @ ", b.sizes());
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  check_frac_bits(fxp_frac_bits);
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "gemm_fxp: bias must be CUDA");
    TORCH_CHECK(bias->dim() == 1, "gemm_fxp: bias must be 1-D");
    TORCH_CHECK(bias->size(0) == b.size(1),
                "gemm_fxp: bias size ", bias->size(0),
                " does not match N=", b.size(1));
    TORCH_CHECK(bias->scalar_type() == a.scalar_type(),
                "gemm_fxp: bias dtype must match a dtype");
  }

  // mma.sync floors: fp16 needs sm_75+, bf16/fp32 need sm_80+.
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int cap_major = props->major;
  const int cap_minor = props->minor;
  const int cap = cap_major * 10 + cap_minor;
  TORCH_CHECK(cap >= 75,
              "gemm_fxp: requires compute capability >= 7.5 (Turing); "
              "device is ", cap_major, ".", cap_minor);
  if (a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kFloat) {
    TORCH_CHECK(cap >= 80,
                "gemm_fxp: ",
                (a.scalar_type() == at::kBFloat16 ? "bfloat16" : "float32"),
                " inputs require compute capability >= 8.0 (Ampere); "
                "device is ", cap_major, ".", cap_minor,
                ". Use float16 inputs on this device.");
  }

  return detail::gemm_fxp_run(std::move(a), std::move(b), std::move(bias),
                              int_bits, fxp_frac_bits);
}

void unified_attention_fxp_op(torch::Tensor q, torch::Tensor kv_cache,
                              torch::Tensor o, torch::Tensor query_start_loc,
                              torch::Tensor seq_lens, torch::Tensor block_table,
                              int64_t max_query_len,
                              c10::optional<torch::Tensor> alibi_slopes,
                              bool is_causal,
                              c10::optional<double> softmax_scale,
                              int64_t fxp_int_bits,
                              int64_t fxp_frac_bits,
                              double logit_softcap, int64_t window_size,
                              int64_t num_kv_splits) {
  TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda() && o.is_cuda(),
              "unified_attention_fxp: all tensors must be CUDA");
  TORCH_CHECK(query_start_loc.is_cuda() && seq_lens.is_cuda() &&
                  block_table.is_cuda(),
              "unified_attention_fxp: metadata tensors must be CUDA");
  TORCH_CHECK(kv_cache.dim() == 5 && kv_cache.size(1) == 2,
              "kv_cache must be (num_blocks, 2, page_size, num_kv_heads, head_dim)");
  TORCH_CHECK(
      q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf ||
          q.scalar_type() == at::kBFloat16,
      "q dtype must be float32 / float16 / bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == q.scalar_type(),
              "kv_cache dtype must match q dtype");
  TORCH_CHECK(o.scalar_type() == q.scalar_type(),
              "o dtype must match q dtype");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
              "query_start_loc must be int32, got ",
              query_start_loc.scalar_type());
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt,
              "seq_lens must be int32, got ", seq_lens.scalar_type());
  TORCH_CHECK(block_table.scalar_type() == at::kInt,
              "block_table must be int32, got ", block_table.scalar_type());
  TORCH_CHECK(fxp_int_bits == 16 || fxp_int_bits == 32 || fxp_int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  check_frac_bits(fxp_frac_bits);
  TORCH_CHECK(num_kv_splits >= 0,
              "num_kv_splits must be >= 0 (0 = auto-pick), got ",
              num_kv_splits);
  if (alibi_slopes.has_value() && alibi_slopes->numel() > 0) {
    TORCH_CHECK(alibi_slopes->scalar_type() == at::kFloat,
                "alibi_slopes must be float32");
  }

  detail::unified_attention_fxp_run(
      std::move(q), std::move(kv_cache), std::move(o),
      std::move(query_start_loc), std::move(seq_lens), std::move(block_table),
      max_query_len, std::move(alibi_slopes), is_causal,
      std::move(softmax_scale), fxp_int_bits, fxp_frac_bits, logit_softcap,
      window_size, num_kv_splits);
}

}  // namespace fxpr
