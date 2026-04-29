// Host-side validation shims for the fxpr CUDA ops. Keeps TORCH_CHECK
// calls and other host-only logic out of nvcc-compiled translation units
// so that tweaks to argument validation don't trigger CUDA recompiles.

#include "ops_internal.h"

#include <ATen/ATen.h>
#include <torch/types.h>

namespace fxpr {

torch::Tensor float_to_fixed_op(torch::Tensor x, int64_t frac_bits,
                                int64_t int_bits) {
  TORCH_CHECK(x.is_cuda(), "float_to_fixed: input must be CUDA");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "int_bits must be 16, 32, or 64");
  TORCH_CHECK(frac_bits >= 0 && frac_bits < 127,
              "frac_bits must be in [0, 127)");
  return detail::float_to_fixed_run(std::move(x), frac_bits, int_bits);
}

torch::Tensor fixed_to_float_op(torch::Tensor x, int64_t frac_bits,
                                int64_t float_bits) {
  TORCH_CHECK(x.is_cuda(), "fixed_to_float: input must be CUDA");
  TORCH_CHECK(float_bits == 16 || float_bits == 32 || float_bits == 64,
              "float_bits must be 16, 32, or 64");
  TORCH_CHECK(frac_bits >= 0 && frac_bits < 127,
              "frac_bits must be in [0, 127)");
  return detail::fixed_to_float_run(std::move(x), frac_bits, float_bits);
}

torch::Tensor rms_norm_fxp_op(torch::Tensor x, torch::Tensor w, double eps,
                              int64_t frac_bits, int64_t int_bits) {
  TORCH_CHECK(x.is_cuda(), "rms_norm: x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "rms_norm: w must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "rms_norm: x must be float32");
  TORCH_CHECK(
      w.scalar_type() == at::kFloat || w.scalar_type() == at::kHalf
          || w.scalar_type() == at::kBFloat16,
      "rms_norm: w must be float32 / float16 / bfloat16");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  return detail::rms_norm_fxp_run(std::move(x), std::move(w), eps, frac_bits,
                                  int_bits);
}

torch::Tensor rms_norm_fxp_residual_op(torch::Tensor x, torch::Tensor residual,
                                       torch::Tensor w, double eps,
                                       int64_t frac_bits, int64_t int_bits) {
  TORCH_CHECK(x.is_cuda(), "rms_norm: x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "rms_norm: w must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "rms_norm: x must be float32");
  TORCH_CHECK(
      w.scalar_type() == at::kFloat || w.scalar_type() == at::kHalf
          || w.scalar_type() == at::kBFloat16,
      "rms_norm: w must be float32 / float16 / bfloat16");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  TORCH_CHECK(residual.is_cuda(), "rms_norm: residual must be CUDA");
  TORCH_CHECK(residual.scalar_type() == at::kFloat,
              "rms_norm: residual must be float32");
  TORCH_CHECK(residual.sizes() == x.sizes(),
              "rms_norm: residual must have the same shape as x");
  return detail::rms_norm_fxp_residual_run(std::move(x), std::move(residual),
                                           std::move(w), eps, frac_bits,
                                           int_bits);
}

torch::Tensor log_softmax_fxp_op(torch::Tensor x, int64_t frac_bits,
                                 int64_t int_bits) {
  TORCH_CHECK(x.is_cuda(), "log_softmax_fxp: input must be CUDA");
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  return detail::log_softmax_fxp_run(std::move(x), frac_bits, int_bits);
}

torch::Tensor gemm_fxp_op(torch::Tensor a, torch::Tensor b, int64_t frac_bits,
                          int64_t int_bits) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "gemm_fxp: inputs must be CUDA");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "gemm_fxp: a must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "gemm_fxp: b must be float32");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm_fxp: 2D inputs required");
  TORCH_CHECK(a.size(1) == b.size(0),
              "gemm_fxp: shape mismatch ", a.sizes(), " @ ", b.sizes());
  TORCH_CHECK(int_bits == 16 || int_bits == 32 || int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  return detail::gemm_fxp_run(std::move(a), std::move(b), frac_bits, int_bits);
}

void unified_attention_fxp_op(torch::Tensor q, torch::Tensor kv_cache,
                              torch::Tensor o, torch::Tensor query_start_loc,
                              torch::Tensor seq_lens, torch::Tensor block_table,
                              int64_t max_query_len,
                              c10::optional<torch::Tensor> alibi_slopes,
                              bool is_causal,
                              c10::optional<double> softmax_scale,
                              int64_t frac_bits, int64_t fxp_int_bits,
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
  TORCH_CHECK(o.scalar_type() == at::kFloat, "o must be float32");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
              "query_start_loc must be int32, got ",
              query_start_loc.scalar_type());
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt,
              "seq_lens must be int32, got ", seq_lens.scalar_type());
  TORCH_CHECK(block_table.scalar_type() == at::kInt,
              "block_table must be int32, got ", block_table.scalar_type());
  TORCH_CHECK(fxp_int_bits == 16 || fxp_int_bits == 32 || fxp_int_bits == 64,
              "fxp_int_bits must be 16/32/64");
  TORCH_CHECK(num_kv_splits >= 1,
              "num_kv_splits must be >= 1, got ", num_kv_splits);
  if (alibi_slopes.has_value() && alibi_slopes->numel() > 0) {
    TORCH_CHECK(alibi_slopes->scalar_type() == at::kFloat,
                "alibi_slopes must be float32");
  }

  detail::unified_attention_fxp_run(
      std::move(q), std::move(kv_cache), std::move(o),
      std::move(query_start_loc), std::move(seq_lens), std::move(block_table),
      max_query_len, std::move(alibi_slopes), is_causal,
      std::move(softmax_scale), frac_bits, fxp_int_bits, logit_softcap,
      window_size, num_kv_splits);
}

}  // namespace fxpr
