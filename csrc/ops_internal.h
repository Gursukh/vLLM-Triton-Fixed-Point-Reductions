#pragma once

#include <ATen/ATen.h>

namespace fxpr {
namespace detail {

at::Tensor float_to_fixed_run(at::Tensor x, int64_t int_bits,
                              int64_t fxp_frac_bits);
at::Tensor fixed_to_float_run(at::Tensor x, int64_t float_bits,
                              int64_t fxp_frac_bits);

at::Tensor rms_norm_fxp_run(at::Tensor x, at::Tensor w, double eps,
                            int64_t int_bits, int64_t fxp_frac_bits);
at::Tensor rms_norm_fxp_residual_run(at::Tensor x, at::Tensor residual,
                                     at::Tensor w, double eps,
                                     int64_t int_bits, int64_t fxp_frac_bits);

at::Tensor log_softmax_fxp_run(at::Tensor x, int64_t int_bits,
                               int64_t fxp_frac_bits);

at::Tensor gemm_fxp_run(at::Tensor a, at::Tensor b,
                        c10::optional<at::Tensor> bias,
                        int64_t int_bits, int64_t fxp_frac_bits);

void unified_attention_fxp_run(at::Tensor q, at::Tensor kv_cache, at::Tensor o,
                               at::Tensor query_start_loc, at::Tensor seq_lens,
                               at::Tensor block_table, int64_t max_query_len,
                               c10::optional<at::Tensor> alibi_slopes,
                               bool is_causal,
                               c10::optional<double> softmax_scale,
                               int64_t fxp_int_bits,
                               int64_t fxp_frac_bits,
                               double logit_softcap, int64_t window_size,
                               int64_t num_kv_splits);

}  // namespace detail
}  // namespace fxpr
