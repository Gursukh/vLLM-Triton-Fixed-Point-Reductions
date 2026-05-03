// Schema and CUDA impl are split so library_ops.py can attach
// register_fake() meta impls.

#include <torch/library.h>
#include <torch/extension.h>

namespace fxpr {

torch::Tensor float_to_fixed_op(torch::Tensor x, int64_t int_bits,
                                int64_t fxp_frac_bits);
torch::Tensor fixed_to_float_op(torch::Tensor x, int64_t float_bits,
                                int64_t fxp_frac_bits);
torch::Tensor rms_norm_fxp_op(torch::Tensor x, torch::Tensor w, double eps,
                              int64_t int_bits, int64_t fxp_frac_bits);
torch::Tensor rms_norm_fxp_residual_op(torch::Tensor x, torch::Tensor residual,
                                       torch::Tensor w, double eps,
                                       int64_t int_bits,
                                       int64_t fxp_frac_bits);
torch::Tensor log_softmax_fxp_op(torch::Tensor x, int64_t int_bits,
                                 int64_t fxp_frac_bits);
torch::Tensor gemm_fxp_op(torch::Tensor a, torch::Tensor b,
                          c10::optional<torch::Tensor> bias,
                          int64_t int_bits, int64_t fxp_frac_bits);
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
                              int64_t num_kv_splits);

}  // namespace fxpr

TORCH_LIBRARY(fxpr, m) {
  m.def("float_to_fixed(Tensor x, int int_bits, int fxp_frac_bits) -> Tensor");
  m.def("fixed_to_float(Tensor x, int float_bits, int fxp_frac_bits) -> Tensor");
  m.def(
      "rms_norm_fxp(Tensor x, Tensor weight_fp32, float eps, "
      "int fxp_int_bits, int fxp_frac_bits) -> Tensor");
  m.def(
      "rms_norm_fxp_residual(Tensor x, Tensor(a!) residual, Tensor weight_fp32, "
      "float eps, int fxp_int_bits, int fxp_frac_bits) -> Tensor");
  m.def("log_softmax_fxp(Tensor x, int fxp_int_bits, int fxp_frac_bits) -> Tensor");
  m.def(
      "gemm_fxp(Tensor a, Tensor b, Tensor? bias, int fxp_int_bits, "
      "int fxp_frac_bits) -> Tensor");
  m.def(
      "unified_attention_fxp(Tensor q, Tensor kv_cache, Tensor(a!) o, "
      "Tensor query_start_loc, Tensor seq_lens, Tensor block_table, "
      "int max_query_len, Tensor? alibi_slopes, bool is_causal, "
      "float? softmax_scale, int fxp_int_bits, int fxp_frac_bits, "
      "float logit_softcap, int window_size, int num_kv_splits) -> ()");
}

TORCH_LIBRARY_IMPL(fxpr, CUDA, m) {
  m.impl("float_to_fixed", &fxpr::float_to_fixed_op);
  m.impl("fixed_to_float", &fxpr::fixed_to_float_op);
  m.impl("rms_norm_fxp", &fxpr::rms_norm_fxp_op);
  m.impl("rms_norm_fxp_residual", &fxpr::rms_norm_fxp_residual_op);
  m.impl("log_softmax_fxp", &fxpr::log_softmax_fxp_op);
  m.impl("gemm_fxp", &fxpr::gemm_fxp_op);
  m.impl("unified_attention_fxp", &fxpr::unified_attention_fxp_op);
}

// Empty module so importing the .so triggers the TORCH_LIBRARY ctors.
PYBIND11_MODULE(_cuda, m) {
  m.doc() = "fxpr_vllm CUDA kernels (registered via torch.ops.fxpr)";
}
