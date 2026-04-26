import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm

from ..library_ops import rms_norm_fxp as rms_norm_fxp_op
from ..library_ops import rms_norm_fxp_residual as rms_norm_fxp_residual_op
from .config import get_runtime_config


@CustomOp.register("rms_norm")
@CustomOp.register_oot(name="RMSNorm")
class DeterministicRMSNorm(RMSNorm):
    def __init__(self, *args, **kwargs) -> None:
        """Cache the runtime fixed-point config; weight is widened in-kernel."""
        super().__init__(*args, **kwargs)
        cfg = get_runtime_config()
        self._fxp_frac_bits = cfg.frac_bits
        self._fxp_int_bits = cfg.fxp_int_bits

    def _det_norm_torch(self, x_fp32: torch.Tensor) -> torch.Tensor:
        """Deterministic RMSNorm in pure torch (CPU path).

        Uses the same fixed-point cast → integer sum → float pipeline as the
        Triton kernel, so CPU and CUDA paths are bit-identical for the same
        inputs (within fp32 rounding of the final ``rrms`` multiply).
        """
        frac_bits = self._fxp_frac_bits
        int_bits = self._fxp_int_bits
        int_dtype = {16: torch.int16, 32: torch.int32, 64: torch.int64}[int_bits]
        scale = float(1 << frac_bits)
        qmax = (1 << (int_bits - 1)) - 1
        qmin = -(1 << (int_bits - 1))

        sq = x_fp32 * x_fp32
        scaled = (sq * scale).round().clamp_(qmin, qmax).to(int_dtype)
        sum_int = scaled.sum(dim=-1, dtype=torch.int64)
        sum_fp = sum_int.to(torch.float32) / scale
        mean_sq = (sum_fp / x_fp32.shape[-1]).clamp_min_(0.0)
        rrms = torch.rsqrt(mean_sq + self.variance_epsilon)
        return x_fp32 * self.weight.to(torch.float32) * rrms.unsqueeze(-1)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Deterministic RMSNorm forward on CPU (or any non-CUDA device).

        Mirrors the CUDA path's fused-residual contract. The integer sum is
        bit-identical to the Triton kernel's reduction, so CPU determinism
        guarantees match the CUDA path's.
        """
        orig_dtype = x.dtype
        if residual is not None:
            new_residual = x.to(torch.float32) + residual.to(torch.float32)
            out = self._det_norm_torch(new_residual)
            return out.to(orig_dtype), new_residual.to(residual.dtype)
        out = self._det_norm_torch(x.to(torch.float32))
        return out.to(orig_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Deterministic RMSNorm forward on CUDA, matching vLLM's fused-residual API.

        When ``residual`` is provided the kernel performs ``residual += x`` in
        fp32 in-place (matching vLLM's fused contract) and normalises the
        accumulated value, returning ``(normalised, residual)``.

        The kernel widens the weight via ``tl.load(W_ptr).to(tl.float32)``
        internally, so we pass ``self.weight`` directly without a Python-side
        fp32 cache. The previous ``data_ptr()``-keyed cache tripped Dynamo
        under ``torch.compile`` (DataPtrVariable can't be ``isinstance``'d).
        """
        orig_dtype = x.dtype

        if residual is not None:
            new_residual_fp32 = residual.to(torch.float32, copy=True)
            out = rms_norm_fxp_residual_op(
                x.to(torch.float32),
                new_residual_fp32,
                self.weight,
                self.variance_epsilon,
                self._fxp_frac_bits,
                self._fxp_int_bits,
            )
            return out.to(orig_dtype), new_residual_fp32.to(residual.dtype)

        out = rms_norm_fxp_op(
            x.to(torch.float32),
            self.weight,
            self.variance_epsilon,
            self._fxp_frac_bits,
            self._fxp_int_bits,
        )
        return out.to(orig_dtype)
