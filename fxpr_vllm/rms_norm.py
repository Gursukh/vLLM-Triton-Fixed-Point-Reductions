import torch

from vllm.model_executor.layers.layernorm import RMSNorm

from .library_ops import rms_norm_fxp as rms_norm_fxp_op
from .library_ops import rms_norm_fxp_residual as rms_norm_fxp_residual_op
from .config import get_runtime_config


class DeterministicRMSNorm(RMSNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = get_runtime_config()
        self._fxp_frac_bits = cfg.frac_bits
        self._fxp_int_bits = cfg.fxp_int_bits

    def _det_norm_torch(self, x_fp32: torch.Tensor) -> torch.Tensor:
        """CPU reference: same fixed-point pipeline as the CUDA kernel, bit-identical."""
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
        """vLLM fused-residual API: with residual, returns (normalised, residual += x)."""
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
