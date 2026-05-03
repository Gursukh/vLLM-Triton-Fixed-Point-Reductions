import torch

from vllm.model_executor.layers.layernorm import RMSNorm

from .library_ops import rms_norm_fxp as rms_norm_fxp_op
from .library_ops import rms_norm_fxp_residual as rms_norm_fxp_residual_op
from .config import get_runtime_config


class DeterministicRMSNorm(RMSNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = get_runtime_config()
        self._fxp_int_bits = cfg.fxp_int_bits

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # vLLM fused-residual API: returns (normalised, residual += x).
        if residual is not None:
            out = rms_norm_fxp_residual_op(
                x,
                residual,
                self.weight,
                self.variance_epsilon,
                self._fxp_int_bits,
            )
            return out, residual

        return rms_norm_fxp_op(
            x,
            self.weight,
            self.variance_epsilon,
            self._fxp_int_bits,
        )
