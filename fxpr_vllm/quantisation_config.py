from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.layers.quantization import register_quantization_config

from .library_ops import gemm_fxp
from .config import DEFAULT_FRAC_BITS, get_runtime_config

logger = logging.getLogger("fxpr_vllm")


@register_quantization_config("fixed_point_det")
class FixedPointConfig(QuantizationConfig):
    def __init__(self, frac_bits: int = DEFAULT_FRAC_BITS) -> None:
        int_bits = get_runtime_config().fxp_int_bits
        if not isinstance(frac_bits, int) or not (0 <= frac_bits < int_bits):
            raise ValueError(
                f"frac_bits must be an int in [0, {int_bits}); got {frac_bits!r}"
            )
        self.frac_bits = frac_bits

    def __repr__(self) -> str:
        return f"FixedPointConfig(frac_bits={self.frac_bits})"

    @classmethod
    def get_name(cls) -> str:
        return "fixed_point_det"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedPointConfig":
        frac_bits = config.get("frac_bits", get_runtime_config().frac_bits)
        return cls(frac_bits=frac_bits)

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return FixedPointLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class FixedPointLinearMethod(QuantizeMethodBase):
    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config
        self.fxp_int_bits = get_runtime_config().fxp_int_bits

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: Sequence[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        total_output_size = sum(output_partition_sizes)

        weight = ModelWeightParameter(
            data=torch.empty(
                total_output_size,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Stored (out, in); GEMM expects (K, N) = (in, out). Cache as fp32.
        with torch.no_grad():
            w = layer.weight.data.to(torch.float32)
            w_t = w.t().contiguous()

        layer.weight_fp32 = w_t
        del layer.weight
        layer.register_parameter("weight", None)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1])
        if x2d.dtype != torch.float32:
            x2d = x2d.to(torch.float32)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()

        out = gemm_fxp(
            x2d,
            layer.weight_fp32,
            self.config.frac_bits,
            self.fxp_int_bits,
        )
        out = out.view(*x.shape[:-1], out.shape[-1])

        if bias is not None:
            out = out + bias.to(torch.float32)

        return out.to(orig_dtype)
