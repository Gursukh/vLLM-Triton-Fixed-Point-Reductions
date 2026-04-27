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

from ..library_ops import (
    compute_per_row_scale,
    gemm_fxp_int8,
    quantise_to_int8,
)
from .config import DEFAULT_FRAC_BITS, get_runtime_config

logger = logging.getLogger("fxpr_vllm")

# Floor for the per-row scale; matches what the test suite uses and
# avoids divide-by-zero on all-zero weight rows.
_SCALE_EPS = 1e-8


@register_quantization_config("fixed_point_det")
class FixedPointConfig(QuantizationConfig):
    def __init__(self, frac_bits: int = DEFAULT_FRAC_BITS) -> None:
        """Create a fixed-point quantisation config.

        Args:
            frac_bits: Number of fractional bits used by the GEMM accumulator.
        """
        int_bits = get_runtime_config().fxp_int_bits
        if not isinstance(frac_bits, int) or not (0 <= frac_bits < int_bits):
            raise ValueError(
                f"frac_bits must be an int in [0, {int_bits}); got {frac_bits!r}"
            )
        self.frac_bits = frac_bits

    def __repr__(self) -> str:
        """Return a human-readable representation including frac_bits."""
        return f"FixedPointConfig(frac_bits={self.frac_bits})"

    @classmethod
    def get_name(cls) -> str:
        """Return the vLLM quantisation method name used in --quantization."""
        return "fixed_point_det"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """Return the activation dtypes accepted by the fixed-point GEMM."""
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the minimum CUDA compute capability (major * 10 + minor)."""
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        """Return the list of HF config filenames consumed by this method (none)."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedPointConfig":
        """Build a :class:`FixedPointConfig` from a serialised dict.

        Args:
            config: Raw dict (e.g. HF quantization_config); frac_bits overrides the runtime default when present.

        Returns:
            A configured :class:`FixedPointConfig`.
        """
        frac_bits = config.get("frac_bits", get_runtime_config().frac_bits)
        return cls(frac_bits=frac_bits)

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
        """Return a :class:`FixedPointLinearMethod` for linear layers, else None."""
        if isinstance(layer, LinearBase):
            return FixedPointLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        """Return activation names that require scaling (none for this method)."""
        return []


class FixedPointLinearMethod(QuantizeMethodBase):
    def __init__(self, config: FixedPointConfig) -> None:
        """Bind this linear method to its parent quantisation config."""
        self.config = config
        # Cache once at construction time; the runtime config is process-global.
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
        """Allocate a weight parameter of shape (sum(output_partition_sizes), input_size_per_partition)."""
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
        """Quantise the weight to int8 + per-output-feature fp16 scale.

        After this runs, the layer carries:
            layer.weight_int8 : (input_size, output_size) int8
            layer.weight_scale: (output_size,) fp16

        weight_int8 is the transpose of the original weight (so apply()
        can call a @ weight_int8 with shape (M, K) @ (K, N)). The
        per-row scale of the weight transpose corresponds to per-output
        feature, computed before the kernel sees any K-partition. This
        matches the split-invariance contract from Migrate.md E.
        """
        with torch.no_grad():
            # Original weight is (output_size, input_size). Transpose so
            # the GEMM contracts along K = input_size: w_t shape
            # (input_size, output_size).
            w = layer.weight.data.to(torch.float32)
            w_t = w.t().contiguous()
            # Per-output-feature scale = per-row scale of w (== per-col of w_t).
            scale = compute_per_row_scale(w, _SCALE_EPS)
            # Quantise w (per-row) then transpose; equivalent to
            # quantising w_t (per-col), but lets us reuse the per-row helper.
            w_int8_rows = quantise_to_int8(w, scale)
            w_int8 = w_int8_rows.t().contiguous()

        layer.weight_int8 = w_int8
        layer.weight_scale = scale
        # Free the original weight; weight_int8 + weight_scale replace it.
        del layer.weight
        layer.register_parameter("weight", None)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run int8 GEMM with per-token activation scale and per-output weight scale."""
        orig_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1])
        # Activation -> fp32 -> per-row scale -> int8.
        if x2d.dtype != torch.float32:
            x2d = x2d.to(torch.float32)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()

        a_scale = compute_per_row_scale(x2d, _SCALE_EPS)
        a_int8 = quantise_to_int8(x2d, a_scale)

        out = gemm_fxp_int8(
            a_int8,
            a_scale,
            layer.weight_int8,
            layer.weight_scale,
            self.config.frac_bits,
            self.fxp_int_bits,
        )
        out = out.view(*x.shape[:-1], out.shape[-1])

        if bias is not None:
            out = out + bias.to(torch.float32)

        return out.to(orig_dtype)
