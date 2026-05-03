import torch

from . import _cuda  # noqa: F401
from .config import get_runtime_config


def deterministic_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    assert logits.is_cuda, "Input tensor must be on CUDA device"

    cfg = get_runtime_config()
    if dim < 0:
        dim += logits.ndim
    if dim != logits.ndim - 1:
        logits = logits.transpose(dim, -1).contiguous()
        transposed = True
    else:
        transposed = False

    out = torch.ops.fxpr.log_softmax_fxp(
        logits, int(cfg.fxp_int_bits), int(cfg.fxp_frac_bits)
    )
    if transposed:
        out = out.transpose(dim, -1).contiguous()
    return out
