import torch

from .. import _cuda  # noqa: F401  (registers torch.ops.fxpr.*)
from .config import get_runtime_config


def deterministic_log_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute log-softmax deterministically via the fixed-point softmax kernel.

    Args:
        logits: CUDA tensor of arbitrary shape (..., V) where V is the
            reduction axis when dim == -1. Any floating dtype is accepted;
            the kernel upcasts internally to float32.
        dim: Axis along which to normalise. Defaults to the last axis.

    Returns:
        Tensor of the same shape and dtype as logits, containing log-softmax
        values that are bitwise-reproducible across SM/warp schedules.
    """
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
        logits, int(cfg.frac_bits), int(cfg.fxp_int_bits)
    )
    if transposed:
        out = out.transpose(dim, -1).contiguous()
    return out
