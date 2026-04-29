import pytest
import torch

import fxpr_vllm._cuda  # noqa: F401  (registers torch.ops.fxpr.*)
from tests.fixed_point_helpers import requires_cuda


def _run_rms_norm_kernel(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.ndim == 2 and w.ndim == 1
    assert x.shape[1] == w.shape[0]
    return torch.ops.fxpr.rms_norm_fxp(x, w, eps, 16, 64)


def _run_rms_norm_float_kernel(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Torch-native fp32 RMSNorm reference."""
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.ndim == 2 and w.ndim == 1
    assert x.shape[1] == w.shape[0]

    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rrms = 1.0 / torch.sqrt(mean_sq + eps)
    return x * w * rrms


def _ordered_float16_sum_of_squares(values: list[float]) -> torch.Tensor:
    acc = torch.zeros((), device="cuda", dtype=torch.float16)
    for v in values:
        t = torch.tensor(v, device="cuda", dtype=torch.float16)
        acc = acc + t * t
    return acc


@requires_cuda
@pytest.mark.parametrize("shape", [(2, 7), (3, 16), (4, 31)])
def test_rms_norm_fixed_point_correctness(shape):
    batch, hidden = shape
    g = torch.Generator(device="cuda").manual_seed(0)

    # Keep values comfortably away from per-element fixed-point saturation at Q16.16.
    x = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    ) * 32.0
    w = (
        torch.rand((hidden,), device="cuda", dtype=torch.float32, generator=g) - 0.5
    ) * 4.0

    got = _run_rms_norm_kernel(x, w, eps=1e-6)
    ref = _run_rms_norm_float_kernel(x, w, eps=1e-6)

    # Fixed-point sum quantization (Q16.16) should stay close to full-float RMSNorm.
    assert torch.allclose(got, ref, atol=5e-4, rtol=5e-4)


@requires_cuda
@pytest.mark.parametrize("int_bits,frac_bits", [(16, 8), (32, 16), (64, 32)])
def test_rms_norm_parametrized_int_bits(int_bits, frac_bits):
    """Exercise _int_dtype_for_bits across all supported (int_bits, frac_bits)."""
    batch, hidden = 3, 16
    g = torch.Generator(device="cuda").manual_seed(int_bits)

    # Keep values small so per-element squares fit even at int_bits=16, frac_bits=8.
    x = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    )
    w = (
        torch.rand((hidden,), device="cuda", dtype=torch.float32, generator=g) - 0.5
    ) * 2.0

    got = torch.ops.fxpr.rms_norm_fxp(x, w, 1e-6, frac_bits, int_bits)
    ref = _run_rms_norm_float_kernel(x, w, eps=1e-6)

    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(got - ref).abs().max().item()} "
        f"at int_bits={int_bits} frac_bits={frac_bits}"
    )


@requires_cuda
def test_rms_norm_associativity_fixed_vs_float16():
    # Squares are [4096, 1, 1, 1, 1], which produce order-dependent fp16 sums:
    # ((4096 + 1) + 1 + 1 + 1) -> 4096, but (1 + 1 + 1 + 1 + 4096) -> 4100.
    order_a = [64.0, 1.0, 1.0, 1.0, 1.0]
    order_b = [1.0, 1.0, 1.0, 1.0, 64.0]

    float16_sum_a = _ordered_float16_sum_of_squares(order_a)
    float16_sum_b = _ordered_float16_sum_of_squares(order_b)
    assert float16_sum_a.item() != float16_sum_b.item()

    x_a = torch.tensor(order_a, device="cuda", dtype=torch.float32)
    perm = torch.tensor([1, 2, 3, 4, 0], device="cuda", dtype=torch.long)
    x_b = x_a[perm]

    x = torch.stack([x_a, x_b], dim=0)
    w = torch.ones((x.shape[1],), device="cuda", dtype=torch.float32)
    y = _run_rms_norm_kernel(x, w, eps=1e-6)

    # With fixed-point sum of squares, the reduction is associative/order-invariant.
    assert torch.equal(y[1], y[0][perm])
