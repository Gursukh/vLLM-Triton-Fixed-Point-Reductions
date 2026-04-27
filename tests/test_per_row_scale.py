"""Tests for compute_per_row_scale.

The split-invariance check is the load-bearing one for tier 2: every
per-row scale must be bit-identical regardless of how the input rows
are partitioned across launches. The whole int8-MMA path falls apart
without this property.
"""

import pytest
import torch

import fxpr_vllm._cuda  # noqa: F401  (registers torch.ops.fxpr.*)
from tests.fixed_point_helpers import requires_cuda


@requires_cuda
def test_per_row_scale_matches_torch_max_abs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((8, 64), device="cuda", dtype=torch.float32, generator=g)
    got = torch.ops.fxpr.compute_per_row_scale(x, 1e-12)

    expected = (x.abs().amax(dim=-1) / 127.0).clamp(min=1e-12).to(torch.float16)
    assert torch.equal(got, expected)


@requires_cuda
def test_per_row_scale_zero_row_uses_eps_floor():
    x = torch.zeros((4, 32), device="cuda", dtype=torch.float32)
    eps = 1.0 / 256.0  # representable in fp16
    got = torch.ops.fxpr.compute_per_row_scale(x, eps)
    expected = torch.full(
        (4,), eps, device="cuda", dtype=torch.float16
    )
    assert torch.equal(got, expected)


@requires_cuda
def test_per_row_scale_split_invariance():
    """Splitting rows across launches must not change the scale.

    This is the load-bearing property for tier 2 deterministic int8 MMA.
    """
    g = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn((16, 128), device="cuda", dtype=torch.float32, generator=g)

    full = torch.ops.fxpr.compute_per_row_scale(x, 1e-12)

    # Run on row halves separately and concatenate.
    s_top = torch.ops.fxpr.compute_per_row_scale(x[:8].contiguous(), 1e-12)
    s_bot = torch.ops.fxpr.compute_per_row_scale(x[8:].contiguous(), 1e-12)
    split = torch.cat([s_top, s_bot], dim=0)

    assert torch.equal(full, split)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_per_row_scale_supports_input_dtypes(dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((4, 32), device="cuda", dtype=torch.float32, generator=g).to(dtype)
    got = torch.ops.fxpr.compute_per_row_scale(x, 1e-12)
    assert got.dtype == torch.float16
    assert got.shape == (4,)
