import pytest
import torch

import fxpr_vllm._cuda  # noqa: F401  (registers torch.ops.fxpr.*)
from fxpr_vllm.sampling import deterministic_log_softmax
from tests.fixed_point_helpers import requires_cuda


@requires_cuda
def test_log_softmax_deterministic_across_runs():
    """deterministic_log_softmax must be bitwise-stable across runs."""
    g = torch.Generator(device="cuda").manual_seed(123)
    logits = torch.randn((4, 1024), device="cuda", dtype=torch.float32, generator=g)

    first = deterministic_log_softmax(logits)
    for _ in range(4):
        again = deterministic_log_softmax(logits)
        assert torch.equal(first, again), (
            f"Non-deterministic log_softmax: max diff "
            f"{(first - again).abs().max().item()}"
        )


@requires_cuda
def test_log_softmax_matches_torch_log_softmax():
    """Output should be close to torch.log_softmax for typical logits."""
    g = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((2, 256), device="cuda", dtype=torch.float32, generator=g)

    got = deterministic_log_softmax(logits)
    ref = torch.log_softmax(logits, dim=-1)

    assert torch.allclose(got, ref, atol=5e-3, rtol=5e-3), (
        f"max error = {(got - ref).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("dim", [-1, 0])
def test_log_softmax_dim_handling(dim):
    """Reducing over a non-last dim should still be deterministic."""
    g = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn((8, 32), device="cuda", dtype=torch.float32, generator=g)

    a = deterministic_log_softmax(logits, dim=dim)
    b = deterministic_log_softmax(logits, dim=dim)
    assert torch.equal(a, b)
