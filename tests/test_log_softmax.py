import pytest
import torch

import fxpr_vllm._cuda  # noqa: F401
from fxpr_vllm.sampling import deterministic_log_softmax
from tests.fixed_point_helpers import requires_cuda


@requires_cuda
def test_log_softmax_deterministic_across_runs():
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
    g = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((2, 256), device="cuda", dtype=torch.float32, generator=g)

    got = deterministic_log_softmax(logits)
    ref = torch.log_softmax(logits, dim=-1)

    assert torch.allclose(got, ref, atol=5e-3, rtol=5e-3), (
        f"max error = {(got - ref).abs().max().item()}"
    )


_DTYPE_TOL = {
    torch.float32: (5e-3, 5e-3),
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (5e-2, 5e-2),
}


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log_softmax_native_dtype(dtype):
    g = torch.Generator(device="cuda").manual_seed(7)
    logits_f32 = torch.randn((2, 256), device="cuda", dtype=torch.float32, generator=g)
    logits = logits_f32.to(dtype)

    got = deterministic_log_softmax(logits)
    assert got.dtype == torch.float32, f"output dtype {got.dtype} != torch.float32"

    ref = torch.log_softmax(logits.to(torch.float32), dim=-1)

    atol, rtol = _DTYPE_TOL[dtype]
    assert torch.allclose(got, ref, atol=atol, rtol=rtol), (
        f"max error = {(got - ref).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("dim", [-1, 0])
def test_log_softmax_dim_handling(dim):
    g = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn((8, 32), device="cuda", dtype=torch.float32, generator=g)

    a = deterministic_log_softmax(logits, dim=dim)
    b = deterministic_log_softmax(logits, dim=dim)
    assert torch.equal(a, b)
