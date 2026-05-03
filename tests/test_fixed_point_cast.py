import pytest
import torch
from tests.fixed_point_helpers import float_to_fixed, requires_cuda, fixed_to_float


@requires_cuda
def test_float_to_fixed_known_values():
    # frac_bits=16 (FXPR_FRAC_BITS).
    x = torch.tensor(
        [
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            0.25,
            1.5,
            -1.5,
            2**15 - 1,
            -(2**15),
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            0,
            1 << 16,
            -(1 << 16),
            1 << 15,
            -(1 << 15),
            1 << 14,
            (1 << 16) + (1 << 15),
            -((1 << 16) + (1 << 15)),
            (2**15 - 1) << 16,
            -(2**15) << 16,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    got = float_to_fixed(x, torch.int32)
    assert torch.equal(got, expected)


@requires_cuda
def test_float_to_fixed_saturation():
    x = torch.tensor([1e30, -1e30, 0.0], device="cuda", dtype=torch.float32)
    got = float_to_fixed(x, torch.int32)
    imax = torch.iinfo(torch.int32).max
    imin = torch.iinfo(torch.int32).min

    assert imax - 256 <= got[0].item() <= imax
    assert imin <= got[1].item() <= imin + 256
    assert got[2].item() == 0


@requires_cuda
def test_float_roundtrip_lossless():
    # All powers of 2, exact at frac_bits=16.
    base = torch.tensor(
        [0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.125, 2.0, -2.0, 10.0, -10.0],
        device="cuda",
        dtype=torch.float32,
    )
    q = float_to_fixed(base, torch.int32)
    back = fixed_to_float(q, torch.float32)
    assert torch.equal(back, base), f"{base} -> {q} -> {back}"


@requires_cuda
def test_float_roundtrip_rounding():
    # step=2^-16; off-grid inputs get rounded.
    frac_bits = 16
    step = 2**-frac_bits
    x = torch.tensor(
        [1e-6, 3e-6, 1.234567e-3, -1.234567e-3, 0.7777777],
        device="cuda",
        dtype=torch.float32,
    )
    back = fixed_to_float(float_to_fixed(x, torch.int32), torch.float32)

    grid_err = (back / step).round() * step - back
    assert torch.allclose(grid_err, torch.zeros_like(grid_err), atol=1e-7)
    assert torch.all((x - back).abs() <= step / 2 + 1e-7)
    assert torch.any(x != back)


@requires_cuda
def test_fixed_roundtrip_lossless():
    # Small ints round-trip through fp32 at frac_bits=16.
    q = torch.tensor(
        [0, 1, -1, 123, -123, (1 << 20), -(1 << 20)], device="cuda", dtype=torch.int32
    )
    f = fixed_to_float(q, torch.float32)
    back = float_to_fixed(f, torch.int32)
    assert torch.equal(back, q)


@requires_cuda
@pytest.mark.parametrize("frac_bits", [8, 16, 32])
def test_float_roundtrip_lossless_parametrized_frac_bits(frac_bits):
    # Powers of 2 inside the representable range are exact at any frac_bits in
    # {8, 16, 32}. Use int64 so the full set fits even at frac_bits=32 (where
    # 1.0 -> 2^32 would saturate int32).
    base = torch.tensor(
        [0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.125],
        device="cuda",
        dtype=torch.float32,
    )
    q = float_to_fixed(base, torch.int64, fxp_frac_bits=frac_bits)
    back = fixed_to_float(q, torch.float32, fxp_frac_bits=frac_bits)
    assert torch.equal(back, base), (
        f"frac_bits={frac_bits}: {base} -> {q} -> {back}"
    )
