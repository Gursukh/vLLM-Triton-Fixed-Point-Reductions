import pytest
import torch

from tests.fixed_point_helpers import prefill_fxp_test, requires_cuda


def _ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group = num_heads // num_kv_heads
    o = torch.zeros_like(q)

    for b in range(b_seq_len.shape[0]):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()
        for h in range(num_heads):
            kv_h = h // kv_group
            qi = q[start : start + seq_len, h, :]
            ki = k[start : start + seq_len, kv_h, :]
            vi = v[start : start + seq_len, kv_h, :]

            scores = (qi @ ki.T) * sm_scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores.masked_fill_(mask, -1e8)
            p = torch.softmax(scores, dim=-1)
            o[start : start + seq_len, h, :] = p @ vi
    return o


def _make_inputs(batch, seq_lens, num_heads, num_kv_heads, head_dim, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn(
        total, num_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )
    k = torch.randn(
        total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )
    v = torch.randn(
        total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )

    b_seq_len = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    b_start_loc = torch.zeros(batch, device="cuda", dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)

    o = torch.empty_like(q)
    return q, k, v, o, b_start_loc, b_seq_len


_PREFILL_SEQ_LEN = 32
_PREFILL_HEADS = 2
_PREFILL_KV_HEADS = 2
_PREFILL_HEAD_DIM = 32
_PREFILL_SM_SCALE = 1.0 / (_PREFILL_HEAD_DIM**0.5)


@requires_cuda
@pytest.mark.parametrize("batch", [1, 2])
def test_prefill_correctness_causal(batch):
    seq_lens = [_PREFILL_SEQ_LEN] * batch
    q, k, v, o, b_start_loc, b_seq_len = _make_inputs(
        batch, seq_lens, _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM
    )

    prefill_fxp_test(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=_PREFILL_SEQ_LEN,
        is_causal=True,
        softmax_scale=_PREFILL_SM_SCALE,
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=True, sm_scale=_PREFILL_SM_SCALE
    )

    assert torch.allclose(
        o, ref, atol=5e-2, rtol=5e-2
    ), f"max error = {(o - ref).abs().max().item()}"


@requires_cuda
def test_prefill_correctness_non_causal():
    q, k, v, o, b_start_loc, b_seq_len = _make_inputs(
        1, [_PREFILL_SEQ_LEN], _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM
    )

    prefill_fxp_test(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=_PREFILL_SEQ_LEN,
        is_causal=False,
        softmax_scale=_PREFILL_SM_SCALE,
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=False, sm_scale=_PREFILL_SM_SCALE
    )

    assert torch.allclose(
        o, ref, atol=5e-2, rtol=5e-2
    ), f"max error = {(o - ref).abs().max().item()}"


@requires_cuda
def test_prefill_variable_seq_lens():
    seq_lens = [_PREFILL_SEQ_LEN // 2, _PREFILL_SEQ_LEN, _PREFILL_SEQ_LEN * 3 // 4]
    q, k, v, o, b_start_loc, b_seq_len = _make_inputs(
        3, seq_lens, _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM, seed=99
    )

    prefill_fxp_test(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=max(seq_lens),
        is_causal=True,
        softmax_scale=_PREFILL_SM_SCALE,
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=True, sm_scale=_PREFILL_SM_SCALE
    )

    assert torch.allclose(
        o, ref, atol=5e-2, rtol=5e-2
    ), f"max error = {(o - ref).abs().max().item()}"


def _float_attention_row(
    q_row: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    sm_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    seq_len = k_rows.shape[0]
    head_dim = q_row.shape[0]

    q_c = q_row.to(dtype)
    k_c = k_rows.to(dtype)

    scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    for j in range(seq_len):
        dot = torch.zeros((), device="cuda", dtype=dtype)
        for d in range(head_dim):
            dot = dot + q_c[d] * k_c[j, d]
        scores[j] = dot * sm_scale

    scores_max = scores.max()
    exp_scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    exp_sum = torch.zeros((), device="cuda", dtype=dtype)
    for j in range(seq_len):
        exp_scores[j] = torch.exp(scores[j] - scores_max)
        exp_sum = exp_sum + exp_scores[j]
    p = exp_scores / exp_sum

    v_c = v_rows.to(dtype)
    out = torch.zeros(head_dim, device="cuda", dtype=dtype)
    for j in range(seq_len):
        for d in range(head_dim):
            out[d] = out[d] + p[j] * v_c[j, d]
    return out


@requires_cuda
def test_float_accumulation_is_order_dependent():
    seq_len, head_dim = 3, 16
    q_row = torch.ones(head_dim, device="cuda", dtype=torch.float32)
    k_rows = torch.ones(seq_len, head_dim, device="cuda", dtype=torch.float32)

    v_rows = torch.zeros(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v_rows[0] = 1e30
    v_rows[1] = -1e30
    v_rows[2] = 1e-30

    sm_scale = 1.0 / (head_dim**0.5)
    out_fwd = _float_attention_row(q_row, k_rows, v_rows, sm_scale, torch.float32)
    perm = torch.tensor([0, 2, 1], device="cuda")
    out_perm = _float_attention_row(
        q_row, k_rows[perm], v_rows[perm], sm_scale, torch.float32
    )

    assert not torch.equal(out_fwd[perm], out_perm), (
        "fp32 attention should differ with permuted KV order"
    )


@requires_cuda
def test_fixedpoint_prefill_is_permutation_equivariant():
    seq_len = _PREFILL_SEQ_LEN
    g = torch.Generator(device="cuda").manual_seed(7)
    shape = (seq_len, _PREFILL_HEADS, _PREFILL_HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 0.5

    b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)

    o_orig = torch.empty_like(q)
    prefill_fxp_test(
        q,
        k,
        v,
        o_orig,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=_PREFILL_SM_SCALE,
    )

    perm = torch.randperm(seq_len, device="cuda")
    o_perm = torch.empty_like(q)
    prefill_fxp_test(
        q,
        k[perm],
        v[perm],
        o_perm,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=_PREFILL_SM_SCALE,
    )

    assert torch.equal(o_orig, o_perm), (
        f"fxp prefill should be KV-permutation invariant, "
        f"max diff = {(o_orig - o_perm).abs().max().item()}"
    )


@requires_cuda
def test_prefill_deterministic_across_runs():
    q, k, v, _, b_start_loc, b_seq_len = _make_inputs(
        1,
        [_PREFILL_SEQ_LEN],
        _PREFILL_HEADS,
        _PREFILL_KV_HEADS,
        _PREFILL_HEAD_DIM,
        seed=77,
    )

    results = []
    for _ in range(5):
        o = torch.empty_like(q)
        prefill_fxp_test(
            q,
            k,
            v,
            o,
            b_start_loc,
            b_seq_len,
            max_input_len=_PREFILL_SEQ_LEN,
            is_causal=True,
            softmax_scale=_PREFILL_SM_SCALE,
        )
        results.append(o)

    for r in results[1:]:
        assert torch.equal(
            results[0], r
        ), f"Non-deterministic: max diff = {(results[0] - r).abs().max().item()}"
