import pytest
import torch

import fxpr_vllm._cuda  # noqa: F401  (registers torch.ops.fxpr.*)
from tests.fixed_point_helpers import requires_cuda

_BLOCK_SIZE = 16
_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 32
_SM_SCALE = 1.0 / (_HEAD_DIM**0.5)
_FRAC_BITS = 14


def _build_decode_inputs(seq_lens, num_blocks_per_seq):
    """Build (q, kv_cache, query_start_loc, seq_lens, block_table) for a
    decode batch. Each request has query_len=1; each seq spans
    num_blocks_per_seq logical blocks of size _BLOCK_SIZE."""
    g = torch.Generator(device="cuda").manual_seed(2026)
    num_requests = len(seq_lens)
    total_blocks = num_requests * num_blocks_per_seq

    q = torch.randn(
        (num_requests, _NUM_HEADS, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
        generator=g,
    )
    # kv_cache shape: (num_blocks, 2, page_size, num_kv_heads, head_dim).
    kv_cache = torch.randn(
        (total_blocks, 2, _BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
        generator=g,
    )

    query_start_loc = torch.arange(
        num_requests + 1, device="cuda", dtype=torch.int32
    )
    seq_lens_t = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    # Each request gets a contiguous slab of physical blocks.
    block_table = torch.arange(
        total_blocks, device="cuda", dtype=torch.int32
    ).reshape(num_requests, num_blocks_per_seq)

    return q, kv_cache, query_start_loc, seq_lens_t, block_table


def _run_decode(q, kv_cache, query_start_loc, seq_lens, block_table, num_kv_splits):
    o = torch.empty_like(q)
    torch.ops.fxpr.unified_attention_fxp(
        q,
        kv_cache,
        o,
        query_start_loc,
        seq_lens,
        block_table,
        1,  # max_query_len (decode)
        None,  # alibi_slopes
        True,  # is_causal
        float(_SM_SCALE),
        _FRAC_BITS,
        32,  # fxp_int_bits
        0.0,  # logit_softcap
        0,  # window_size
        int(num_kv_splits),
    )
    return o


@requires_cuda
@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_decode_invariant_across_kv_splits(num_kv_splits):
    """Decode (block_size=16, multi-block) must be bitwise-identical across kv_splits."""
    seq_lens = [48, 48]  # 3 blocks per seq
    q, kv_cache, qsl, sl, bt = _build_decode_inputs(seq_lens, num_blocks_per_seq=3)

    o_ref = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=1)
    o_got = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=num_kv_splits)

    assert torch.equal(o_ref, o_got), (
        f"Decode output diverged at num_kv_splits={num_kv_splits}, "
        f"max diff = {(o_ref - o_got).abs().max().item()}"
    )


@requires_cuda
def test_decode_deterministic_across_runs():
    """Same decode inputs produce bitwise identical outputs across runs."""
    seq_lens = [32, 48, 16]
    q, kv_cache, qsl, sl, bt = _build_decode_inputs(seq_lens, num_blocks_per_seq=3)

    first = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=4)
    for _ in range(4):
        again = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=4)
        assert torch.equal(first, again), (
            f"Non-deterministic decode: max diff = {(first - again).abs().max().item()}"
        )
