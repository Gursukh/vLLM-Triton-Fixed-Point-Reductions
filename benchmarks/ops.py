from __future__ import annotations

from typing import Any, Callable

import torch

import fxpr_vllm.library_ops  # noqa: F401


H = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
FFN = 3072
VOCAB = 152000
PAGE_SIZE = 16

_INT_BYTES = {16: 2, 32: 4, 64: 8}
_INT_DTYPE = {16: torch.int16, 32: torch.int32, 64: torch.int64}
_FRAC_BITS = 16


def _build_cast_f2i(shape, cfg):
    n = shape["numel"]
    ib = cfg["int_bits"]
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    def fn():
        return torch.ops.fxpr.float_to_fixed(x, ib, _FRAC_BITS)

    return fn, n, n * (4 + _INT_BYTES[ib])


def _build_cast_i2f(shape, cfg):
    n = shape["numel"]
    ib = cfg["int_bits"]
    info = torch.iinfo(_INT_DTYPE[ib])
    x = torch.randint(info.min // 4, info.max // 4, (n,),
                      device="cuda", dtype=_INT_DTYPE[ib])

    def fn():
        return torch.ops.fxpr.fixed_to_float(x, 32, _FRAC_BITS)

    return fn, n, n * (_INT_BYTES[ib] + 4)


def _build_rms_norm(shape, cfg):
    N, Hd = shape["N"], shape["H"]
    ib = cfg["int_bits"]
    x = torch.randn(N, Hd, device="cuda", dtype=torch.float32)
    w = torch.randn(Hd, device="cuda", dtype=torch.float32)

    def fn():
        return torch.ops.fxpr.rms_norm_fxp(x, w, 1e-5, ib, _FRAC_BITS)

    return fn, 4 * N * Hd, 2 * N * Hd * 4 + Hd * 4


def _build_rms_norm_residual(shape, cfg):
    N, Hd = shape["N"], shape["H"]
    ib = cfg["int_bits"]
    x = torch.randn(N, Hd, device="cuda", dtype=torch.float32)
    w = torch.randn(Hd, device="cuda", dtype=torch.float32)
    res_init = torch.randn(N, Hd, device="cuda", dtype=torch.float32)
    res = res_init.clone()

    # res is mutated in place; reset every call.
    def fn():
        res.copy_(res_init)
        return torch.ops.fxpr.rms_norm_fxp_residual(x, res, w, 1e-5, ib, _FRAC_BITS)

    return fn, 5 * N * Hd, 3 * N * Hd * 4 + Hd * 4


def _build_log_softmax(shape, cfg):
    N, V = shape["N"], shape["V"]
    ib = cfg["int_bits"]
    x = torch.randn(N, V, device="cuda", dtype=torch.float32)

    def fn():
        return torch.ops.fxpr.log_softmax_fxp(x, ib, _FRAC_BITS)

    return fn, 5 * N * V, 2 * N * V * 4


def _build_gemm(shape, cfg):
    M, N, K = shape["M"], shape["N"], shape["K"]
    ib = cfg["int_bits"]
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    def fn():
        return torch.ops.fxpr.gemm_fxp(a, b, bias, ib, _FRAC_BITS)

    return fn, 2 * M * N * K, (M * K + K * N + M * N + N) * 4


def _build_attn(shape, cfg):
    q_t = shape["q"]
    ctx = shape["ctx"]
    heads = shape["heads"]
    kv_heads = shape["kv_heads"]
    head_dim = shape["head_dim"]
    page_size = shape.get("page_size", PAGE_SIZE)
    is_causal = shape.get("kind") == "prefill"
    ib = cfg["int_bits"]
    num_kv_splits = cfg.get("num_kv_splits", 8)
    dtype = torch.float16

    num_blocks_needed = (ctx + page_size - 1) // page_size
    num_blocks = num_blocks_needed + 4

    q = torch.randn(q_t, heads, head_dim, device="cuda", dtype=dtype) * 0.02
    kv_cache = torch.randn(num_blocks, 2, page_size, kv_heads, head_dim,
                           device="cuda", dtype=dtype) * 0.02
    o = torch.zeros(q_t, heads, head_dim, device="cuda", dtype=torch.dtype)
    qsl = torch.tensor([0, q_t], dtype=torch.int32, device="cuda")
    sl = torch.tensor([ctx], dtype=torch.int32, device="cuda")
    bt = torch.arange(num_blocks_needed, dtype=torch.int32,
                      device="cuda").view(1, num_blocks_needed)

    def fn():
        torch.ops.fxpr.unified_attention_fxp(
            q, kv_cache, o, qsl, sl, bt, q_t,
            None, is_causal, None,
            ib, _FRAC_BITS, 0.0, 0, num_kv_splits,
        )
        return o

    flops = 2 * (2 * heads * q_t * ctx * head_dim) + 5 * heads * q_t * ctx
    bytes_ = (
        q_t * heads * head_dim * dtype.itemsize
        + 2 * ctx * kv_heads * head_dim * dtype.itemsize
        + q_t * heads * head_dim * 4
    )
    return fn, flops, bytes_


OPS: dict[str, dict[str, Any]] = {
    "cast_f2i": {
        "build": _build_cast_f2i,
        "shapes": [
            {"numel": 1024 * H, "label": "prefill_1k_act"},
            {"numel": H, "label": "decode_act"},
        ],
        "cache_mode": "cold",
        "ncu_regex": r"float_to_fixed_kernel",
    },
    "cast_i2f": {
        "build": _build_cast_i2f,
        "shapes": [
            {"numel": 1024 * H, "label": "prefill_1k_act"},
            {"numel": H, "label": "decode_act"},
        ],
        "cache_mode": "cold",
        "ncu_regex": r"fixed_to_float_kernel",
    },
    "rms_norm": {
        "build": _build_rms_norm,
        "shapes": [
            {"N": 1024, "H": H, "label": "prefill_1k"},
            {"N": 1, "H": H, "label": "decode"},
        ],
        "cache_mode": "cold",
        "ncu_regex": r"rms_norm_kernel",
    },
    "rms_norm_residual": {
        "build": _build_rms_norm_residual,
        "shapes": [
            {"N": 1024, "H": H, "label": "prefill_1k"},
            {"N": 1, "H": H, "label": "decode"},
        ],
        "cache_mode": "cold",
        "ncu_regex": r"rms_norm",
    },
    "log_softmax": {
        "build": _build_log_softmax,
        "shapes": [
            {"N": 1, "V": VOCAB, "label": "decode_1"},
            {"N": 32, "V": VOCAB, "label": "decode_32"},
        ],
        "cache_mode": "cold",
        "ncu_regex": r"log_softmax",
    },
    "gemm": {
        "build": _build_gemm,
        "shapes": [
            {"M": 1024, "N": N_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM,
             "K": H, "label": "qkv_proj"},
            {"M": 1024, "N": H, "K": N_HEADS * HEAD_DIM, "label": "o_proj"},
            {"M": 1024, "N": 2 * FFN, "K": H, "label": "gate_up_proj"},
            {"M": 1024, "N": H, "K": FFN, "label": "down_proj"},
        ],
        "cache_mode": "warm",
        "ncu_regex": r"gemm_fxp",
    },
    "attn": {
        "build": _build_attn,
        "shapes": [
            {"q": 1024, "ctx": 1024, "kind": "prefill",
             "heads": N_HEADS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
             "label": "prefill_1k"},
            {"q": 1, "ctx": 4096, "kind": "decode",
             "heads": N_HEADS, "kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
             "label": "decode_4k"},
        ],
        "cache_mode": "cold",
        "ncu_regex": r"attn_(split_max|split_dv|combine)_kernel",
    },
}


def build_one(op_name: str, shape: dict, config: dict) -> tuple[Callable, int, int]:
    return OPS[op_name]["build"](shape, config)


def list_ops() -> list[str]:
    return list(OPS.keys())
