import torch
import triton
import triton.language as tl

from .fixed_point import fixed_to_float, float_to_fixed


# =====================================================================
# Tier 2: int16 fxp via 4-way int8 MMA emulation (tensor-core path).
# =====================================================================
#
# The scalar / 3D-broadcast paths above quantise each per-element product
# in fp32 then integer-sum. Determinism-preserving but bypasses tensor
# cores. The kernels below compute the same K-sum using *integer* tensor
# core MMAs (int8 × int8 → int32, exact integer reduction), recovering
# tensor-core throughput while keeping the project's invariant:
#
#   1. Operand quantisation uses a STATIC scale (2^Q_FRAC_BITS), the same
#      regardless of how K is partitioned across SMs / warps / KV splits
#      → cross-split bit identity holds unconditionally.
#   2. The MMA reduction is exact integer addition → associative, no fp32
#      reduction inside the dot.
#   3. Each operand is split into a balanced (signed-int8) hi/lo pair so
#      the int16-precision multiply a * b is recovered exactly via
#      four int8 MMAs:
#          a · b = (a_hi·256 + a_lo)·(b_hi·256 + b_lo)
#                = 65536·(a_hi·b_hi) + 256·(a_hi·b_lo + a_lo·b_hi) + a_lo·b_lo
#      Each ·-product is a separate tl.dot. The linear combination is
#      done in int64.
#   4. Cross-K-block accumulation is in int64 at the unscaled
#      2·Q_FRAC_BITS scale; the right shift to FRAC_BITS happens *once*
#      at the end, after all K is summed → identical truncation
#      regardless of how K is split.


@triton.jit
def _quantize_split_int8(x_fp, Q_FRAC_BITS: tl.constexpr):
    """Quantise x_fp to int16 fxp at scale 2**Q_FRAC_BITS and split
    each value into a balanced (hi, lo) int8 pair such that
    int16_value == hi * 256 + lo with lo ∈ [-128, 127] and
    hi ∈ [-128, 127].

    The clamp range is [-32639, 32639] (slightly inside int16) so the
    balanced split never carries hi to 128 (which doesn't fit in int8).
    """
    Q_SCALE: tl.constexpr = float(1 << Q_FRAC_BITS)
    INT16_LIMIT: tl.constexpr = 32639  # 127*256 + 127
    x_scaled = x_fp.to(tl.float32) * Q_SCALE
    x_int = tl.minimum(
        tl.maximum(
            tl.extra.libdevice.rint(x_scaled).to(tl.int32),
            -INT16_LIMIT,
        ),
        INT16_LIMIT,
    )
    # Balanced decomposition: take the low 8 bits as a signed int8 first,
    # then derive hi so that hi * 256 + lo == x_int exactly.
    x_lo_uint = x_int & 0xFF  # in [0, 255]
    x_lo = tl.where(x_lo_uint >= 128, x_lo_uint - 256, x_lo_uint).to(tl.int8)
    x_hi = ((x_int - x_lo.to(tl.int32)) >> 8).to(tl.int8)
    return x_hi, x_lo


@triton.jit
def dot_chunk_fxp_mma_int16(
    a_fp,
    b_fp,
    Q_FRAC_BITS: tl.constexpr,
):
    """Dot product of two fp tiles via 4-way int8 tensor-core MMA.

    Returns an int64 [M, N] tile holding Σ_d a_int16[m,d] * b_int16[d,n]
    at scale 2**(2*Q_FRAC_BITS). The caller is responsible for shifting
    to the desired output scale and casting to FXP_DTYPE.

    Determinism: every per-element value is an exact integer multiply of
    two clamped quantised int16s; the reduction inside tl.dot is an
    exact int32 integer sum (associative). The 65536/256/1 linear
    combination of the four int32 partials is exact in int64.
    """
    a_hi, a_lo = _quantize_split_int8(a_fp, Q_FRAC_BITS)
    b_hi, b_lo = _quantize_split_int8(b_fp, Q_FRAC_BITS)

    hh = tl.dot(a_hi, b_hi, out_dtype=tl.int32).to(tl.int64)
    hl = tl.dot(a_hi, b_lo, out_dtype=tl.int32).to(tl.int64)
    lh = tl.dot(a_lo, b_hi, out_dtype=tl.int32).to(tl.int64)
    ll = tl.dot(a_lo, b_lo, out_dtype=tl.int32).to(tl.int64)

    return hh * 65536 + (hl + lh) * 256 + ll


@triton.jit
def gemm_fxp_mma_kernel(
    a_row_ptrs,
    b_col_ptrs,
    stride_a_k,
    stride_b_k,
    row_mask,
    col_mask,
    Lk,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    Q_FRAC_BITS: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    """Tensor-core GEMM device function. Sums in int64 across K-blocks at
    scale 2**(2*Q_FRAC_BITS), then shifts once to FRAC_BITS and casts
    to FXP_DTYPE before the final fixed_to_float."""
    acc_int64 = tl.zeros([ROWS, COLS], dtype=tl.int64)

    for k_start in tl.range(0, Lk, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_offs < Lk

        a = tl.load(
            a_row_ptrs[:, None] + k_offs[None, :] * stride_a_k,
            mask=row_mask[:, None] & k_valid[None, :],
            other=0.0,
        )
        b = tl.load(
            b_col_ptrs[None, :] + k_offs[:, None] * stride_b_k,
            mask=k_valid[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc_int64 += dot_chunk_fxp_mma_int16(a, b, Q_FRAC_BITS)

    SHIFT_R: tl.constexpr = 2 * Q_FRAC_BITS - FRAC_BITS
    if SHIFT_R > 0:
        acc_fxp = (acc_int64 >> SHIFT_R).to(FXP_DTYPE)
    elif SHIFT_R < 0:
        acc_fxp = (acc_int64 << (-SHIFT_R)).to(FXP_DTYPE)
    else:
        acc_fxp = acc_int64.to(FXP_DTYPE)

    return fixed_to_float(acc_fxp, FRAC_BITS, tl.float32)


@triton.jit
def gemm_fxp_mma(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    Q_FRAC_BITS: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    """Tensor-core fxp GEMM. Same grid swizzle as :func:`gemm_fxp`; replaces
    the inner dot helper with the 4-way int8 MMA emulation."""
    GROUP_SIZE_M: tl.constexpr = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    a_row_ptrs = a_ptr + offs_m * stride_am
    b_col_ptrs = b_ptr + offs_n * stride_bn

    c = gemm_fxp_mma_kernel(
        a_row_ptrs=a_row_ptrs,
        b_col_ptrs=b_col_ptrs,
        stride_a_k=stride_ak,
        stride_b_k=stride_bk,
        row_mask=row_mask,
        col_mask=col_mask,
        Lk=K,
        ROWS=BLOCK_SIZE_M,
        COLS=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
        Q_FRAC_BITS=Q_FRAC_BITS,
        FRAC_BITS=FRAC_BITS,
        FXP_DTYPE=FXP_DTYPE,
    )

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def _require_int8_mma_capable() -> None:
    """Tier 2 needs int8 tensor-core MMA. Triton 3.6's
    TritonGPUAccelerateMatmul pass crashes on int8 inputs for sm_75
    (Turing); the path is reliable from sm_80 (Ampere) onwards.
    """
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        raise NotImplementedError(
            f"Tier 2 (int8-MMA) GEMM requires compute capability >= 8.0; "
            f"got {cap[0]}.{cap[1]}. Triton's int8 tl.dot path is unreliable "
            f"on sm_75; use the scalar/3D-broadcast path instead."
        )


def launch_gemm_fxp_mma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    q_frac_bits: int = 8,
    frac_bits: int = 16,
    fxp_int_bits: int = 32,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Launch the tensor-core fxp GEMM. Test-side helper.

    q_frac_bits controls the operand-side quantisation scale
    (2**q_frac_bits). For typical LLM activations / weights bounded
    well within ±100, q_frac_bits=8 (range ±127, resolution 1/256) is
    a reasonable default. frac_bits is the output scale (matches the
    project's runtime FRAC_BITS).

    Raises:
        NotImplementedError: on GPUs with compute capability below 8.0.
            See :func:`_require_int8_mma_capable`.
    """
    _require_int8_mma_capable()
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.float32, torch.bfloat16)

    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    fxp_dtype = {16: tl.int16, 32: tl.int32, 64: tl.int64}[fxp_int_bits]
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    gemm_fxp_mma[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        Q_FRAC_BITS=q_frac_bits,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
    )
    return c


@triton.jit
def dot_chunk_fxp_ptr(
    a_row_ptrs,
    b_col_ptrs,
    stride_a_d,
    stride_b_d,
    row_mask,
    col_mask,
    D,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    """Determinism-preserving dot product where both operands live in memory.

    For each runtime d in [0, D):
      * load a 1-D column of A (shape [M]) from a_row_ptrs + d * stride_a_d
      * load a 1-D row of B (shape [N]) from b_col_ptrs + d * stride_b_d
      * form the per-element fp32 outer product, quantise, and integer-add
        into the accumulator.

    The K-loop is a *runtime* tl.range (not tl.static_range), so the
    compiled basic block is small — Triton emits the body once and iterates
    at runtime. This drops kernel compile time from minutes to seconds while
    keeping the cross-split / cross-warp determinism guarantee: the integer
    accumulator is associative, and each per-element product is quantised
    before any add.
    """
    M: tl.constexpr = row_mask.shape[0]
    N: tl.constexpr = col_mask.shape[0]
    acc = tl.zeros([M, N], dtype=FXP_DTYPE)
    for d in tl.range(D, loop_unroll_factor=4):
        a_col = tl.load(
            a_row_ptrs + d * stride_a_d, mask=row_mask, other=0.0
        ).to(tl.float32)
        b_row = tl.load(
            b_col_ptrs + d * stride_b_d, mask=col_mask, other=0.0
        ).to(tl.float32)
        outer = a_col[:, None] * b_row[None, :]
        acc += float_to_fixed(outer, FRAC_BITS, FXP_DTYPE)
    return acc


@triton.jit
def dot_chunk_fxp_tile(a, b, FRAC_BITS: tl.constexpr, FXP_DTYPE: tl.constexpr):
    """Determinism-preserving dot product for register-resident tiles.

    Used by the pv-dot in attention, where attention_weights is computed
    in registers and has no memory backing. Falls back to the tile-selector
    pattern (tl.where + tl.sum to extract column d) but inside a
    tl.range runtime loop rather than tl.static_range — same compile
    speedup as dot_chunk_fxp_ptr, paying an O(M*D + D*N) per-iteration
    selector cost in exchange for not needing memory pointers.
    """
    D: tl.constexpr = a.shape[1]
    M: tl.constexpr = a.shape[0]
    N: tl.constexpr = b.shape[1]
    d_arange: tl.constexpr = tl.arange(0, D)
    acc = tl.zeros([M, N], dtype=FXP_DTYPE)
    for d in tl.range(D, loop_unroll_factor=4):
        sel = d_arange == d
        a_col = tl.sum(tl.where(sel[None, :], a, 0.0), 1)
        b_row = tl.sum(tl.where(sel[:, None], b, 0.0), 0)
        outer = a_col[:, None] * b_row[None, :]
        acc += float_to_fixed(outer, FRAC_BITS, FXP_DTYPE)
    return acc


@triton.jit
def paged_kv_location(
    page_table,
    stride_page_table_batch,
    batch_index,
    logical_positions,
    mask,
    PAGE_SIZE: tl.constexpr,
):
    """Translate logical token positions into (physical_block, in_block_offset)."""
    # Compute the logical block and in-block offset
    logical_block = logical_positions // PAGE_SIZE
    in_block = logical_positions % PAGE_SIZE

    # Load the physical block from the page table
    physical_block = tl.load(
        page_table + batch_index * stride_page_table_batch + logical_block,
        mask=mask,
        other=0,
    ).to(tl.int64)
    return physical_block, in_block


@triton.jit
def gemm_fxp_kernel(
    a_row_ptrs,
    b_col_ptrs,
    stride_a_k,
    stride_b_k,
    row_mask,
    col_mask,
    Lk,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    # Integer accumulator: K-sums are exact across permutations of K because
    # every per-element product is quantised before any addition.
    acc = tl.zeros([ROWS, COLS], dtype=FXP_DTYPE)

    for k_start in tl.range(0, Lk, D_CHUNK):
        # Number of valid K elements in this chunk (handles non-multiple Lk).
        d = tl.minimum(D_CHUNK, Lk - k_start)
        acc += dot_chunk_fxp_ptr(
            a_row_ptrs=a_row_ptrs + k_start * stride_a_k,
            b_col_ptrs=b_col_ptrs + k_start * stride_b_k,
            stride_a_d=stride_a_k,
            stride_b_d=stride_b_k,
            row_mask=row_mask,
            col_mask=col_mask,
            D=d,
            FRAC_BITS=FRAC_BITS,
            FXP_DTYPE=FXP_DTYPE,
        )

    return fixed_to_float(acc, FRAC_BITS, tl.float32)


@triton.jit
def gemm_fxp(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    # Grid and Swizzle Logic
    GROUP_SIZE_M: tl.constexpr = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block Pointers and Masks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    a_row_ptrs = a_ptr + offs_m * stride_am
    b_col_ptrs = b_ptr + offs_n * stride_bn

    # Call the device helper function!
    c = gemm_fxp_kernel(
        a_row_ptrs=a_row_ptrs,
        b_col_ptrs=b_col_ptrs,
        stride_a_k=stride_ak,
        stride_b_k=stride_bk,
        row_mask=row_mask,
        col_mask=col_mask,
        Lk=K,
        ROWS=BLOCK_SIZE_M,
        COLS=BLOCK_SIZE_N,
        D_CHUNK=BLOCK_SIZE_K,
        FRAC_BITS=FRAC_BITS,
        FXP_DTYPE=FXP_DTYPE,
    )

    # Store the resulting block
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)
