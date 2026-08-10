"""FlashAttention-2 backward pass in Triton.

Built up incrementally, mirroring the forward file:

* `_bwd_dv_kernel` -- gradient with respect to V. Needs only the recomputed
  probabilities, so it is where the recomputation strategy is proven.
* `_bwd_preprocess_kernel` -- D_i = rowsum(dO * O), the softmax-Jacobian term.
* `_bwd_dk_kernel` / `_bwd_dq_kernel` -- gradients w.r.t. K and Q. They need
  separate kernels because dK is parallel over K blocks while dQ is parallel
  over Q blocks; fusing them would require atomics on one side.

The backward pass never stores the N x N probability matrix. It rebuilds each
block from Q, K and the saved log-sum-exp L, since p_ij = exp(s_ij - L_i).
That costs extra matmul FLOPs and saves HBM bandwidth, which is the right
trade on bandwidth-bound hardware.

Note the traversal is transposed relative to the forward pass. Forward
parallelizes over Q blocks with K/V streaming inside; dV accumulates into a
fixed K/V block across all Q blocks, so backward parallelizes over K/V blocks
with Q streaming inside.

A note on `num_stages`. Triton pipelines the inner loop by prefetching several
copies of the streamed tiles. At HEAD_DIM=128 each K or V tile is 16 KB, so the
default three stages of both is already 98 KB -- and these kernels also hold q,
do and an fp32 accumulator live. That overruns sm_86's ~100 KB of shared memory
per SM, so large head dims drop to two stages. Cards with more shared memory
(A100 164 KB, H100 228 KB) would not need this.

Importing this module requires Triton, which is Linux/GPU-only.
"""

import math

import torch
import triton
import triton.language as tl

from flash_attn.triton_fwd import SUPPORTED_DTYPES


@triton.jit
def _bwd_dv_kernel(
    Q,
    K,
    DO,
    LSE,
    DV,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    H,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """dV = P^T dO, accumulated over every Q block.

    One program owns one K/V block and streams all relevant Q blocks past it.
    P is recomputed each iteration as exp(s - L), never loaded.
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    off_h_kv = off_h // GROUP_SIZE

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h_kv * stride_kh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    l_base = LSE + off_b * stride_lb + off_h * stride_lh
    # Written per QUERY head. Contributions from the heads sharing a K/V head
    # are summed afterwards, outside the kernel.
    dv_base = DV + off_b * stride_dvb + off_h * stride_dvh

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # This block's keys stay resident; Q streams past them.
    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Mirror of the forward's causal bound: query i sees key j only when i >= j,
    # so the earliest query that can see this block is row start_n * BLOCK_N.
    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Out-of-range rows get L = +inf, so exp(s - L) is exactly 0 and they
        # contribute nothing -- no separate masking of p needed for them.
        l_i = tl.load(l_base + offs_m * stride_lm, mask=offs_m < N_CTX, other=float("inf"))

        # Recomputation: rebuild this block of P from Q, K and L.
        s = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(s - l_i[:, None])

        if IS_CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        # (BLOCK_N, BLOCK_M) @ (BLOCK_M, HEAD_DIM) -> (BLOCK_N, HEAD_DIM)
        dv += tl.dot(tl.trans(p).to(do.dtype), do)

    dv_ptrs = dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N_CTX)


def backward_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Gradient of the attention output with respect to V.

    Args:
        q, k: (B, H, N, D) forward inputs.
        do: (B, H, N, D) gradient flowing into the attention output.
        lse: (B, H, N) log-sum-exp saved by the forward pass.
        causal: must match the forward pass.
        sm_scale: must match the forward pass. Defaults to 1/sqrt(D).

    Returns:
        (B, H, N, D) gradient with respect to V, in the dtype of `do`.

    V itself is not needed: dV = P^T dO depends only on the probabilities.
    """
    if q.shape != do.shape:
        raise ValueError(f"q and do must share a shape, got {q.shape}, {do.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(f"q and k may differ only in head count, got {q.shape}, {k.shape}")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"query heads must be a multiple of kv heads, got {q.shape[1]} and {k.shape[1]}"
        )
    if do.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"only fp16 and bf16 are supported, got {do.dtype}")
    b, h, n, d = q.shape
    group_size = h // k.shape[1]
    if lse.shape != (b, h, n):
        raise ValueError(f"lse must be (B, H, N) = {(b, h, n)}, got {tuple(lse.shape)}")
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    dv, group_shape = _grouped_output(q, do, group_size)

    # Parallel over K/V blocks -- the transpose of the forward grid. Under GQA
    # there is one program per QUERY head, so each writes its own slot.
    grid = (triton.cdiv(n, block_n), b * h)

    _bwd_dv_kernel[grid](
        q,
        k,
        do,
        lse,
        dv,
        *q.stride(),
        *k.stride(),
        *do.stride(),
        *lse.stride(),
        *dv.stride(),
        h,
        n,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        num_warps=8 if d >= 128 else 4,
        num_stages=2 if d >= 128 else 3,
    )
    return _collapse_groups(dv, group_shape, do.dtype)


@triton.jit
def _bwd_preprocess_kernel(
    O,
    DO,
    DELTA,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dm,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Precompute D_i = rowsum(dO * O).

    The softmax Jacobian needs sum_k p_ik dP_ik, which looks like it requires a
    full row of P. It does not:

        sum_k p_ik dP_ik = sum_k p_ik (dO_i . v_k) = dO_i . (sum_k p_ik v_k)
                         = dO_i . O_i

    So the quantity is a plain row-wise dot product of two tensors we already
    have, costing O(N*D). This is the identity that makes the backward pass fit
    in the same memory budget as the forward.

    It gets its own kernel because every Q block and every K block needs it.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    o_base = O + off_b * stride_ob + off_h * stride_oh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    d_base = DELTA + off_b * stride_db + off_h * stride_dh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod

    o = tl.load(o_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0).to(tl.float32)

    tl.store(d_base + offs_m * stride_dm, tl.sum(o * do, axis=1), mask=offs_m < N_CTX)


@triton.jit
def _bwd_dk_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    H,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """dK = sm_scale * dS^T Q, with dS = P * (dP - D).

    Parallel over K blocks with Q streaming inside -- the same traversal as
    dV, because both accumulate into a fixed K block.
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    off_h_kv = off_h // GROUP_SIZE

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h_kv * stride_kh
    v_base = V + off_b * stride_vb + off_h_kv * stride_vh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    l_base = LSE + off_b * stride_lb + off_h * stride_lh
    d_base = DELTA + off_b * stride_lb + off_h * stride_lh
    dk_base = DK + off_b * stride_dkb + off_h * stride_dkh

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        l_i = tl.load(l_base + offs_m * stride_lm, mask=offs_m < N_CTX, other=float("inf"))
        delta_i = tl.load(d_base + offs_m * stride_lm, mask=offs_m < N_CTX, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(s - l_i[:, None])
        if IS_CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        # dP = dO V^T, then the softmax Jacobian. Kept in fp32: the subtraction
        # cancels when dP is close to D, which amplifies relative error.
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta_i[:, None])

        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

    dk = dk * sm_scale
    dk_ptrs = dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < N_CTX)


@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    H,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """dQ = sm_scale * dS K, with dS = P * (dP - D).

    Parallel over Q blocks with K/V streaming inside -- the same traversal as
    the forward pass, because dQ accumulates into a fixed Q block. That is why
    dQ needs its own kernel rather than sharing dK's.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    off_h_kv = off_h // GROUP_SIZE

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h_kv * stride_kh
    v_base = V + off_b * stride_vb + off_h_kv * stride_vh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    l_base = LSE + off_b * stride_lb + off_h * stride_lh
    d_base = DELTA + off_b * stride_lb + off_h * stride_lh
    dq_base = DQ + off_b * stride_dqb + off_h * stride_dqh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    l_i = tl.load(l_base + offs_m * stride_lm, mask=offs_m < N_CTX, other=float("inf"))
    delta_i = tl.load(d_base + offs_m * stride_lm, mask=offs_m < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(s - l_i[:, None])
        p = tl.where(offs_n[None, :] < N_CTX, p, 0.0)
        if IS_CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta_i[:, None])

        dq += tl.dot(ds.to(k.dtype), k)

    dq = dq * sm_scale
    dq_ptrs = dq_base + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def backward_preprocess(o: torch.Tensor, do: torch.Tensor, block_m: int = 64) -> torch.Tensor:
    """D_i = rowsum(dO * O), the softmax-Jacobian correction term.

    Returns:
        (B, H, N) fp32 tensor.
    """
    if o.shape != do.shape:
        raise ValueError(f"o and do must share a shape, got {o.shape}, {do.shape}")
    b, h, n, d = o.shape

    delta = torch.empty((b, h, n), device=o.device, dtype=torch.float32)
    grid = (triton.cdiv(n, block_m), b * h)

    _bwd_preprocess_kernel[grid](
        o,
        do,
        delta,
        *o.stride(),
        *do.stride(),
        *delta.stride(),
        h,
        n,
        BLOCK_M=block_m,
        HEAD_DIM=d,
    )
    return delta


def _grouped_output(q, do, group_size):
    """Buffer the dK/dV kernels write into, plus how to collapse it.

    With GROUP_SIZE query heads sharing a K/V head, each query head produces a
    partial gradient for that head. One program per query head writes its own
    slot and the slots are summed afterwards. The scratch buffer is (B, H_q, N,
    D) -- the size of Q, linear in N -- so it does not reintroduce quadratic
    memory. A production kernel accumulates in-register instead, at the cost of
    a nested loop that unrolls badly at large group sizes.

    Group size 1 takes the original path exactly: same dtype, same buffer, no
    reduction, so ordinary multi-head attention is untouched.
    """
    if group_size == 1:
        return torch.empty_like(do), None
    b, h_q, n, d = q.shape
    # fp32 scratch: summing up to GROUP_SIZE partials in fp16 would lose
    # precision the rest of the backward pass carefully preserves.
    return torch.empty((b, h_q, n, d), device=q.device, dtype=torch.float32), (
        b,
        h_q // group_size,
        group_size,
        n,
        d,
    )


def _collapse_groups(buf, shape, dtype):
    if shape is None:
        return buf
    return buf.view(*shape).sum(dim=2).to(dtype)


def _check_backward_inputs(q, k, v, do, lse, delta, block_m, block_n):
    if k.shape != v.shape:
        raise ValueError(f"k and v must share a shape, got {k.shape}, {v.shape}")
    if q.shape != do.shape:
        raise ValueError(f"q and do must share a shape, got {q.shape}, {do.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(f"q and k/v may differ only in head count, got {q.shape}, {k.shape}")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"query heads must be a multiple of kv heads, got {q.shape[1]} and {k.shape[1]}"
        )
    if do.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"only fp16 and bf16 are supported, got {do.dtype}")
    b, h, n, d = q.shape
    if lse.shape != (b, h, n) or delta.shape != (b, h, n):
        raise ValueError(f"lse and delta must be (B, H, N) = {(b, h, n)}")
    # The kernels address lse and delta through one set of strides, which is
    # only valid while the two are laid out identically.
    if lse.stride() != delta.stride():
        raise ValueError(f"lse and delta must share strides, got {lse.stride()}, {delta.stride()}")
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")
    return b, h, n, d


def backward_dk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Gradient with respect to K. Returns (B, H, N, D) in the dtype of `do`."""
    b, h, n, d = _check_backward_inputs(q, k, v, do, lse, delta, block_m, block_n)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    group_size = h // k.shape[1]
    dk, group_shape = _grouped_output(q, do, group_size)
    grid = (triton.cdiv(n, block_n), b * h)

    _bwd_dk_kernel[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        *lse.stride(),
        *dk.stride(),
        h,
        n,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        num_warps=8 if d >= 128 else 4,
        num_stages=2 if d >= 128 else 3,
    )
    return _collapse_groups(dk, group_shape, do.dtype)


def backward_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Gradient with respect to Q. Returns (B, H, N, D) in the dtype of `do`."""
    b, h, n, d = _check_backward_inputs(q, k, v, do, lse, delta, block_m, block_n)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    group_size = h // k.shape[1]
    dq = torch.empty_like(do)
    grid = (triton.cdiv(n, block_m), b * h)

    _bwd_dq_kernel[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        *lse.stride(),
        *dq.stride(),
        h,
        n,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        num_warps=8 if d >= 128 else 4,
        num_stages=2 if d >= 128 else 3,
    )
    return dq
