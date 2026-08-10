"""FlashAttention-2 backward pass in Triton.

Built up incrementally, mirroring the forward file:

* `_bwd_dv_kernel` -- gradient with respect to V. Needs only the recomputed
  probabilities, so it is where the recomputation strategy is proven.

The backward pass never stores the N x N probability matrix. It rebuilds each
block from Q, K and the saved log-sum-exp L, since p_ij = exp(s_ij - L_i).
That costs extra matmul FLOPs and saves HBM bandwidth, which is the right
trade on bandwidth-bound hardware.

Note the traversal is transposed relative to the forward pass. Forward
parallelizes over Q blocks with K/V streaming inside; dV accumulates into a
fixed K/V block across all Q blocks, so backward parallelizes over K/V blocks
with Q streaming inside.

Importing this module requires Triton, which is Linux/GPU-only.
"""

import math

import torch
import triton
import triton.language as tl


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
):
    """dV = P^T dO, accumulated over every Q block.

    One program owns one K/V block and streams all relevant Q blocks past it.
    P is recomputed each iteration as exp(s - L), never loaded.
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    l_base = LSE + off_b * stride_lb + off_h * stride_lh
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
    if not (q.shape == k.shape == do.shape):
        raise ValueError(f"q/k/do must share a shape, got {q.shape}, {k.shape}, {do.shape}")
    b, h, n, d = q.shape
    if lse.shape != (b, h, n):
        raise ValueError(f"lse must be (B, H, N) = {(b, h, n)}, got {tuple(lse.shape)}")
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    dv = torch.empty_like(do)

    # Parallel over K/V blocks -- the transpose of the forward grid.
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
        num_warps=8 if d >= 128 else 4,
    )
    return dv
