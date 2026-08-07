"""FlashAttention-2 forward pass in Triton.

Built up incrementally. Right now this holds only the skeleton: a kernel that
loads one Q tile and stores it back unchanged. It computes nothing, but it
exercises the full plumbing -- grid launch, program IDs, strided tile loads,
boundary masking, stores -- so that later steps debug arithmetic rather than
addressing.

Tiles are addressed with explicit pointer arithmetic rather than
`tl.make_block_ptr`. The block-pointer API's main advantage is enabling TMA on
sm_90 (Hopper); on Ampere (sm_86) it buys nothing, and the explicit form keeps
the index math visible.

Importing this module requires Triton, which is Linux/GPU-only. It is
deliberately not imported from `flash_attn/__init__.py`.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_q_kernel(
    Q,
    O,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Copy one (BLOCK_M, HEAD_DIM) tile of Q into O.

    Grid is 2D: axis 0 indexes Q blocks along the sequence, axis 1 indexes
    (batch, head) pairs. That is already the FlashAttention-2 parallelization --
    FA-2's change over FA-1 is precisely that Q blocks run in parallel rather
    than being an inner loop, which is why axis 0 exists.
    """
    start_m = tl.program_id(0)  # which Q block along the sequence
    off_hz = tl.program_id(1)  # flattened (batch, head)

    off_b = off_hz // H
    off_h = off_hz % H

    # Move to this (batch, head)'s slice. Strides are passed in rather than
    # assumed so non-contiguous inputs (e.g. a transposed view) still work.
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    o_base = O + off_b * stride_ob + off_h * stride_oh

    # Row indices this program owns, and the full head dimension.
    # tl.arange requires a power-of-two length, so HEAD_DIM must be one.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Outer-product broadcast into a 2D tile of addresses.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    # The last block is partial whenever N_CTX % BLOCK_M != 0. Masking on load
    # and store is what keeps that case from reading or writing out of bounds.
    mask_m = offs_m[:, None] < N_CTX

    q = tl.load(q_ptrs, mask=mask_m, other=0.0)
    tl.store(o_ptrs, q, mask=mask_m)


def copy_q(q: torch.Tensor, block_m: int = 128) -> torch.Tensor:
    """Return a copy of `q` produced by the Triton skeleton kernel.

    Exists only to validate the launch/addressing path. If this does not return
    a bit-exact copy, nothing built on top of it can be trusted.
    """
    if q.dim() != 4:
        raise ValueError(f"expected 4D (B, H, N, D) tensor, got {q.dim()}D")
    b, h, n, d = q.shape
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    o = torch.empty_like(q)

    # One program per (Q block, batch-head pair).
    grid = (triton.cdiv(n, block_m), b * h)

    _copy_q_kernel[grid](
        q,
        o,
        *q.stride(),
        *o.stride(),
        h,
        n,
        BLOCK_M=block_m,
        HEAD_DIM=d,
    )
    return o
