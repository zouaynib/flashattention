"""FlashAttention-2 forward pass in Triton.

Built up incrementally:

* `_copy_q_kernel` -- the skeleton. Loads one Q tile and stores it back
  unchanged. Computes nothing, but exercises the full plumbing (grid launch,
  program IDs, strided tile loads, boundary masking, stores) so that later
  steps debug arithmetic rather than addressing.
* `_qk_tile_kernel` -- one tile of S = QK^T / sqrt(d). Introduces `tl.dot`
  and the fp32 accumulator.
* `_softmax_tile_kernel` -- safe softmax over that one tile. Introduces the
  max/exp/sum machinery and the -inf masking of out-of-range keys. Not yet
  online: the denominator covers one K block, not the whole row.
* `_running_max_kernel` -- the first online step. Loops over every K block
  tracking a running row max, and introduces the FA-2 structure where Q is
  loaded once and K/V stream past it.

Tiles are addressed with explicit pointer arithmetic rather than
`tl.make_block_ptr`. The block-pointer API's main advantage is enabling TMA on
sm_90 (Hopper); on Ampere (sm_86) it buys nothing, and the explicit form keeps
the index math visible.

Importing this module requires Triton, which is Linux/GPU-only. It is
deliberately not imported from `flash_attn/__init__.py`.
"""

import math

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


@triton.jit
def _qk_tile_kernel(
    Q,
    K,
    S,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sm,
    stride_sn,
    H,
    N_CTX,
    sm_scale,
    start_m,
    start_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Compute a single (BLOCK_M, BLOCK_N) tile of S = QK^T / sqrt(d).

    One program per (batch, head); the tile indices `start_m`/`start_n` are
    runtime arguments so a test can ask for any specific tile.

    On sm_86 `tl.dot` lowers to the tensor-core MMA path, which requires every
    dimension to be at least 16 -- so BLOCK_M, BLOCK_N and HEAD_DIM all have a
    hard floor of 16.
    """
    off_hz = tl.program_id(0)
    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    s_base = S + off_b * stride_sb + off_h * stride_sh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of Q
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # rows of K
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd

    # Out-of-range rows load as 0. A zero row contributes 0 to the dot product,
    # so the garbage stays confined to tile entries the caller already knows
    # are invalid -- it never contaminates a valid entry.
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)  # (BLOCK_M, HEAD_DIM)
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)  # (BLOCK_N, HEAD_DIM)

    # K is stored (N, D) like Q, so transpose to (D, BLOCK_N) for the matmul.
    # tl.dot accumulates in fp32 even though the inputs are fp16: fp16 has only
    # ~3 decimal digits, and summing HEAD_DIM products in fp16 would lose
    # precision that the softmax then amplifies.
    s = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N), fp32
    s = s * sm_scale

    # Tile-local output: the (m, n) tile is written to S[:, :, :BLOCK_M, :BLOCK_N].
    offs_sm = tl.arange(0, BLOCK_M)
    offs_sn = tl.arange(0, BLOCK_N)
    s_ptrs = s_base + offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn
    tl.store(s_ptrs, s)


def qk_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    start_m: int = 0,
    start_n: int = 0,
    block_m: int = 64,
    block_n: int = 64,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute one tile of the score matrix S = QK^T / sqrt(d).

    Args:
        q, k: (B, H, N, D) tensors.
        start_m, start_n: which tile, in units of `block_m` / `block_n`.
        sm_scale: defaults to 1/sqrt(D).

    Returns:
        (B, H, block_m, block_n) fp32 tile. Entries whose row or column falls
        past N are zero-padded and carry no meaning.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must share a shape, got {q.shape}, {k.shape}")
    b, h, n, d = q.shape
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    # fp32 output: tl.dot accumulates in fp32 and we keep that precision, since
    # the running softmax statistics downstream are maintained in fp32 too.
    s = torch.empty((b, h, block_m, block_n), device=q.device, dtype=torch.float32)

    _qk_tile_kernel[(b * h,)](
        q,
        k,
        s,
        *q.stride(),
        *k.stride(),
        *s.stride(),
        h,
        n,
        sm_scale,
        start_m,
        start_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
    )
    return s


@triton.jit
def _softmax_tile_kernel(
    Q,
    K,
    P,
    M,
    L,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    stride_mb,
    stride_mh,
    stride_mm,
    H,
    N_CTX,
    sm_scale,
    start_m,
    start_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Safe softmax over a single (BLOCK_M, BLOCK_N) tile of S.

    Computes, per row of the tile:
        m = max_j s_j            (row max, for numerical safety)
        p = exp(s - m)           (UNNORMALIZED -- see below)
        l = sum_j p_j            (the partial denominator)

    `p` is deliberately not divided by `l`. FlashAttention carries numerator
    and denominator separately and normalizes once at the very end, because
    `l` is still accumulating over later K blocks. Dividing here would bake in
    a denominator that is about to change.

    This is a correct softmax over the wrong set: the true denominator runs
    over all N keys, not just this block's BLOCK_N. Steps 6-8 fix that.
    """
    off_hz = tl.program_id(0)
    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    p_base = P + off_b * stride_pb + off_h * stride_ph
    m_base = M + off_b * stride_mb + off_h * stride_mh
    l_base = L + off_b * stride_mb + off_h * stride_mh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    s = tl.dot(q, tl.trans(k)) * sm_scale

    # Out-of-range KEYS must be -inf, not 0. A zero score becomes exp(0 - m),
    # a positive weight, which would steal probability mass from real keys.
    # This is invisible unless N_CTX is not a multiple of BLOCK_N.
    s = tl.where(offs_n[None, :] < N_CTX, s, float("-inf"))

    m = tl.max(s, axis=1)  # (BLOCK_M,)
    p = tl.exp(s - m[:, None])  # (BLOCK_M, BLOCK_N), unnormalized
    l = tl.sum(p, axis=1)  # (BLOCK_M,)

    offs_pm = tl.arange(0, BLOCK_M)
    offs_pn = tl.arange(0, BLOCK_N)
    p_ptrs = p_base + offs_pm[:, None] * stride_pm + offs_pn[None, :] * stride_pn
    tl.store(p_ptrs, p)
    tl.store(m_base + offs_pm * stride_mm, m)
    tl.store(l_base + offs_pm * stride_mm, l)


def softmax_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    start_m: int = 0,
    start_n: int = 0,
    block_m: int = 64,
    block_n: int = 64,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Safe softmax over one tile of S = QK^T / sqrt(d).

    Returns:
        p: (B, H, block_m, block_n) unnormalized weights exp(s - m).
        m: (B, H, block_m) row maxima over this tile.
        l: (B, H, block_m) row sums of `p` over this tile.

        The normalized softmax of the tile is `p / l[..., None]`.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must share a shape, got {q.shape}, {k.shape}")
    b, h, n, d = q.shape
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    p = torch.empty((b, h, block_m, block_n), device=q.device, dtype=torch.float32)
    m = torch.empty((b, h, block_m), device=q.device, dtype=torch.float32)
    l = torch.empty((b, h, block_m), device=q.device, dtype=torch.float32)

    _softmax_tile_kernel[(b * h,)](
        q,
        k,
        p,
        m,
        l,
        *q.stride(),
        *k.stride(),
        *p.stride(),
        *m.stride(),
        h,
        n,
        sm_scale,
        start_m,
        start_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
    )
    return p, m, l


@triton.jit
def _running_max_kernel(
    Q,
    K,
    M,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_mb,
    stride_mh,
    stride_mm,
    H,
    N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Track the row max of S online, across every K block.

    Safe softmax needs max_j s_ij over the whole row, but the row is N wide and
    does not fit in SRAM -- that is the premise of the whole algorithm. So the
    max is accumulated one block at a time:

        m^(t) = max(m^(t-1), max_j s^(t)_ij),   m^(0) = -inf

    After the final block m equals the true row max.

    Note the structure: the Q block is loaded ONCE, before the loop, and K
    blocks stream past it. That asymmetry is FlashAttention-2 -- Q stays
    resident in registers while K/V are read from HBM once each.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    m_base = M + off_b * stride_mb + off_h * stride_mh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Loaded once and reused for every K block below.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_n[None, :] < N_CTX, s, float("-inf"))

        m = tl.maximum(m, tl.max(s, axis=1))

    tl.store(m_base + offs_m * stride_mm, m, mask=offs_m < N_CTX)


def running_max(
    q: torch.Tensor,
    k: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Row maxima of S = QK^T / sqrt(d), computed online over K blocks.

    Returns:
        (B, H, N) fp32 tensor equal to S.max(dim=-1) -- but computed without
        ever materializing a full row of S.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must share a shape, got {q.shape}, {k.shape}")
    b, h, n, d = q.shape
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    m = torch.empty((b, h, n), device=q.device, dtype=torch.float32)

    # The real FlashAttention-2 grid: one program per (Q block, batch-head).
    grid = (triton.cdiv(n, block_m), b * h)

    _running_max_kernel[grid](
        q,
        k,
        m,
        *q.stride(),
        *k.stride(),
        *m.stride(),
        h,
        n,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
    )
    return m
