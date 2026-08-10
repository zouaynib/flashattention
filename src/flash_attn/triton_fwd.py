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
* `_running_sum_kernel` -- adds the running denominator l and the rescaling
  factor alpha = exp(m_old - m_new). This is the core of online softmax.
* `_running_output_kernel` -- adds the output accumulator. Online softmax is
  complete here; steps 9-10 are assembly and masking.
* `_fwd_kernel` / `flash_attention_forward` -- the assembled forward pass,
  normalized in-kernel and returning the input dtype, with causal masking.
  This is the public API.

The earlier kernels are kept deliberately. In a production repo they would be
dead code; here each one pins down a single idea and has a test suite proving
it, which is the point of the project.

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

# tl.dot reaches tensor cores only for these. fp32 would fall back to a much
# slower path and fp64 is unsupported outright, so the contract is explicit
# rather than failing somewhere deep inside Triton.
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


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


@triton.jit
def _running_sum_kernel(
    Q,
    K,
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
    """Online softmax denominator: running max AND running sum, with rescaling.

    When a later block raises the running max, every term already accumulated
    into l was exponentiated against a stale, too-small maximum. They are all
    wrong by the SAME factor, so a single multiply repairs the whole sum:

        alpha^(t) = exp(m^(t-1) - m^(t))
        l^(t)     = alpha^(t) * l^(t-1) + sum_j exp(s^(t)_j - m^(t))

    This is the heart of FlashAttention: a summary of arbitrarily many past
    keys can be retroactively corrected in O(1) per row. alpha is always in
    (0, 1], so rescaling only ever shrinks past contributions -- it cannot
    overflow.

    On the first iteration m^(0) = -inf gives alpha = 0, which correctly zeroes
    the empty accumulator. That relies on every row seeing at least one valid
    key; causal masking (step 10) breaks that assumption and needs care.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    m_base = M + off_b * stride_mb + off_h * stride_mh
    l_base = L + off_b * stride_mb + off_h * stride_mh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_n[None, :] < N_CTX, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))

        # The correction factor. Everything accumulated so far was measured
        # against m_i; this converts it to the m_new scale.
        alpha = tl.exp(m_i - m_new)

        # Masked entries are -inf, so exp(-inf - m_new) == 0: no weight.
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    tl.store(m_base + offs_m * stride_mm, m_i, mask=offs_m < N_CTX)
    tl.store(l_base + offs_m * stride_mm, l_i, mask=offs_m < N_CTX)


def running_softmax_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax statistics for every row of S, computed online over K blocks.

    Returns:
        m: (B, H, N) row maxima.
        l: (B, H, N) softmax denominators, sum_j exp(s_ij - m_i).

        The true softmax of row i is exp(s_ij - m_i) / l_i -- computed here
        without ever materializing a full row of S.
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
    l = torch.empty((b, h, n), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(n, block_m), b * h)

    _running_sum_kernel[grid](
        q,
        k,
        m,
        l,
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
    return m, l


@triton.jit
def _running_output_kernel(
    Q,
    K,
    V,
    O,
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
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
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
    """Online softmax, complete: running max, running sum, running output.

    The output accumulator takes the same correction as the denominator, just
    broadcast across the head dimension:

        O^(t) = alpha^(t) * O^(t-1) + P^(t) V^(t)

    and the attention output is O^(T) / l^(T) once the loop ends.

    Normalization is deferred to the very end rather than applied per block.
    That is the FlashAttention-2 change over FA-1: it removes a divide from the
    inner loop, and non-matmul work is roughly an order of magnitude less
    efficient than tensor-core matmul on this hardware.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = O + off_b * stride_ob + off_h * stride_oh
    m_base = M + off_b * stride_mb + off_h * stride_mh
    l_base = L + off_b * stride_mb + off_h * stride_mh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_n[None, :] < N_CTX, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Same correction, broadcast over the head dimension. P is cast to the
        # input dtype so this hits tensor cores; the accumulator stays fp32.
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)
    tl.store(m_base + offs_m * stride_mm, m_i, mask=offs_m < N_CTX)
    tl.store(l_base + offs_m * stride_mm, l_i, mask=offs_m < N_CTX)


def running_output_accumulator(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unnormalized attention output plus its softmax statistics.

    Returns:
        o: (B, H, N, D) fp32, the UNNORMALIZED accumulator sum_j exp(s_ij - m_i) v_j.
        m: (B, H, N) row maxima.
        l: (B, H, N) softmax denominators.

        Attention output is o / l[..., None]. Step 9 wraps that up.
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v must share a shape, got {q.shape}, {k.shape}, {v.shape}")
    b, h, n, d = q.shape
    for name, value in (("block_m", block_m), ("block_n", block_n), ("head_dim", d)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    o = torch.empty((b, h, n, d), device=q.device, dtype=torch.float32)
    m = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
    l = torch.empty((b, h, n), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(n, block_m), b * h)

    _running_output_kernel[grid](
        q,
        k,
        v,
        o,
        m,
        l,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *m.stride(),
        h,
        n,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
    )
    return o, m, l


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    O,
    L,
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
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    H,
    N_CTX_Q,
    N_CTX_KV,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """The complete FlashAttention-2 forward pass.

    Q-block outer loop (the grid), K/V-block inner loop (below). One Q tile is
    held in registers while K and V stream past it once, so HBM traffic is
    O(N^2 d / M) instead of the O(N^2) of a materialized score matrix.

    Normalization happens here, before the store: dividing in PyTorch afterwards
    would mean another full read-modify-write of O through HBM.

    l >= 1 always -- the row-max term contributes exp(0) = 1 -- so the division
    needs no zero guard.

    Q and K/V may have different lengths, which is what KV-cache decoding
    needs: N_CTX_Q new queries against N_CTX_KV cached keys.

    Causal masking stops the inner loop at the diagonal rather than masking
    every block. K/V blocks strictly above the diagonal are never loaded, so
    causal attention does about half the work of non-causal.

    That also keeps the arithmetic safe. Had the loop run to N_CTX and masked,
    an entirely-masked block would leave m = -inf, making alpha = exp(-inf +
    inf) = NaN. Stopping at the diagonal guarantees every row sees at least its
    own diagonal element, so m is always finite.

    Supports grouped-query attention: GROUP_SIZE query heads share one K/V
    head, so query head h reads KV head h // GROUP_SIZE. K and V are addressed
    through that mapping rather than being replicated to H_q heads -- expanding
    them would discard exactly the memory saving GQA exists for. GROUP_SIZE = 1
    is ordinary multi-head attention.

    Also writes the log-sum-exp L = m + log(l), one fp32 value per query row.
    That single number is all the backward pass needs to rebuild any attention
    probability, since p_ij = exp(s_ij - L_i). Storing P itself would cost
    O(N^2) per head; L costs O(N).
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H  # query head
    off_h_kv = off_h // GROUP_SIZE  # the K/V head it shares

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h_kv * stride_kh
    v_base = V + off_b * stride_vb + off_h_kv * stride_vh
    o_base = O + off_b * stride_ob + off_h * stride_oh
    l_base = L + off_b * stride_lb + off_h * stride_lh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal rows never look past their own index, so the loop stops at the
    # last block containing the diagonal. Everything above it is skipped.
    # Bottom-right alignment: the N_CTX_Q queries are the LAST N_CTX_Q
    # positions of an N_CTX_KV-long sequence, so query row m may attend up to
    # key m + offset. With one query against a full cache, that is every key.
    causal_offset = N_CTX_KV - N_CTX_Q

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M + causal_offset, N_CTX_KV)
    else:
        hi = N_CTX_KV

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX_KV, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX_KV, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(offs_n[None, :] < N_CTX_KV, s, float("-inf"))

        # Only blocks straddling the diagonal actually need this; blocks fully
        # below it are entirely valid. Applying it uniformly is correct but
        # leaves non-matmul work in the hot loop -- splitting the loop into an
        # unmasked range plus a masked diagonal range is a Phase 5 tuning item.
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] + causal_offset >= offs_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q)

    # p_ij = exp(s_ij - L_i) recovers the normalized probabilities directly,
    # with no division -- which is why m and l are folded into one value.
    tl.store(l_base + offs_m * stride_lm, m_i + tl.log(l_i), mask=offs_m < N_CTX_Q)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    return_lse: bool = False,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 forward pass.

    Args:
        q: (B, H_q, M, D) -- M queries.
        k, v: (B, H_kv, N, D) -- N keys. H_q must be a multiple of H_kv (pass
            H_kv < H_q for grouped-query attention, H_kv = 1 for multi-query).
            M and N may differ: M < N is KV-cache decoding, where the M queries
            are the last M positions of an N-long sequence. Causal masking is
            aligned bottom-right accordingly, and requires M <= N.
        causal: if True, position i attends only to positions j <= i.
        return_lse: also return the per-row log-sum-exp, which the backward
            pass uses to recompute attention probabilities blockwise.
        sm_scale: defaults to 1/sqrt(D).
        block_m, block_n: tile sizes. Default to a heuristic based on D.

    Returns:
        (B, H_q, M, D) attention output in the input dtype, or a tuple of that
        plus the (B, H_q, M) fp32 log-sum-exp when `return_lse` is set.
    """
    if k.shape != v.shape:
        raise ValueError(f"k and v must share a shape, got {k.shape}, {v.shape}")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("expected 4D (B, H, N, D) tensors")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError(
            f"q and k/v may differ only in head count and length, got "
            f"{q.shape} and {k.shape}"
        )
    if causal and q.shape[2] > k.shape[2]:
        # offset = kv_len - q_len would be negative, so the earliest queries
        # could attend to nothing: m stays -inf and alpha becomes NaN.
        raise ValueError(
            f"causal attention needs at least as many keys as queries, got "
            f"{q.shape[2]} queries and {k.shape[2]} keys"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"query heads must be a multiple of kv heads (grouped-query attention), "
            f"got {q.shape[1]} and {k.shape[1]}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"only fp16 and bf16 are supported (the tensor-core dtypes), got {q.dtype}"
        )
    b, h, n, d = q.shape
    n_kv = k.shape[2]
    # How many query heads share each K/V head. 1 is ordinary multi-head.
    group_size = h // k.shape[1]
    if d & (d - 1) != 0:
        raise ValueError(f"head_dim must be a power of two, got {d}")
    if d < 16:
        raise ValueError(f"head_dim must be >= 16 for the tensor-core path, got {d}")

    # Heuristic rather than autotuning: @triton.autotune recompiles across the
    # config space per shape, which would contaminate the Phase 5 latency
    # measurements. Tuning belongs with the benchmark harness, where the
    # tradeoff is measurable.
    if block_m is None:
        block_m = 64 if d >= 128 else 128
    if block_n is None:
        block_n = 64
    for name, value in (("block_m", block_m), ("block_n", block_n)):
        if value < 16:
            raise ValueError(f"{name} must be >= 16 for the tensor-core path, got {value}")

    # A 64x128 fp32 accumulator is already ~64 registers per thread at 4 warps;
    # widening to 8 warps halves the per-thread pressure at large head dims.
    num_warps = 8 if d >= 128 else 4

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    o = torch.empty_like(q)
    # Always written: a few hundred KB against a multi-MB output is noise, and
    # it keeps the autograd path from needing a second kernel variant.
    lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(n, block_m), b * h)

    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *lse.stride(),
        h,
        n,
        n_kv,
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        GROUP_SIZE=group_size,
        num_warps=num_warps,
    )
    return (o, lse) if return_lse else o
 
