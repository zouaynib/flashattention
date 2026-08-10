"""Scaled dot-product attention in plain PyTorch.

Two implementations, with different jobs:

* `naive_attention` -- the CORRECTNESS reference. Writes the softmax out by
  hand so the `m` (row max) and `l` (row sum) statistics are visible; the
  online softmax in the Triton kernel computes exactly these two quantities,
  one K/V block at a time instead of over the whole row at once. Being
  explicit costs extra intermediate tensors, which does not matter when the
  only goal is a trustworthy answer.

* `standard_attention` -- the BENCHMARK baseline. The attention a competent
  engineer actually writes (fused `torch.softmax`, no hand-rolled statistics),
  and close to what nanoGPT and similar codebases use.

Both materialize the full N x N score matrix -- that is the point, and what the
tiled kernel eliminates. But they do not materialize the SAME NUMBER of them,
and benchmarking against the explicit version would be dishonest: it allocates
roughly four N x N-scale tensors (S, its masked copy, P, and the normalized
copy of P) against `standard_attention`'s two. At B=4, H=8, N=8192 in fp16 each
is ~4.3 GB, so the difference is over 10 GB -- enough to move the out-of-memory
point by about 2x in sequence length on a 24 GB card. That would make the
headline benchmark result an artifact of our own baseline rather than a fact
about attention.
"""

import math

import torch


def _check_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> None:
    if k.shape != v.shape:
        raise ValueError(f"k/v must share a shape, got {k.shape}, {v.shape}")
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("expected 4D (B, H, N, D) tensors")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q/k/v must share a shape apart from length, got {q.shape}, {k.shape}")
    if causal and q.shape[2] > k.shape[2]:
        raise ValueError(
            f"causal attention needs at least as many keys as queries, got "
            f"{q.shape[2]} queries and {k.shape[2]} keys: the earliest rows would "
            f"attend to nothing"
        )


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """True where a key must be masked out.

    Aligned bottom-right: the q_len queries are the LAST q_len positions of a
    kv_len-long sequence, so query row m may attend to keys j <= m + (kv_len -
    q_len). That is the convention KV-cache decoding needs -- with q_len = 1 the
    single query sees every cached key. Top-left alignment (j <= m) would let a
    freshly generated token attend to almost nothing.

    When q_len == kv_len the offset is zero and this is the usual triangle.
    """
    offset = kv_len - q_len
    rows = torch.arange(q_len, device=device)[:, None]
    cols = torch.arange(kv_len, device=device)[None, :]
    return cols > rows + offset


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Attention over (batch, heads, seq_len, head_dim) tensors.

    Args:
        q, k, v: shape (B, H, N, D). Must share dtype and device.
        causal: if True, position i may only attend to positions j <= i.
        sm_scale: softmax temperature. Defaults to 1/sqrt(D), the standard
            scaling that keeps the pre-softmax variance from growing with D.

    Returns:
        Output tensor of shape (B, H, N, D).

    Peak memory is dominated by the (B, H, N, N) score matrix, so this OOMs at
    sequence lengths the tiled kernel handles comfortably.
    """
    _check_shapes(q, k, v, causal)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    # S = QK^T / sqrt(d) -- the (B, H, N, N) matrix we are trying to avoid.
    s = (q @ k.transpose(-2, -1)) * sm_scale

    if causal:
        s = s.masked_fill(_causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))

    # Safe softmax, written out. Subtracting the row max before exponentiating
    # bounds exp() at 1.0; without it, large scores overflow in fp16.
    m = s.max(dim=-1, keepdim=True).values  # running max `m_i` in the tiled version
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)  # running sum `l_i` in the tiled version

    return (p / l) @ v


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Attention as ordinarily written in PyTorch, for benchmarking.

    Same math as `naive_attention`, but leaning on the fused `torch.softmax`
    instead of hand-rolling the max/exp/sum. That removes two N x N-scale
    intermediates, which is why this -- not the explicit version -- is the fair
    baseline to measure the Triton kernel against.

    It still materializes S and P, so its memory grows quadratically with the
    sequence length. That is the property under test.

    Args:
        q, k, v: shape (B, H, N, D). Must share dtype and device.
        causal: if True, position i may only attend to positions j <= i.
        sm_scale: defaults to 1/sqrt(D).

    Returns:
        Output tensor of shape (B, H, N, D).
    """
    _check_shapes(q, k, v, causal)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    s = (q @ k.transpose(-2, -1)) * sm_scale

    if causal:
        s = s.masked_fill(_causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))

    return torch.softmax(s, dim=-1) @ v
