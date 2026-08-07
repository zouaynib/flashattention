"""Naive scaled dot-product attention in plain PyTorch.

This is the reference implementation: correctness ground truth for the Triton
kernels, and the memory-hungry baseline that FlashAttention is measured against.

It deliberately materializes the full N x N score matrix. That is the whole
point -- the quadratic HBM traffic here is what the tiled kernel eliminates.

The softmax is written out by hand rather than calling `torch.softmax` so the
`m` (row max) and `l` (row sum) statistics are visible. The online softmax in
the Triton kernel computes exactly these two quantities, one K/V block at a
time, instead of over the whole row at once.
"""

import math

import torch


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
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must share a shape, got {q.shape}, {k.shape}, {v.shape}")
    if q.dim() != 4:
        raise ValueError(f"expected 4D (B, H, N, D) tensors, got {q.dim()}D")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    # S = QK^T / sqrt(d) -- the (B, H, N, N) matrix we are trying to avoid.
    s = (q @ k.transpose(-2, -1)) * sm_scale

    if causal:
        # Mask strictly-upper-triangular entries (j > i) before the softmax, so
        # they contribute exactly zero weight. Row i always keeps at least the
        # diagonal entry, so no row is fully masked and m stays finite.
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(diagonal=1)
        s = s.masked_fill(mask, float("-inf"))

    # Safe softmax, written out. Subtracting the row max before exponentiating
    # bounds exp() at 1.0; without it, large scores overflow in fp16.
    m = s.max(dim=-1, keepdim=True).values  # running max `m_i` in the tiled version
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)  # running sum `l_i` in the tiled version

    return (p / l) @ v
