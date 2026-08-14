"""FlashAttention-2 as a differentiable PyTorch operation.

Wires the forward and backward Triton kernels into a `torch.autograd.Function`
so the kernel behaves like any built-in op: put it in a model, call
`loss.backward()`, and gradients flow.

What gets saved for backward is the point of the whole project: Q, K, V, O and
the log-sum-exp L. Every one of those is O(N*D) or smaller. Nothing quadratic
in the sequence length is ever stored, so activation memory for attention grows
linearly with context length instead of quadratically.

A note on validation: `torch.autograd.gradcheck` is the usual way to verify a
custom Function, but it requires float64 and `tl.dot` only accepts fp16/bf16 on
tensor cores. gradcheck is therefore structurally unavailable. The tests instead
compare against autograd through the fp32 naive baseline, which is the strongest
check this dtype permits.
"""

import math

import torch

from flash_attn_scratch.triton_bwd import backward_dk, backward_dq, backward_dv, backward_preprocess
from flash_attn_scratch.triton_fwd import flash_attention_forward


class FlashAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the Triton forward and backward kernels."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # Resolve the default here, once. If backward re-derived it separately,
        # a caller passing an explicit scale to forward would get gradients
        # computed against a different scale -- silently.
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])

        o, lse = flash_attention_forward(
            q, k, v, causal=causal, sm_scale=sm_scale, return_lse=True
        )

        # Five tensors, all O(N*D) or smaller. No N x N anything.
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        causal, sm_scale = ctx.causal, ctx.sm_scale

        delta = backward_preprocess(o, do)
        need_q, need_k, need_v = ctx.needs_input_grad[:3]

        # Skip whole kernels when a gradient is not wanted -- frozen K/V
        # projections are common enough for this to be worth the branch.
        dq = (
            backward_dq(q, k, v, do, lse, delta, causal=causal, sm_scale=sm_scale)
            if need_q
            else None
        )
        dk = (
            backward_dk(q, k, v, do, lse, delta, causal=causal, sm_scale=sm_scale)
            if need_k
            else None
        )
        dv = backward_dv(q, k, do, lse, causal=causal, sm_scale=sm_scale) if need_v else None

        # One gradient per forward input; `causal` and `sm_scale` are not tensors.
        return dq, dk, dv, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Differentiable FlashAttention-2, a drop-in for scaled_dot_product_attention.

    Args:
        q: (B, H_q, M, D).
        k, v: (B, H_kv, N, D). M and N may differ. H_q must be a multiple of H_kv; pass H_kv < H_q
            for grouped-query attention or H_kv = 1 for multi-query. dK and dV
            come back with H_kv heads, matching their inputs.
        causal: if True, position i attends only to positions j <= i.
        sm_scale: softmax temperature, defaulting to 1/sqrt(D).

    Returns:
        (B, H, N, D) attention output in the input dtype, with gradients wired
        to q, k and v.
    """
    return FlashAttentionFunction.apply(q, k, v, causal, sm_scale)
