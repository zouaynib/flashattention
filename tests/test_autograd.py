"""Step 14: the autograd.Function, tested as a real PyTorch operation.

These tests use `loss.backward()` rather than calling the backward kernels
directly, so they exercise what a model would actually hit: tensor saving,
gradient accumulation, needs_input_grad, and composition with other ops.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.autograd import flash_attention  # noqa: E402
from flash_attn.naive import naive_attention  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 5e-2
RTOL = 2e-2


def _make(b, h, n, d, requires_grad=True):
    return [
        torch.randn(b, h, n, d, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
        for _ in range(3)
    ]


@pytest.mark.parametrize(
    "b,h,n,d",
    [(1, 1, 64, 64), (2, 4, 256, 64), (1, 1, 100, 32), (2, 2, 17, 16)],
)
@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_naive_autograd(b, h, n, d, causal):
    """End to end: loss.backward() through the kernel vs through the baseline."""
    torch.manual_seed(0)
    q, k, v = _make(b, h, n, d)
    ref = [t.detach().float().requires_grad_(True) for t in (q, k, v)]

    flash_attention(q, k, v, causal=causal).sum().backward()
    naive_attention(*ref, causal=causal).sum().backward()

    for got, want, name in zip((q, k, v), ref, "qkv"):
        torch.testing.assert_close(
            got.grad.float(), want.grad, atol=ATOL, rtol=RTOL, msg=f"d{name} mismatch"
        )


def test_forward_value_is_unchanged_by_the_wrapper():
    torch.manual_seed(0)
    q, k, v = _make(1, 2, 256, 64, requires_grad=False)

    from flash_attn.triton_fwd import flash_attention_forward

    assert torch.equal(flash_attention(q, k, v, causal=True), flash_attention_forward(q, k, v, causal=True))


def test_composes_with_other_operations():
    """Attention in the middle of a graph, with a non-trivial loss.

    A `.sum()` loss sends dO = ones, which can hide orientation bugs. Weighting
    by a random tensor makes dO non-uniform.
    """
    torch.manual_seed(0)
    q, k, v = _make(1, 2, 128, 64)
    ref = [t.detach().float().requires_grad_(True) for t in (q, k, v)]
    w = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16)

    (flash_attention(q * 2.0, k, v, causal=True) * w).sum().backward()
    (naive_attention(ref[0] * 2.0, ref[1], ref[2], causal=True) * w.float()).sum().backward()

    for got, want, name in zip((q, k, v), ref, "qkv"):
        torch.testing.assert_close(
            got.grad.float(), want.grad, atol=ATOL, rtol=RTOL, msg=f"d{name} mismatch"
        )


@pytest.mark.parametrize("which", [0, 1, 2])
def test_partial_requires_grad(which):
    """Only the requested gradient is produced; the others stay None.

    This is the needs_input_grad path, which skips whole kernels.
    """
    torch.manual_seed(0)
    tensors = [
        torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16) for _ in range(3)
    ]
    tensors[which].requires_grad_(True)

    flash_attention(*tensors, causal=True).sum().backward()

    for i, t in enumerate(tensors):
        if i == which:
            assert t.grad is not None and torch.isfinite(t.grad).all()
        else:
            assert t.grad is None


def test_gradients_accumulate_across_backward_calls():
    """Standard autograd semantics: two backwards add into .grad."""
    torch.manual_seed(0)
    q, k, v = _make(1, 1, 64, 64)

    flash_attention(q, k, v).sum().backward()
    first = q.grad.clone()
    flash_attention(q, k, v).sum().backward()

    torch.testing.assert_close(q.grad.float(), (2 * first).float(), atol=ATOL, rtol=RTOL)


def test_no_grad_mode_produces_no_graph():
    q, k, v = _make(1, 1, 64, 64)
    with torch.no_grad():
        out = flash_attention(q, k, v)
    assert not out.requires_grad


def test_explicit_scale_is_used_by_backward():
    """Forward resolves sm_scale once and stashes it. If backward re-derived
    the default instead, an explicit scale would give wrong gradients."""
    torch.manual_seed(0)
    scale = 0.1
    q, k, v = _make(1, 2, 128, 64)
    ref = [t.detach().float().requires_grad_(True) for t in (q, k, v)]

    flash_attention(q, k, v, sm_scale=scale).sum().backward()
    naive_attention(*ref, sm_scale=scale).sum().backward()

    for got, want, name in zip((q, k, v), ref, "qkv"):
        torch.testing.assert_close(
            got.grad.float(), want.grad, atol=ATOL, rtol=RTOL, msg=f"d{name} mismatch"
        )


def test_handles_noncontiguous_inputs():
    """Transformers produce (B, N, H, D) and transpose before attention."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    out = flash_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
    out.sum().backward()

    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.shape == t.shape


def test_output_dtype_and_device_match_input():
    q, k, v = _make(1, 1, 64, 64)
    out = flash_attention(q, k, v)
    assert out.dtype == q.dtype and out.device == q.device
