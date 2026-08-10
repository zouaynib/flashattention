"""Step 19: grouped-query attention in the backward pass.

The subtlety is dK and dV. One K/V head is read by GROUP_SIZE query heads, so
its gradient is the SUM of the gradients each of those heads produces. Get that
wrong and the most likely symptom is a gradient that is too small by a factor of
GROUP_SIZE -- which still trains, just badly, and would never be caught by a
shape check.

The reference expands K and V to H_q heads and lets autograd do the summing,
which is exactly what `repeat_interleave` backward does.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.autograd import flash_attention  # noqa: E402
from flash_attn.naive import naive_attention  # noqa: E402

pytestmark = pytest.mark.gpu

TOL = {torch.float16: (5e-2, 2e-2), torch.bfloat16: (2e-1, 8e-2)}


def _reference_grads(q, k, v, causal):
    """Autograd through the expanded reference.

    repeat_interleave's backward sums the replicated heads' gradients back into
    the original head, which is precisely the accumulation the kernel must do.
    """
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    g = q.shape[1] // k.shape[1]

    naive_attention(
        qf, kf.repeat_interleave(g, dim=1), vf.repeat_interleave(g, dim=1), causal=causal
    ).sum().backward()
    return qf.grad, kf.grad, vf.grad


@pytest.mark.parametrize(
    "h_q,h_kv",
    [
        (8, 8),  # MHA -- must stay on the original, unreduced code path
        (8, 4),  # group size 2
        (8, 2),  # group size 4
        (32, 8),  # Llama-3-8B
        (8, 1),  # multi-query: every head folds into one
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_gradients_match_expanded_reference(h_q, h_kv, causal):
    torch.manual_seed(0)
    b, n, d = 1, 256, 64
    atol, rtol = TOL[torch.float16]

    q = torch.randn(b, h_q, n, d, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16, requires_grad=True)

    flash_attention(q, k, v, causal=causal).sum().backward()
    want_q, want_k, want_v = _reference_grads(q, k, v, causal)

    assert k.grad.shape == (b, h_kv, n, d), "dK must not come back expanded"
    assert v.grad.shape == (b, h_kv, n, d), "dV must not come back expanded"

    for got, want, name in zip((q.grad, k.grad, v.grad), (want_q, want_k, want_v), "qkv"):
        torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol, msg=f"d{name}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gqa_gradients_in_both_dtypes(dtype):
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    b, h_q, h_kv, n, d = 1, 16, 4, 384, 64

    q = torch.randn(b, h_q, n, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=dtype, requires_grad=True)

    flash_attention(q, k, v, causal=True).sum().backward()
    want_q, want_k, want_v = _reference_grads(q, k, v, True)

    for got, want, name in zip((q.grad, k.grad, v.grad), (want_q, want_k, want_v), "qkv"):
        torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol, msg=f"d{name}")


def test_group_gradients_are_summed_not_averaged_or_dropped():
    """The scale of the accumulation, isolated.

    Give every query head in a group identical Q rows. Each contributes the same
    gradient to the shared K/V head, so dV under H_kv=1 must be exactly
    GROUP_SIZE times the dV from a single-head run with the same data. Summing
    wrongly -- taking one head, or averaging -- changes this factor.
    """
    torch.manual_seed(0)
    b, h_q, n, d = 1, 4, 128, 64

    one = torch.randn(b, 1, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, 1, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, 1, n, d, device="cuda", dtype=torch.float16)

    # Four identical query heads against one K/V head.
    q_many = one.expand(b, h_q, n, d).contiguous().requires_grad_(True)
    k_many = k.clone().requires_grad_(True)
    v_many = v.clone().requires_grad_(True)
    flash_attention(q_many, k_many, v_many, causal=True).sum().backward()

    # One query head against the same K/V head.
    q_one = one.clone().requires_grad_(True)
    k_one = k.clone().requires_grad_(True)
    v_one = v.clone().requires_grad_(True)
    flash_attention(q_one, k_one, v_one, causal=True).sum().backward()

    torch.testing.assert_close(
        v_many.grad.float(), h_q * v_one.grad.float(), atol=1e-1, rtol=5e-2
    )
    torch.testing.assert_close(
        k_many.grad.float(), h_q * k_one.grad.float(), atol=1e-1, rtol=5e-2
    )


def test_mha_backward_is_bit_identical_to_before():
    """Group size 1 must not touch the scratch-buffer path at all."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    ref = [t.detach().float().requires_grad_(True) for t in (q, k, v)]

    flash_attention(q, k, v, causal=True).sum().backward()
    naive_attention(*ref, causal=True).sum().backward()

    atol, rtol = TOL[torch.float16]
    for got, want, name in zip((q, k, v), ref, "qkv"):
        assert got.grad.dtype == torch.float16, "MHA path should not upcast gradients"
        torch.testing.assert_close(got.grad.float(), want.grad, atol=atol, rtol=rtol, msg=f"d{name}")


def test_gqa_gradients_are_finite():
    torch.manual_seed(0)
    for h_q, h_kv in [(8, 2), (32, 8), (8, 1)]:
        q = torch.randn(1, h_q, 300, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, h_kv, 300, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, h_kv, 300, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        flash_attention(q, k, v, causal=True).sum().backward()
        for t in (q, k, v):
            assert torch.isfinite(t.grad).all()
