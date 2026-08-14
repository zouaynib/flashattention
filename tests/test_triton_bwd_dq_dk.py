"""Step 13: gradients with respect to Q and K, plus the D = rowsum(dO * O) term.

All three gradients are checked in one pass per case: the fp32 autograd
reference is the expensive part, so computing it once and comparing dQ, dK and
dV together is both faster and a stronger joint check.

Tolerances are looser than the forward tests for a substantive reason:
dS = P * (dP - D) subtracts two similar quantities, and that cancellation
amplifies relative error wherever dP_ij is close to D_i. It is inherent to
attention backward, which is why the intermediate is held in fp32.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.naive import naive_attention  # noqa: E402
from flash_attn_scratch.triton_bwd import (  # noqa: E402
    backward_dk,
    backward_dq,
    backward_dv,
    backward_preprocess,
)
from flash_attn_scratch.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 5e-2
RTOL = 2e-2


def _reference_grads(q, k, v, do, causal):
    qf, kf, vf = (t.float().detach().requires_grad_(True) for t in (q, k, v))
    naive_attention(qf, kf, vf, causal=causal).backward(do.float())
    return qf.grad, kf.grad, vf.grad


def _triton_grads(q, k, v, do, causal, block_m=64, block_n=64):
    o, lse = flash_attention_forward(q, k, v, causal=causal, return_lse=True)
    delta = backward_preprocess(o, do)
    kw = dict(causal=causal, block_m=block_m, block_n=block_n)
    return (
        backward_dq(q, k, v, do, lse, delta, **kw),
        backward_dk(q, k, v, do, lse, delta, **kw),
        backward_dv(q, k, do, lse, causal=causal, block_m=block_m, block_n=block_n),
    )


def test_delta_matches_rowsum_of_do_times_o():
    """D_i = dO_i . O_i, the identity that removes P from the softmax Jacobian."""
    torch.manual_seed(0)
    for b, h, n, d in [(1, 2, 256, 64), (1, 1, 100, 32)]:
        q, k, v, do = (
            torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(4)
        )
        o = flash_attention_forward(q, k, v)

        got = backward_preprocess(o, do)
        want = (o.float() * do.float()).sum(dim=-1)

        assert got.shape == (b, h, n)
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),
        (2, 4, 256, 64),
        (1, 1, 100, 32),  # partial blocks in both loop directions
        (2, 2, 17, 16),
        (1, 2, 384, 128),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_all_gradients_match_autograd(b, h, n, d, causal):
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    got = _triton_grads(q, k, v, do, causal)
    want = _reference_grads(q, k, v, do, causal)

    for g, w, name in zip(got, want, ("dQ", "dK", "dV")):
        assert g.shape == (b, h, n, d)
        torch.testing.assert_close(g.float(), w, atol=ATOL, rtol=RTOL, msg=f"{name} mismatch")


@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_gradients_independent_of_block_sizes(block_m, block_n, causal):
    """dQ and dK traverse the tiles in opposite directions, and under causal
    each derives its own loop bound. Both must land on the same answer."""
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    got = _triton_grads(q, k, v, do, causal, block_m=block_m, block_n=block_n)
    want = _reference_grads(q, k, v, do, causal)

    for g, w, name in zip(got, want, ("dQ", "dK", "dV")):
        torch.testing.assert_close(g.float(), w, atol=ATOL, rtol=RTOL, msg=f"{name} mismatch")


def test_causal_first_query_has_zero_gradient():
    """Under causal masking, query 0 attends to exactly one key with p = 1.

    A softmax over a single element is constant, so its Jacobian vanishes and
    dQ[0] must be exactly zero -- no matter what dO is. This isolates the
    (dP - D) cancellation: at row 0, dP_00 and D_0 are equal by construction,
    so any error in the D term shows up immediately here.
    """
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    dq, _, _ = _triton_grads(q, k, v, do, causal=True)

    torch.testing.assert_close(
        dq[:, :, 0, :].float(), torch.zeros_like(dq[:, :, 0, :]).float(), atol=1e-3, rtol=1e-3
    )
    # And the rest is genuinely nonzero, so the check above is not vacuous.
    assert dq[:, :, 1:, :].abs().max() > 1e-3


def test_causal_gradients_differ_from_non_causal():
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    causal = _triton_grads(q, k, v, do, causal=True)
    full = _triton_grads(q, k, v, do, causal=False)

    for c, f, name in zip(causal, full, ("dQ", "dK", "dV")):
        assert not torch.allclose(c, f, atol=1e-2), f"{name} unchanged by causal masking"


def test_custom_scale_propagates_to_gradients():
    """sm_scale multiplies dQ and dK directly, so a dropped factor shows here."""
    torch.manual_seed(0)
    scale = 0.1
    q, k, v, do = (
        torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    o, lse = flash_attention_forward(q, k, v, sm_scale=scale, return_lse=True)
    delta = backward_preprocess(o, do)
    dq = backward_dq(q, k, v, do, lse, delta, sm_scale=scale)
    dk = backward_dk(q, k, v, do, lse, delta, sm_scale=scale)

    qf, kf, vf = (t.float().detach().requires_grad_(True) for t in (q, k, v))
    naive_attention(qf, kf, vf, sm_scale=scale).backward(do.float())

    torch.testing.assert_close(dq.float(), qf.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(dk.float(), kf.grad, atol=ATOL, rtol=RTOL)


def test_gradients_are_finite():
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )
    for causal in (False, True):
        for g in _triton_grads(q, k, v, do, causal):
            assert torch.isfinite(g).all()


def test_rejects_wrong_delta_shape():
    q, k, v, do = (torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16) for _ in range(4))
    lse = torch.zeros(1, 2, 64, device="cuda")
    with pytest.raises(ValueError, match="lse and delta must be"):
        backward_dq(q, k, v, do, lse, torch.zeros(1, 2, 32, device="cuda"))


def test_rejects_mismatched_stats_strides():
    """The kernels index lse and delta through one set of strides, so a delta
    with a different layout would be read wrongly rather than loudly."""
    q, k, v, do = (torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16) for _ in range(4))
    lse = torch.zeros(1, 2, 64, device="cuda")
    delta = torch.zeros(2, 1, 64, device="cuda").transpose(0, 1)

    assert delta.shape == lse.shape and delta.stride() != lse.stride()
    with pytest.raises(ValueError, match="share strides"):
        backward_dq(q, k, v, do, lse, delta)
