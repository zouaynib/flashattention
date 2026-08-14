"""Differing query and key lengths in the backward pass.

Mirrors the forward case with one asymmetry. dQ's loop bound is the same as the
forward pass -- query row m reads up to key m + offset. dK and dV invert it: key
block n is first visible to query row n - offset, so their Q loop starts there
rather than at zero.

Getting that bound wrong does not crash. It silently drops or double-counts
gradient contributions for a band of rows near the diagonal, which is exactly
the kind of error that trains a slightly wrong model.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.autograd import flash_attention  # noqa: E402
from flash_attn_scratch.naive import naive_attention  # noqa: E402

pytestmark = pytest.mark.gpu

TOL = {torch.float16: (5e-2, 2e-2), torch.bfloat16: (2e-1, 8e-2)}


def _reference_grads(q, k, v, causal):
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    naive_attention(qf, kf, vf, causal=causal).sum().backward()
    return qf.grad, kf.grad, vf.grad


@pytest.mark.parametrize(
    "m,n",
    [
        (256, 256),  # square: unchanged
        (1, 256),  # decode
        (16, 256),  # chunked prefill
        (128, 300),  # neither a block multiple
        (100, 128),
        (255, 256),  # off by one -- one row's bound differs
        (64, 65),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_gradients_match_naive(m, n, causal):
    torch.manual_seed(0)
    b, h, d = 1, 4, 64
    atol, rtol = TOL[torch.float16]

    q = torch.randn(b, h, m, d, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16, requires_grad=True)

    flash_attention(q, k, v, causal=causal).sum().backward()
    want_q, want_k, want_v = _reference_grads(q, k, v, causal)

    assert q.grad.shape == (b, h, m, d)
    assert k.grad.shape == v.grad.shape == (b, h, n, d)

    for got, want, name in zip((q.grad, k.grad, v.grad), (want_q, want_k, want_v), "qkv"):
        torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol, msg=f"d{name}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_gradients_in_both_dtypes(dtype):
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    b, h, m, n, d = 1, 4, 64, 320, 64

    q = torch.randn(b, h, m, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, n, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, n, d, device="cuda", dtype=dtype, requires_grad=True)

    flash_attention(q, k, v, causal=True).sum().backward()
    want_q, want_k, want_v = _reference_grads(q, k, v, True)

    for got, want, name in zip((q.grad, k.grad, v.grad), (want_q, want_k, want_v), "qkv"):
        torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol, msg=f"d{name}")


def test_varlen_with_gqa_gradients():
    """Every shape generalization at once: fewer queries, fewer KV heads."""
    torch.manual_seed(0)
    b, h_q, h_kv, m, n, d = 1, 16, 4, 32, 256, 64
    atol, rtol = TOL[torch.float16]
    g = h_q // h_kv

    q = torch.randn(b, h_q, m, d, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16, requires_grad=True)

    flash_attention(q, k, v, causal=True).sum().backward()

    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    naive_attention(
        qf, kf.repeat_interleave(g, dim=1), vf.repeat_interleave(g, dim=1), causal=True
    ).sum().backward()

    assert k.grad.shape == (b, h_kv, n, d)
    for got, want, name in zip((q.grad, k.grad, v.grad), (qf.grad, kf.grad, vf.grad), "qkv"):
        torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol, msg=f"d{name}")


def test_keys_beyond_every_query_receive_no_gradient():
    """With M queries against N keys and bottom-right causal alignment, every
    key is visible to the last query, so no key is unreachable. But if the
    dK/dV loop bound were computed with the wrong sign, early keys would be
    skipped -- so assert every key gets a nonzero gradient."""
    torch.manual_seed(0)
    b, h, m, n, d = 1, 2, 32, 128, 64

    q = torch.randn(b, h, m, d, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16, requires_grad=True)

    flash_attention(q, k, v, causal=True).sum().backward()

    per_key = v.grad.float().abs().sum(dim=-1)  # (b, h, n)
    assert (per_key > 0).all(), "some keys received no gradient at all"


def test_decode_step_gradient_matches_the_matching_slice():
    """A single decode step's gradients must equal what the full causal pass
    produces for that query row, restricted to the keys it can see."""
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 64, 64
    pos = 40

    q_full = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    k_full = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v_full = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    # One query against the cache up to `pos`.
    q1 = q_full[:, :, pos : pos + 1].clone().requires_grad_(True)
    k1 = k_full[:, :, : pos + 1].clone().requires_grad_(True)
    v1 = v_full[:, :, : pos + 1].clone().requires_grad_(True)
    flash_attention(q1, k1, v1, causal=True).sum().backward()

    # The same row, obtained by differentiating only that row of a full pass.
    qf = q_full.detach().float().requires_grad_(True)
    kf = k_full.detach().float().requires_grad_(True)
    vf = v_full.detach().float().requires_grad_(True)
    naive_attention(qf, kf, vf, causal=True)[:, :, pos : pos + 1].sum().backward()

    atol, rtol = TOL[torch.float16]
    torch.testing.assert_close(q1.grad.float(), qf.grad[:, :, pos : pos + 1], atol=atol, rtol=rtol)
    torch.testing.assert_close(v1.grad.float(), vf.grad[:, :, : pos + 1], atol=atol, rtol=rtol)


def test_rejects_causal_with_more_queries_than_keys():
    q = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    with pytest.raises(ValueError, match="at least as many keys as queries"):
        flash_attention(q, k, k, causal=True)


def test_varlen_gradients_are_finite():
    torch.manual_seed(0)
    for m, n in [(1, 256), (33, 300), (128, 129)]:
        q = torch.randn(1, 2, m, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        flash_attention(q, k, v, causal=True).sum().backward()
        for t in (q, k, v):
            assert torch.isfinite(t.grad).all(), f"non-finite gradient at M={m}, N={n}"
