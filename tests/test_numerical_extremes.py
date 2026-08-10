"""Step 16: tests that fail if the max subtraction is removed.

Every other test in this repo uses N(0,1) inputs, where scores land in roughly
[-5, 5]. exp() of that is unremarkable, so the safe-softmax machinery -- the
running max m_i, the whole reason online softmax tracks a maximum at all -- has
never been load-bearing. Delete `- m_new` from the kernel and those tests still
pass.

These do not. Scores here reach the hundreds, where:

  * exp(s) overflows fp32 to +inf above ~88.7, and
  * exp(s) underflows to exactly 0 below ~-88, making the denominator 0 and the
    result NaN.

Each test asserts its own premise -- that an unsafe softmax really would fail on
this input -- so none of them can quietly go vacuous if the data changes.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.autograd import flash_attention  # noqa: E402
from flash_attn.naive import naive_attention  # noqa: E402
from flash_attn.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

TOL = {torch.float16: (2e-2, 1e-2), torch.bfloat16: (8e-2, 4e-2)}
DTYPES = [torch.float16, torch.bfloat16]

# exp() in fp32 overflows above this; below its negation it underflows to zero.
EXP_OVERFLOW = 88.7


def _scores(q, k):
    return (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_large_positive_scores_do_not_overflow(dtype, causal):
    """Scores in the hundreds. Without max subtraction, exp() returns +inf."""
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    b, h, n, d = 1, 2, 256, 64

    q = torch.randn(b, h, n, d, device="cuda", dtype=dtype) * 8
    k = torch.randn(b, h, n, d, device="cuda", dtype=dtype) * 8
    v = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

    s = _scores(q, k)
    assert s.max() > EXP_OVERFLOW, f"premise failed: max score {s.max():.1f} cannot overflow exp"
    assert torch.isinf(torch.exp(s)).any(), "premise failed: unsafe exp did not overflow"

    got = flash_attention_forward(q, k, v, causal=causal)
    want = naive_attention(q.float(), k.float(), v.float(), causal=causal)

    assert torch.isfinite(got).all(), "kernel produced inf/nan on large scores"
    torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_large_negative_scores_do_not_underflow(dtype, causal):
    """Every score far below zero. Without max subtraction every exp() underflows
    to 0, the denominator becomes 0, and the output is NaN.

    Positive queries against negative keys makes the sign structural rather than
    relying on a shift, which does not reliably move a row maximum.
    """
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    b, h, n, d = 1, 2, 256, 64

    q = (torch.randn(b, h, n, d, device="cuda", dtype=dtype).abs() + 0.5) * 8
    k = -(torch.randn(b, h, n, d, device="cuda", dtype=dtype).abs() + 0.5) * 8
    v = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

    s = _scores(q, k)
    assert s.max() < -EXP_OVERFLOW, f"premise failed: max score {s.max():.1f} does not underflow"
    assert (torch.exp(s).sum(dim=-1) == 0).any(), "premise failed: unsafe denominator not zero"

    got = flash_attention_forward(q, k, v, causal=causal)
    want = naive_attention(q.float(), k.float(), v.float(), causal=causal)

    assert torch.isfinite(got).all(), "kernel produced inf/nan on large negative scores"
    torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_survives_extreme_scores(dtype):
    """The backward pass re-exponentiates via exp(s - L), so it needs the same
    protection. It also divides nothing, but a NaN in P propagates everywhere."""
    torch.manual_seed(0)
    atol, rtol = (5e-2, 2e-2) if dtype is torch.float16 else (2e-1, 8e-2)
    b, h, n, d = 1, 2, 128, 64

    q = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) * 8).requires_grad_(True)
    k = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) * 8).requires_grad_(True)
    v = torch.randn(b, h, n, d, device="cuda", dtype=dtype, requires_grad=True)

    assert _scores(q, k).max() > EXP_OVERFLOW, "premise failed: scores too small"

    ref = [t.detach().float().requires_grad_(True) for t in (q, k, v)]
    flash_attention(q, k, v, causal=True).sum().backward()
    naive_attention(*ref, causal=True).sum().backward()

    for got, want, name in zip((q, k, v), ref, "qkv"):
        assert torch.isfinite(got.grad).all(), f"d{name} contains inf/nan"
        torch.testing.assert_close(got.grad.float(), want.grad, atol=atol, rtol=rtol)


def test_logsumexp_respects_its_mathematical_bounds():
    """L = m + log(l) must satisfy  rowmax <= L <= rowmax + log(N).

    The lower bound holds because l >= 1: the row-max term contributes exp(0).
    The upper bound holds because every one of the N terms is at most 1. Both
    break immediately if the running max is wrong, and the check is meaningful
    at extreme magnitudes where a naive implementation would return inf.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) * 8
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) * 8
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    _, lse = flash_attention_forward(q, k, v, return_lse=True)
    row_max = _scores(q, k).max(dim=-1).values

    assert torch.isfinite(lse).all()
    assert (lse >= row_max - 1e-2).all(), "L below the row max implies l < 1"
    assert (lse <= row_max + math.log(n) + 1e-2).all(), "L too large implies terms above 1"


@pytest.mark.parametrize("dtype", DTYPES)
def test_moderately_large_scores(dtype):
    """Just below the overflow threshold -- the regime a real model with a large
    learning rate might transiently reach."""
    torch.manual_seed(0)
    atol, rtol = TOL[dtype]
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=dtype) * 4
    k = torch.randn(1, 2, 256, 64, device="cuda", dtype=dtype) * 4
    v = torch.randn(1, 2, 256, 64, device="cuda", dtype=dtype)

    got = flash_attention_forward(q, k, v, causal=True)
    want = naive_attention(q.float(), k.float(), v.float(), causal=True)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol)
