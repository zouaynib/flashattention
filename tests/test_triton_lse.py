"""Step 11: the saved forward statistic, L = m + log(l).

L is what makes the backward pass memory-efficient. Instead of storing the
N x N probability matrix, we store one fp32 per query row and rebuild any
probability on demand as p_ij = exp(s_ij - L_i).
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 2e-3
RTOL = 1e-3


def _reference_lse(q, k, causal, sm_scale):
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.logsumexp(s, dim=-1)


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),
        (2, 4, 256, 64),
        pytest.param(1, 2, 1024, 64, marks=pytest.mark.slow),
        (1, 1, 100, 32),  # partial last block
        (2, 2, 17, 16),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_lse_matches_torch_logsumexp(b, h, n, d, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    _, lse = flash_attention_forward(q, k, v, causal=causal, return_lse=True)
    want = _reference_lse(q, k, causal, 1.0 / math.sqrt(d))

    assert lse.shape == (b, h, n)
    torch.testing.assert_close(lse, want, atol=ATOL, rtol=RTOL)


def test_lse_reconstructs_the_probability_matrix():
    """The identity the backward pass depends on: exp(s - L) is softmax(s).

    If this holds, the backward kernel can rebuild any block of P from Q, K and
    a single number per row -- no stored attention matrix.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    _, lse = flash_attention_forward(q, k, v, return_lse=True)

    s = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(d))
    rebuilt = torch.exp(s - lse[..., None])

    torch.testing.assert_close(rebuilt, torch.softmax(s, dim=-1), atol=ATOL, rtol=RTOL)
    # And therefore each row is a probability distribution.
    torch.testing.assert_close(
        rebuilt.sum(dim=-1), torch.ones(b, h, n, device="cuda"), atol=ATOL, rtol=RTOL
    )


def test_lse_storage_is_linear_not_quadratic():
    """The memory claim, asserted rather than described.

    L holds one value per query row. The probability matrix it replaces holds N
    per row, so the saving grows with context length.
    """
    b, h, n, d = 1, 4, 1024, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    _, lse = flash_attention_forward(q, k, v, return_lse=True)

    assert lse.numel() == b * h * n
    saved_bytes = lse.numel() * lse.element_size()
    p_matrix_bytes = b * h * n * n * 2  # what storing P in fp16 would cost
    assert saved_bytes * 100 < p_matrix_bytes, "L should be orders of magnitude smaller than P"


def test_return_lse_false_returns_a_bare_tensor():
    """The inference path must not change shape of its return value."""
    q, k, v = (torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out = flash_attention_forward(q, k, v)
    assert isinstance(out, torch.Tensor)


def test_output_is_identical_whether_or_not_lse_is_returned():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    bare = flash_attention_forward(q, k, v, causal=True)
    with_lse, _ = flash_attention_forward(q, k, v, causal=True, return_lse=True)

    assert torch.equal(bare, with_lse)


def test_lse_is_fp32_and_finite():
    """fp32 regardless of input dtype: L is exponentiated in the backward pass,
    where fp16 rounding would be amplified."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    for causal in (False, True):
        _, lse = flash_attention_forward(q, k, v, causal=causal, return_lse=True)
        assert lse.dtype == torch.float32
        assert torch.isfinite(lse).all()
