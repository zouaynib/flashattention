"""Step 7: the online softmax denominator, with rescaling.

The claim under test: l accumulated blockwise, with alpha = exp(m_old - m_new)
correcting each time the max grows, equals sum_j exp(s_ij - m_i) over the whole
row. Get alpha wrong and l is silently too large -- which later shows up as
attention weights that do not sum to 1.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.triton_fwd import running_softmax_stats  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 1e-3
RTOL = 1e-3


def _reference_stats(q, k, sm_scale):
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    m = s.max(dim=-1).values
    l = torch.exp(s - m[..., None]).sum(dim=-1)
    return m, l


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),  # single block: alpha is never exercised
        (2, 4, 256, 64),  # four blocks
        (1, 2, 1024, 64),  # many blocks, more chances to rescale
        (1, 1, 100, 32),  # partial last block
        (2, 2, 17, 16),  # N smaller than one block
    ],
)
def test_stats_match_full_row(b, h, n, d):
    torch.manual_seed(0)
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got_m, got_l = running_softmax_stats(q, k, block_m=64, block_n=64)
    want_m, want_l = _reference_stats(q, k, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got_m, want_m, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(got_l, want_l, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("spike_at", ["first", "last"])
def test_rescaling_handles_a_late_maximum(spike_at):
    """The test that actually exercises alpha.

    A single dominant key is planted either in the first K block or the last.
    With the spike last, the running max jumps near the end and every previously
    accumulated term must be rescaled by alpha. Drop the rescaling and this
    case is wildly wrong while the 'first' case still looks fine -- which is
    exactly how this bug hides.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16).abs() + 0.5
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    idx = 0 if spike_at == "first" else n - 1
    k[:, :, idx, :] = 4.0  # huge positive score against all-positive queries

    got_m, got_l = running_softmax_stats(q, k, block_m=64, block_n=64)
    want_m, want_l = _reference_stats(q, k, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got_m, want_m, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(got_l, want_l, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
def test_result_independent_of_kv_block_size(block_n):
    """Block size changes how many times alpha is applied. The answer must not
    change -- that is the whole correctness argument for online softmax."""
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 300, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got_m, got_l = running_softmax_stats(q, k, block_m=64, block_n=block_n)
    want_m, want_l = _reference_stats(q, k, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got_m, want_m, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(got_l, want_l, atol=ATOL, rtol=RTOL)


def test_denominator_normalizes_softmax_to_one():
    """The point of l: exp(s - m) / l must sum to exactly 1 across the row."""
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 256, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    m, l = running_softmax_stats(q, k, block_m=64, block_n=64)

    s = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(d))
    weights = torch.exp(s - m[..., None]) / l[..., None]

    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(b, h, n, device="cuda"), atol=ATOL, rtol=RTOL
    )


def test_padded_keys_contribute_nothing_to_the_sum():
    """With N not a multiple of BLOCK_N, l must count exactly N terms.

    Padded keys masked to -inf give exp(-inf - m) == 0. Masked to 0 they would
    each add exp(-m) to the denominator, inflating it.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 70, 32  # last block is 58/64 padding
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    _, got_l = running_softmax_stats(q, k, block_m=64, block_n=64)
    _, want_l = _reference_stats(q, k, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got_l, want_l, atol=ATOL, rtol=RTOL)
    # A denominator can never exceed the number of real keys (each term <= 1).
    assert (got_l <= n + ATOL).all()


def test_all_finite():
    torch.manual_seed(0)
    q, k = (torch.randn(1, 2, 100, 32, device="cuda", dtype=torch.float16) for _ in range(2))
    for tensor in running_softmax_stats(q, k):
        assert torch.isfinite(tensor).all()
