"""Step 6: the running row max, checked against torch.max over the full row.

This is the first step where the kernel loops over K blocks. The property under
test is that a value accumulated blockwise equals the value computed over the
whole row at once -- the induction that the rest of online softmax rests on.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.triton_fwd import running_max  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 1e-3
RTOL = 1e-3


def _reference_max(q, k, sm_scale):
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    return s.max(dim=-1).values


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),  # exactly one K block
        (2, 4, 256, 64),  # four K blocks
        pytest.param(1, 2, 1024, 64, marks=pytest.mark.slow),  # many blocks
        (1, 1, 100, 32),  # N not a multiple of BLOCK_N -> partial last block
        (2, 2, 17, 16),  # N smaller than one block
    ],
)
def test_running_max_matches_full_row(b, h, n, d):
    torch.manual_seed(0)
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got = running_max(q, k, block_m=64, block_n=64)
    want = _reference_max(q, k, 1.0 / math.sqrt(d))

    assert got.shape == (b, h, n)
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
def test_result_independent_of_kv_block_size(block_n):
    """How the row is chopped up must not change the maximum.

    This is the real content of the step: different block_n means a different
    number of loop iterations and a different accumulation order, and the
    answer has to be identical regardless.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 300, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got = running_max(q, k, block_m=64, block_n=block_n)
    want = _reference_max(q, k, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_result_independent_of_q_block_size(block_m):
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 300, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got = running_max(q, k, block_m=block_m, block_n=64)
    want = _reference_max(q, k, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_padded_keys_never_win_the_max():
    """Out-of-range keys are -inf, so they can never become the row max.

    If they were masked to 0 instead, any row whose true max is negative would
    wrongly report 0 -- a bug that only shows up on unlucky data.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 70, 32  # 70 = 64 + 6, so the last block is mostly padding

    # Every score must be negative for this test to mean anything. Shifting q by
    # a constant does NOT achieve that -- it adds -c * sum_d(k_jd), whose sign
    # varies per key, and a max over many keys then lands positive anyway.
    # Instead make the sign structural: positive queries against negative keys
    # gives an all-negative product, so a padded key masked to 0 rather than
    # -inf would win every single row max.
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16).abs() + 0.5
    k = -(torch.randn(b, h, n, d, device="cuda", dtype=torch.float16).abs() + 0.5)

    got = running_max(q, k, block_m=64, block_n=64)
    want = _reference_max(q, k, 1.0 / math.sqrt(d))

    assert (want < 0).all(), "test setup failed to produce negative maxima"
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_no_nans_and_all_finite():
    torch.manual_seed(0)
    q, k = (torch.randn(1, 2, 100, 32, device="cuda", dtype=torch.float16) for _ in range(2))
    assert torch.isfinite(running_max(q, k)).all()


def test_custom_softmax_scale():
    torch.manual_seed(0)
    q, k = (torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16) for _ in range(2))

    got = running_max(q, k, sm_scale=0.25)
    want = _reference_max(q, k, 0.25)

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)
