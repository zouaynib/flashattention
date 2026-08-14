"""Safe softmax over a single tile.

Not online yet -- the denominator here covers one K block, not the whole row.
These tests check that the tile-local softmax is exactly right, so that when
the running max, sum and accumulator add cross-block bookkeeping, any failure is in the bookkeeping.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.triton_fwd import softmax_tile  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 1e-4
RTOL = 1e-3


def _reference_scores(q, k, start_m, start_n, block_m, block_n, sm_scale):
    qs = q[:, :, start_m * block_m : (start_m + 1) * block_m, :].float()
    ks = k[:, :, start_n * block_n : (start_n + 1) * block_n, :].float()
    return (qs @ ks.transpose(-2, -1)) * sm_scale


@pytest.mark.parametrize("start_m,start_n", [(0, 0), (1, 2), (3, 1)])
def test_normalized_p_matches_torch_softmax(start_m, start_n):
    torch.manual_seed(0)
    b, h, n, d = 2, 2, 256, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    p, _, l = softmax_tile(q, k, start_m=start_m, start_n=start_n, block_m=64, block_n=64)
    s = _reference_scores(q, k, start_m, start_n, 64, 64, 1.0 / math.sqrt(d))

    torch.testing.assert_close(p / l[..., None], torch.softmax(s, dim=-1), atol=ATOL, rtol=RTOL)


def test_m_and_l_match_reference():
    """m is the row max; l is the row sum of exp(s - m) over this tile."""
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 128, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    _, m, l = softmax_tile(q, k, block_m=64, block_n=64)
    s = _reference_scores(q, k, 0, 0, 64, 64, 1.0 / math.sqrt(d))

    want_m = s.max(dim=-1).values
    want_l = torch.exp(s - want_m[..., None]).sum(dim=-1)

    torch.testing.assert_close(m, want_m, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(l, want_l, atol=ATOL, rtol=RTOL)


def test_p_is_unnormalized():
    """P must be exp(s - m), NOT the normalized softmax.

    FlashAttention carries numerator and denominator separately; l is still
    accumulating over later K blocks, so normalizing here would be wrong.
    """
    torch.manual_seed(0)
    q, k = (torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16) for _ in range(2))

    p, _, l = softmax_tile(q, k, block_m=64, block_n=64)

    # The row max entry is exp(0) == 1 exactly, which normalized softmax never is.
    torch.testing.assert_close(p.max(dim=-1).values, torch.ones_like(l), atol=ATOL, rtol=RTOL)
    # Rows do not sum to 1 -- they sum to l.
    torch.testing.assert_close(p.sum(dim=-1), l, atol=ATOL, rtol=RTOL)
    assert not torch.allclose(p.sum(dim=-1), torch.ones_like(l), atol=1e-2)


def test_out_of_range_keys_get_exactly_zero_weight():
    """The -inf masking test. Padded key columns must contribute nothing.

    If they were masked to 0 instead of -inf they would become exp(0 - m) > 0
    and steal probability mass from real keys -- silently, and only when
    N_CTX is not a multiple of BLOCK_N.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 100, 32  # 100 = 64 + 36
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    p, _, l = softmax_tile(q, k, start_m=0, start_n=1, block_m=64, block_n=64)

    valid_n = n - 64  # 36 real keys in this block
    assert torch.all(p[:, :, :, valid_n:] == 0), "padded keys received nonzero weight"

    # And the valid part is a correct softmax over just those 36 keys.
    s = _reference_scores(q, k, 0, 1, 64, 64, 1.0 / math.sqrt(d))[:, :, :, :valid_n]
    got = p[:, :, :, :valid_n] / l[..., None]
    torch.testing.assert_close(got, torch.softmax(s, dim=-1), atol=ATOL, rtol=RTOL)


def test_no_nans_or_infs():
    torch.manual_seed(0)
    q, k = (torch.randn(1, 2, 100, 32, device="cuda", dtype=torch.float16) for _ in range(2))
    for tensor in softmax_tile(q, k, start_n=1, block_m=64, block_n=64):
        assert torch.isfinite(tensor).all()


def test_shapes_and_dtypes():
    q, k = (torch.randn(2, 3, 128, 64, device="cuda", dtype=torch.float16) for _ in range(2))
    p, m, l = softmax_tile(q, k, block_m=32, block_n=64)

    assert p.shape == (2, 3, 32, 64)
    assert m.shape == l.shape == (2, 3, 32)
    assert p.dtype == m.dtype == l.dtype == torch.float32
