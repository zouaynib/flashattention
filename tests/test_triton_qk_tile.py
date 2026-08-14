"""One tile of S = QK^T / sqrt(d), against a manual PyTorch slice.

Tolerances are tight on purpose. Both sides consume the *same* fp16 tensors,
so fp16 input rounding is common to both and cancels; tensor cores multiply
fp16 pairs into exact fp32 products and accumulate in fp32, exactly as the
fp32 reference does. The only remaining difference is summation order, worth
~1e-6. A tolerance of 1e-3 therefore still fails loudly on a real bug (a
missing transpose or a wrong stride moves values by O(1)), which a sloppy
1e-2 tolerance might let through.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.triton_fwd import qk_tile  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 1e-3
RTOL = 1e-3


def _reference_tile(q, k, start_m, start_n, block_m, block_n, sm_scale):
    """The PyTorch slice the kernel tile should equal."""
    qs = q[:, :, start_m * block_m : (start_m + 1) * block_m, :].float()
    ks = k[:, :, start_n * block_n : (start_n + 1) * block_n, :].float()
    return (qs @ ks.transpose(-2, -1)) * sm_scale


@pytest.mark.parametrize("start_m,start_n", [(0, 0), (0, 1), (2, 0), (1, 3)])
def test_tile_matches_reference(start_m, start_n):
    torch.manual_seed(0)
    b, h, n, d = 2, 4, 256, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got = qk_tile(q, k, start_m=start_m, start_n=start_n, block_m=64, block_n=64)
    want = _reference_tile(q, k, start_m, start_n, 64, 64, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64)])
def test_tile_shapes(block_m, block_n):
    """Tile geometry is a free parameter; results must not depend on it."""
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got = qk_tile(q, k, block_m=block_m, block_n=block_n)
    want = _reference_tile(q, k, 0, 0, block_m, block_n, 1.0 / math.sqrt(d))

    assert got.shape == (b, h, block_m, block_n)
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_partial_tile_valid_region():
    """When N is not a multiple of the block size the final tile is padded.

    Padded rows/cols are meaningless, but every in-range entry must still be
    exact -- zero-padding must not bleed into valid results.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 100, 32  # 100 = 64 + 36, so tile (1, 1) is partial
    q, k = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(2))

    got = qk_tile(q, k, start_m=1, start_n=1, block_m=64, block_n=64)
    want = _reference_tile(q, k, 1, 1, 64, 64, 1.0 / math.sqrt(d))  # 36x36 valid

    valid = n - 64
    torch.testing.assert_close(got[:, :, :valid, :valid], want, atol=ATOL, rtol=RTOL)
    # Everything past the valid region is zero, from the masked loads.
    assert torch.all(got[:, :, valid:, :] == 0)
    assert torch.all(got[:, :, :, valid:] == 0)


def test_custom_softmax_scale():
    torch.manual_seed(0)
    q, k = (torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16) for _ in range(2))

    got = qk_tile(q, k, sm_scale=0.25)
    want = _reference_tile(q, k, 0, 0, 64, 64, 0.25)

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_output_is_fp32():
    """fp16 inputs, fp32 scores: the running softmax stats depend on this."""
    q, k = (torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16) for _ in range(2))
    assert qk_tile(q, k).dtype == torch.float32


@pytest.mark.parametrize("block_m,block_n", [(8, 64), (64, 8)])
def test_rejects_blocks_below_tensor_core_minimum(block_m, block_n):
    q, k = (torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16) for _ in range(2))
    with pytest.raises(ValueError, match=">= 16"):
        qk_tile(q, k, block_m=block_m, block_n=block_n)


def test_rejects_mismatched_shapes():
    q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="share a shape"):
        qk_tile(q, k)
