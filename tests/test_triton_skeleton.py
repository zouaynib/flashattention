"""Prove the Triton launch and addressing path works.

`importorskip` runs at collection time, so this file is skipped cleanly on a
machine without Triton (macOS) instead of erroring during import.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.triton_fwd import copy_q  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 128, 64),  # exactly one block
        (2, 4, 512, 64),  # several blocks, several heads
        (1, 2, 200, 32),  # N not a multiple of BLOCK_M -> partial last block
        (1, 1, 7, 16),  # N smaller than one block
        (3, 2, 1024, 128),  # larger head_dim
    ],
)
def test_copy_is_bit_exact(b, h, n, d):
    torch.manual_seed(0)
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    assert torch.equal(copy_q(q), q)


def test_handles_noncontiguous_input():
    """Strides are passed explicitly, so a transposed view must still work.
    Real attention code hits this constantly via (B, N, H, D) -> (B, H, N, D)."""
    torch.manual_seed(0)
    q = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
    assert not q.is_contiguous()
    assert torch.equal(copy_q(q), q)


@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_block_size_does_not_change_result(block_m):
    """Tiling is an implementation detail; the result must not depend on it."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16)
    assert torch.equal(copy_q(q, block_m=block_m), q)


def test_rejects_non_power_of_two_head_dim():
    q = torch.randn(1, 1, 64, 48, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="power of two"):
        copy_q(q)
