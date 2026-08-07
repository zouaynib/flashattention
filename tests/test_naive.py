"""Correctness of the naive baseline against PyTorch's own SDPA.

Everything downstream is compared against `naive_attention`, so it has to be
right first. These run on CPU in fp32: the point is to validate the algorithm,
not the numerics of any particular dtype.
"""

import pytest
import torch
import torch.nn.functional as F

from flash_attn.naive import naive_attention

SHAPES = [
    (1, 1, 8, 8),  # smallest case that exercises the math
    (2, 4, 64, 32),
    (1, 2, 128, 64),  # non-square N vs D
    (3, 1, 33, 16),  # N not a multiple of any natural block size
]


@pytest.mark.parametrize("b,h,n,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(b, h, n, d, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, dtype=torch.float32) for _ in range(3))

    got = naive_attention(q, k, v, causal=causal)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_gradients_match_sdpa(causal):
    """Autograd through the baseline gives the reference gradients that the
    hand-written backward kernel (steps 12-13) will be checked against."""
    torch.manual_seed(0)
    tensors = [torch.randn(2, 2, 32, 16, dtype=torch.float32, requires_grad=True) for _ in range(3)]
    q, k, v = tensors
    ref = [t.detach().clone().requires_grad_(True) for t in tensors]

    naive_attention(q, k, v, causal=causal).sum().backward()
    F.scaled_dot_product_attention(*ref, is_causal=causal).sum().backward()

    for got, want, name in zip(tensors, ref, "qkv"):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-5, msg=f"d{name} mismatch")


def test_causal_masks_the_future():
    """A causal row must be unaffected by keys past its own position."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 16, 8, dtype=torch.float32) for _ in range(3))

    full = naive_attention(q, k, v, causal=True)
    # Row 0 attends only to position 0, so it is exactly v[0] regardless of the rest.
    torch.testing.assert_close(full[0, 0, 0], v[0, 0, 0], atol=1e-6, rtol=1e-6)


def test_rejects_mismatched_shapes():
    q = torch.randn(1, 1, 8, 8)
    with pytest.raises(ValueError, match="share a shape"):
        naive_attention(q, torch.randn(1, 1, 8, 4), q)
