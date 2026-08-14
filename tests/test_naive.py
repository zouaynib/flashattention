"""Correctness of the naive baseline against PyTorch's own SDPA.

Everything downstream is compared against `naive_attention`, so it has to be
right first. These run on CPU in fp32: the point is to validate the algorithm,
not the numerics of any particular dtype.
"""

import pytest
import torch
import torch.nn.functional as F

from flash_attn_scratch.naive import naive_attention, standard_attention

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


# --- standard_attention: the benchmark baseline -------------------------------


@pytest.mark.parametrize("b,h,n,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_standard_matches_naive(b, h, n, d, causal):
    """Same math, fewer intermediates. The answers must be identical."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, dtype=torch.float32) for _ in range(3))

    got = standard_attention(q, k, v, causal=causal)
    want = naive_attention(q, k, v, causal=causal)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("b,h,n,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_standard_matches_sdpa(b, h, n, d, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, dtype=torch.float32) for _ in range(3))

    got = standard_attention(q, k, v, causal=causal)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_standard_gradients_match_naive(causal):
    """The benchmark baseline is also used for backward timings, so its
    gradients have to be right too."""
    torch.manual_seed(0)
    tensors = [torch.randn(2, 2, 32, 16, dtype=torch.float32, requires_grad=True) for _ in range(3)]
    ref = [t.detach().clone().requires_grad_(True) for t in tensors]

    standard_attention(*tensors, causal=causal).sum().backward()
    naive_attention(*ref, causal=causal).sum().backward()

    for got, want, name in zip(tensors, ref, "qkv"):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-5, rtol=1e-5, msg=f"d{name}")


def test_standard_rejects_mismatched_shapes():
    q = torch.randn(1, 1, 8, 8)
    with pytest.raises(ValueError, match="share a shape"):
        standard_attention(q, torch.randn(1, 1, 8, 4), q)


@pytest.mark.gpu
def test_standard_uses_less_peak_memory_than_explicit():
    """The reason this baseline exists, asserted with a measurement.

    The explicit version allocates roughly four N x N-scale tensors against
    this one's two. Benchmarking the Triton kernel against the explicit version
    would inflate the result with our own inefficiency rather than measuring
    attention.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 1024, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    def peak_bytes(fn):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn(q, k, v, causal=True)
        return torch.cuda.max_memory_allocated()

    explicit = peak_bytes(naive_attention)
    standard = peak_bytes(standard_attention)

    assert standard < explicit, (
        f"expected the fused baseline to allocate less: "
        f"explicit={explicit / 2**20:.0f} MiB, standard={standard / 2**20:.0f} MiB"
    )
