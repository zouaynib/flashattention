"""Step 9: the assembled forward pass, end to end.

Checked against both references: the naive baseline from step 2 (which is
transparently correct) and PyTorch's own SDPA (which is independently
implemented). Agreeing with both is a much stronger claim than agreeing with
either alone.
"""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.naive import naive_attention  # noqa: E402
from flash_attn.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 2e-2
RTOL = 1e-2


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),
        (2, 4, 256, 64),
        (1, 2, 1024, 64),
        (4, 8, 512, 32),
        (1, 1, 100, 32),  # partial last Q block
        (2, 2, 17, 16),  # N smaller than one block
        (1, 2, 384, 128),  # large head dim -> 8 warps
    ],
)
def test_matches_naive_baseline(b, h, n, d):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v)
    want = naive_attention(q.float(), k.float(), v.float())

    assert got.shape == (b, h, n, d)
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("b,h,n,d", [(1, 1, 128, 64), (2, 4, 512, 64), (1, 2, 256, 32)])
def test_matches_pytorch_sdpa(b, h, n, d):
    """Against an independent implementation, not just our own reference."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_long_sequence():
    """4096 tokens: the regime the project is actually about.

    The naive baseline materializes a 4096x4096 score matrix per head here;
    this kernel never allocates one.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 4096, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_output_dtype_matches_input():
    q, k, v = (torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    assert flash_attention_forward(q, k, v).dtype == torch.float16


@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64), (64, 128)])
def test_block_sizes_do_not_change_the_answer(block_m, block_n):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, block_m=block_m, block_n=block_n)
    want = naive_attention(q.float(), k.float(), v.float())

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_handles_noncontiguous_input():
    """Transformers produce (B, N, H, D) and transpose. That view is not
    contiguous, and real callers pass it in directly."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
        for _ in range(3)
    )
    assert not q.is_contiguous()

    got = flash_attention_forward(q, k, v)
    want = naive_attention(q.float(), k.float(), v.float())

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_custom_softmax_scale():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, sm_scale=0.1)
    want = naive_attention(q.float(), k.float(), v.float(), sm_scale=0.1)

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_default_scale_is_one_over_sqrt_d():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    explicit = flash_attention_forward(q, k, v, sm_scale=1.0 / math.sqrt(64))
    default = flash_attention_forward(q, k, v)

    torch.testing.assert_close(default, explicit)


def test_rejects_mixed_dtypes():
    q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="share a dtype"):
        flash_attention_forward(q, q, v)


def test_rejects_mismatched_shapes():
    q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="share a shape"):
        flash_attention_forward(q, q, v)
