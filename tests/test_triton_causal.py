"""Step 10: causal masking, tested separately from the unmasked case.

Causal is the LLM-relevant configuration -- every decoder-only transformer uses
it. It is also where the loop bound and the mask have to agree exactly: an
off-by-one in either leaks information from the future or drops the diagonal,
and both errors still produce plausible-looking numbers.
"""

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
        (1, 1, 64, 64),  # exactly one block: entirely diagonal
        (2, 4, 256, 64),  # several blocks
        (1, 2, 1024, 64),  # many blocks skipped above the diagonal
        (1, 1, 100, 32),  # N not a multiple of the block size
        (2, 2, 17, 16),  # N smaller than one block
        (1, 2, 384, 128),  # large head dim
    ],
)
def test_causal_matches_naive_baseline(b, h, n, d):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True)
    want = naive_attention(q.float(), k.float(), v.float(), causal=True)

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("b,h,n,d", [(1, 1, 128, 64), (2, 4, 512, 64), (1, 2, 300, 32)])
def test_causal_matches_pytorch_sdpa(b, h, n, d):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_first_row_attends_only_to_itself():
    """Row 0 has exactly one valid key, so its output must equal v[0] exactly.

    Catches a mask off-by-one that shape-level tests cannot see.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True)

    torch.testing.assert_close(got[:, :, 0, :].float(), v[:, :, 0, :].float(), atol=ATOL, rtol=RTOL)


def test_future_tokens_cannot_influence_the_past():
    """The defining property of causality, tested directly.

    Perturb the last token's key and value. Every output before it must be
    bit-identical. If the loop bound or the mask is wrong by even one column,
    information leaks backwards and this fails.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    before = flash_attention_forward(q, k, v, causal=True)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, -1, :] = 5.0
    v2[:, :, -1, :] = -5.0
    after = flash_attention_forward(q, k2, v2, causal=True)

    assert torch.equal(before[:, :, :-1, :], after[:, :, :-1, :]), "future token leaked into past"
    assert not torch.equal(before[:, :, -1, :], after[:, :, -1, :]), "last row should have changed"


def test_causal_differs_from_non_causal():
    """Guards against the mask silently doing nothing."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    causal = flash_attention_forward(q, k, v, causal=True)
    full = flash_attention_forward(q, k, v, causal=False)

    assert not torch.allclose(causal, full, atol=1e-2)
    # The last row sees everything either way, so it must agree.
    torch.testing.assert_close(
        causal[:, :, -1, :].float(), full[:, :, -1, :].float(), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64)])
def test_causal_independent_of_block_sizes(block_m, block_n):
    """Block geometry changes which blocks are skipped and where the diagonal
    falls inside a tile. The answer must not move."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True, block_m=block_m, block_n=block_n)
    want = naive_attention(q.float(), k.float(), v.float(), causal=True)

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_causal_produces_no_nans():
    """The trap from step 7: a fully-masked block would give alpha = NaN.

    Bounding the loop at the diagonal is what prevents it, so this is the
    regression test for that reasoning.
    """
    torch.manual_seed(0)
    for n in (16, 17, 64, 65, 100, 128, 129, 512):
        q, k, v = (torch.randn(1, 1, n, 32, device="cuda", dtype=torch.float16) for _ in range(3))
        out = flash_attention_forward(q, k, v, causal=True)
        assert torch.isfinite(out).all(), f"non-finite output at N={n}"
