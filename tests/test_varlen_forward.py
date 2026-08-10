"""Step 20: differing query and key lengths in the forward pass.

M queries against N keys. The case that matters is M < N -- KV-cache decoding,
where the M new queries are the LAST M positions of an N-long sequence and must
see every cached key before them.

Causal masking is aligned BOTTOM-RIGHT: query row m attends to keys
j <= m + (N - M). Top-left alignment (j <= m) is the other plausible reading and
is wrong for decoding -- it would let a freshly generated token attend to almost
nothing. PyTorch's `is_causal=True` is ambiguous on non-square inputs and has
shifted between versions, so these tests build the mask explicitly rather than
depending on it.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.autograd import flash_attention  # noqa: E402
from flash_attn.naive import naive_attention  # noqa: E402
from flash_attn.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL, RTOL = 2e-2, 1e-2


@pytest.mark.parametrize(
    "m,n",
    [
        (256, 256),  # square: must be unchanged
        (1, 256),  # single-token decode against a full cache
        (1, 1),  # degenerate
        (16, 256),  # chunked prefill
        (128, 300),  # neither is a block multiple
        (100, 128),
        (255, 256),  # off by one
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_matches_naive(m, n, causal):
    torch.manual_seed(0)
    b, h, d = 2, 4, 64

    q = torch.randn(b, h, m, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    got = flash_attention_forward(q, k, v, causal=causal)
    want = naive_attention(q.float(), k.float(), v.float(), causal=causal)

    assert got.shape == (b, h, m, d)
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_single_query_sees_every_cached_key():
    """The decode case, checked directly rather than via a reference.

    With M = 1 the causal mask must be vacuous: the one query is the newest
    token and every cached key precedes it. If the mask were top-left aligned,
    this query would see only key 0 and the output would equal v[0].
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 128, 64

    q = torch.randn(b, h, 1, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    causal = flash_attention_forward(q, k, v, causal=True)
    full = flash_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(causal.float(), full.float(), atol=ATOL, rtol=RTOL)
    # And it is emphatically not the top-left reading, which would give v[0].
    assert not torch.allclose(causal[:, :, 0].float(), v[:, :, 0].float(), atol=1e-1)


def test_decoding_one_token_at_a_time_matches_a_single_full_pass():
    """The property KV-cache generation depends on.

    Running the full sequence in one causal pass, or feeding one query at a time
    against a growing cache, must give the same answer for every position. This
    is what makes incremental decoding valid, and it fails immediately if the
    causal offset is wrong by even one.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 64, 64
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    full = flash_attention_forward(q, k, v, causal=True)

    for pos in (0, 1, 17, 63):
        step = flash_attention_forward(
            q[:, :, pos : pos + 1], k[:, :, : pos + 1], v[:, :, : pos + 1], causal=True
        )
        torch.testing.assert_close(
            step.float(), full[:, :, pos : pos + 1].float(), atol=ATOL, rtol=RTOL,
            msg=f"incremental decode disagrees at position {pos}",
        )


def test_chunked_prefill_matches_a_single_pass():
    """Same property for chunks rather than single tokens -- how real servers
    prefill a long prompt without one huge activation."""
    torch.manual_seed(0)
    b, h, n, d, chunk = 1, 2, 256, 64, 64
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    full = flash_attention_forward(q, k, v, causal=True)

    for start in range(0, n, chunk):
        stop = start + chunk
        piece = flash_attention_forward(
            q[:, :, start:stop], k[:, :, :stop], v[:, :, :stop], causal=True
        )
        torch.testing.assert_close(
            piece.float(), full[:, :, start:stop].float(), atol=ATOL, rtol=RTOL,
            msg=f"chunk starting at {start} disagrees",
        )


def test_varlen_with_gqa():
    """Both shape generalizations at once, which is what a real decode step is:
    few queries, many cached keys, fewer KV heads than query heads."""
    torch.manual_seed(0)
    b, h_q, h_kv, m, n, d = 1, 32, 8, 4, 512, 64

    q = torch.randn(b, h_q, m, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)

    got = flash_attention_forward(q, k, v, causal=True)
    want = naive_attention(
        q.float(),
        k.repeat_interleave(h_q // h_kv, dim=1).float(),
        v.repeat_interleave(h_q // h_kv, dim=1).float(),
        causal=True,
    )
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_lse_shape_follows_the_query_length():
    q = torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16)
    _, lse = flash_attention_forward(q, k, k, causal=True, return_lse=True)
    assert lse.shape == (1, 2, 8)


def test_rejects_causal_with_more_queries_than_keys():
    """The offset would go negative and the earliest rows would attend to
    nothing, leaving m = -inf and alpha = NaN."""
    q = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="at least as many keys as queries"):
        flash_attention_forward(q, k, k, causal=True)


def test_more_queries_than_keys_is_fine_without_causal():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)

    got = flash_attention_forward(q, k, v, causal=False)
    want = naive_attention(q.float(), k.float(), v.float(), causal=False)
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_autograd_rejects_varlen_for_now():
    q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    with pytest.raises(NotImplementedError, match="step 21"):
        flash_attention(q, k, k)
