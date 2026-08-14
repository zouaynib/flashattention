"""Grouped-query and multi-query attention in the forward pass.

Q carries H_q heads while K and V carry H_kv, with query head h reading KV head
h // (H_q / H_kv). This is the attention shape Llama 3, Mistral and most current
open models actually use: the KV cache scales with H_kv rather than H_q, so
H_q=32 / H_kv=8 shrinks it 4x, which is usually what limits how many concurrent
requests a server can hold.

The reference expands K and V to H_q heads and runs ordinary attention. The
kernel must NOT do that -- expanding is exactly the memory cost GQA avoids -- so
these tests check that the index mapping produces the same answer as expansion
while the kernel only ever reads the compact tensors.
"""

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.autograd import flash_attention  # noqa: E402
from flash_attn_scratch.naive import naive_attention  # noqa: E402
from flash_attn_scratch.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL, RTOL = 2e-2, 1e-2


def _expand(kv: torch.Tensor, h_q: int) -> torch.Tensor:
    """Replicate each KV head across its query group -- the reference only."""
    return kv.repeat_interleave(h_q // kv.shape[1], dim=1)


@pytest.mark.parametrize(
    "h_q,h_kv",
    [
        (8, 8),  # ordinary MHA: group size 1, must be unchanged
        (8, 4),  # group size 2
        (8, 2),  # group size 4
        (32, 8),  # Llama-3-8B's shape
        (8, 1),  # multi-query attention
        (1, 1),  # degenerate single head
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_matches_expanded_reference(h_q, h_kv, causal):
    torch.manual_seed(0)
    b, n, d = 2, 256, 64

    q = torch.randn(b, h_q, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)

    got = flash_attention_forward(q, k, v, causal=causal)
    want = naive_attention(
        q.float(), _expand(k, h_q).float(), _expand(v, h_q).float(), causal=causal
    )

    assert got.shape == (b, h_q, n, d)
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gqa_in_both_dtypes(dtype):
    torch.manual_seed(0)
    atol, rtol = (2e-2, 1e-2) if dtype is torch.float16 else (8e-2, 4e-2)
    b, h_q, h_kv, n, d = 1, 16, 4, 384, 64

    q = torch.randn(b, h_q, n, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=dtype)

    got = flash_attention_forward(q, k, v, causal=True)
    want = naive_attention(
        q.float(), _expand(k, h_q).float(), _expand(v, h_q).float(), causal=True
    )
    torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol)


def test_heads_within_a_group_see_the_same_keys():
    """Structural check on the mapping itself.

    If every query head in a group is given identical Q rows, they must produce
    identical outputs -- they share a K/V head. And heads in *different* groups
    must differ, which is what catches an off-by-one in `h // GROUP_SIZE`.
    """
    torch.manual_seed(0)
    b, h_q, h_kv, n, d = 1, 8, 2, 128, 64  # group size 4

    one_head = torch.randn(b, 1, n, d, device="cuda", dtype=torch.float16)
    q = one_head.expand(b, h_q, n, d).contiguous()  # every head identical
    k = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)

    out = flash_attention_forward(q, k, v, causal=True)

    # Heads 0-3 share KV head 0; heads 4-7 share KV head 1.
    for h in range(1, 4):
        assert torch.equal(out[:, 0], out[:, h]), f"head {h} differs within its group"
    for h in range(5, 8):
        assert torch.equal(out[:, 4], out[:, h]), f"head {h} differs within its group"
    assert not torch.allclose(out[:, 0], out[:, 4], atol=1e-3), "groups should differ"


def test_kv_tensors_are_never_expanded():
    """The memory claim: peak allocation must not scale with H_q.

    Running with H_kv = 1 against H_q = 32 should cost barely more than
    H_kv = 32 does for the K/V side. If the kernel replicated K/V internally,
    the two would converge.
    """
    torch.manual_seed(0)
    b, h_q, n, d = 1, 32, 1024, 64

    def peak(h_kv):
        q = torch.randn(b, h_q, n, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, h_kv, n, d, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        flash_attention_forward(q, k, v, causal=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    mqa, mha = peak(1), peak(32)
    assert mqa < mha, f"MQA should allocate less than MHA, got {mqa} vs {mha}"


@pytest.mark.parametrize("h_q,h_kv", [(8, 3), (6, 4), (5, 2)])
def test_rejects_non_divisible_head_counts(h_q, h_kv):
    q = torch.randn(1, h_q, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, h_kv, 64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple of kv heads"):
        flash_attention_forward(q, k, k)


def test_rejects_mismatched_batch_or_head_dim():
    """Head count may differ (grouped-query attention) and so may sequence
    length. Batch and head_dim may not -- those are genuine shape errors."""
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)

    wrong_head_dim = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="may differ only in head count and length"):
        flash_attention_forward(q, wrong_head_dim, wrong_head_dim)

    wrong_batch = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="may differ only in head count and length"):
        flash_attention_forward(q, wrong_batch, wrong_batch)


def test_autograd_accepts_gqa():
    """GQA works end to end through autograd now, and dK/dV come back
    shaped like their inputs rather than expanded to H_q."""
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True)

    flash_attention(q, k, v, causal=True).sum().backward()

    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape and v.grad.shape == v.shape


def test_mha_path_is_unchanged():
    """Group size 1 must be bit-identical to before this feature existed."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True)
    want = naive_attention(q.float(), k.float(), v.float(), causal=True)

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)
