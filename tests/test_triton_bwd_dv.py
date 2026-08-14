"""Step 12: gradient with respect to V, tested in isolation.

Ground truth is autograd through the naive baseline, which step 2 already
verified against PyTorch's SDPA gradients. So this is checked against a
reference that is itself independently validated.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.naive import naive_attention  # noqa: E402
from flash_attn_scratch.triton_bwd import backward_dv  # noqa: E402
from flash_attn_scratch.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 2e-2
RTOL = 1e-2


def _reference_dv(q, k, v, do, causal):
    """dV from autograd through the fp32 naive baseline."""
    qf, kf, vf = (t.float().detach().requires_grad_(True) for t in (q, k, v))
    naive_attention(qf, kf, vf, causal=causal).backward(do.float())
    return vf.grad


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),
        (2, 4, 256, 64),
        pytest.param(1, 2, 1024, 64, marks=pytest.mark.slow),
        (1, 1, 100, 32),  # partial blocks in both loops
        (2, 2, 17, 16),
        (1, 2, 384, 128),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_dv_matches_autograd(b, h, n, d, causal):
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    _, lse = flash_attention_forward(q, k, v, causal=causal, return_lse=True)
    got = backward_dv(q, k, do, lse, causal=causal)
    want = _reference_dv(q, k, v, do, causal)

    assert got.shape == (b, h, n, d)
    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_dv_independent_of_block_sizes(block_m, block_n, causal):
    """Backward tiling is independent of forward tiling, and neither may change
    the gradient. Under causal masking the block geometry also decides where
    the Q loop starts, so this exercises that bound too."""
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    _, lse = flash_attention_forward(q, k, v, causal=causal, return_lse=True)
    got = backward_dv(q, k, do, lse, causal=causal, block_m=block_m, block_n=block_n)
    want = _reference_dv(q, k, v, do, causal)

    torch.testing.assert_close(got.float(), want, atol=ATOL, rtol=RTOL)


def test_causal_dv_differs_from_non_causal():
    """Guards against the causal mask silently doing nothing in backward."""
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )

    _, lse_c = flash_attention_forward(q, k, v, causal=True, return_lse=True)
    _, lse_f = flash_attention_forward(q, k, v, causal=False, return_lse=True)

    assert not torch.allclose(
        backward_dv(q, k, do, lse_c, causal=True),
        backward_dv(q, k, do, lse_f, causal=False),
        atol=1e-2,
    )


def test_first_query_routes_its_gradient_to_the_first_key_alone():
    """Under causal masking, row 0 of P is exactly [1, 0, ..., 0].

    So if only query 0 carries an incoming gradient, all of it must land on key
    0 and nowhere else: dV[0] == dO[0], and every other row of dV is zero.
    This pins down both the mask and the P^T orientation -- transposing the
    wrong way would spread the gradient across keys instead of concentrating
    it, which the aggregate comparison above can mask.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 128, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    do = torch.zeros(b, h, n, d, device="cuda", dtype=torch.float16)
    do[:, :, 0, :] = torch.randn(b, h, d, device="cuda", dtype=torch.float16)

    _, lse = flash_attention_forward(q, k, v, causal=True, return_lse=True)
    dv = backward_dv(q, k, do, lse, causal=True)

    torch.testing.assert_close(dv[:, :, 0, :].float(), do[:, :, 0, :].float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        dv[:, :, 1:, :].float(), torch.zeros_like(dv[:, :, 1:, :]).float(), atol=ATOL, rtol=RTOL
    )


def test_custom_scale_must_match_forward():
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )
    scale = 0.1

    _, lse = flash_attention_forward(q, k, v, sm_scale=scale, return_lse=True)
    got = backward_dv(q, k, do, lse, sm_scale=scale)

    qf, kf, vf = (t.float().detach().requires_grad_(True) for t in (q, k, v))
    naive_attention(qf, kf, vf, sm_scale=scale).backward(do.float())

    torch.testing.assert_close(got.float(), vf.grad, atol=ATOL, rtol=RTOL)


def test_dv_is_finite():
    torch.manual_seed(0)
    q, k, v, do = (
        torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(4)
    )
    for causal in (False, True):
        _, lse = flash_attention_forward(q, k, v, causal=causal, return_lse=True)
        assert torch.isfinite(backward_dv(q, k, do, lse, causal=causal)).all()


def test_rejects_wrong_lse_shape():
    q, k, v, do = (torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16) for _ in range(4))
    with pytest.raises(ValueError, match="lse must be"):
        backward_dv(q, k, do, torch.zeros(1, 2, 32, device="cuda"))
