"""The full correctness sweep across shapes, dtypes and causal settings.

Tolerances are per-dtype rather than one loose bound for both. bf16 carries 8
mantissa bits against fp16's 10, so it is roughly 4x less precise per value; a
single tolerance wide enough for bf16 would let real fp16 regressions through.

bf16 is included because it is what most LLM training actually uses -- its
8-bit exponent matches fp32's range, which removes the loss-scaling machinery
fp16 requires. Correctness there is not optional for an LLM-focused kernel.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn_scratch.autograd import flash_attention  # noqa: E402
from flash_attn_scratch.naive import naive_attention  # noqa: E402
from flash_attn_scratch.triton_fwd import flash_attention_forward  # noqa: E402

pytestmark = pytest.mark.gpu

# (forward atol/rtol, backward atol/rtol). Backward is looser because
# dS = P * (dP - D) cancels when dP is close to D.
TOL = {
    torch.float16: ((2e-2, 1e-2), (5e-2, 2e-2)),
    torch.bfloat16: ((8e-2, 4e-2), (2e-1, 8e-2)),
}

DTYPES = [torch.float16, torch.bfloat16]

SHAPES = [
    (1, 1, 64, 64),  # minimal
    (2, 4, 256, 64),  # typical small transformer
    (1, 2, 512, 32),  # small head dim
    (4, 8, 128, 16),  # smallest legal head dim, many heads
    (1, 2, 384, 128),  # large head dim -> 8 warps, 2 pipeline stages
    (1, 1, 100, 64),  # N not a multiple of any block size
    (3, 2, 17, 32),  # N smaller than one block
]


@pytest.mark.parametrize("b,h,n,d", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_forward_sweep(b, h, n, d, dtype, causal):
    torch.manual_seed(0)
    (atol, rtol), _ = TOL[dtype]
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=causal)
    want = naive_attention(q.float(), k.float(), v.float(), causal=causal)

    assert got.shape == (b, h, n, d) and got.dtype == dtype
    torch.testing.assert_close(got.float(), want, atol=atol, rtol=rtol)


@pytest.mark.parametrize("b,h,n,d", [(1, 1, 64, 64), (2, 4, 256, 64), (1, 1, 100, 64)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_forward_against_pytorch_sdpa(b, h, n, d, dtype, causal):
    """Second, independent reference -- PyTorch's own fused implementation."""
    torch.manual_seed(0)
    (atol, rtol), _ = TOL[dtype]
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=causal)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("b,h,n,d", [(1, 1, 64, 64), (2, 4, 256, 64), (1, 1, 100, 32), (1, 2, 384, 128)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_backward_sweep(b, h, n, d, dtype, causal):
    torch.manual_seed(0)
    _, (atol, rtol) = TOL[dtype]
    q, k, v = (
        torch.randn(b, h, n, d, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)
    )
    ref = [t.detach().float().requires_grad_(True) for t in (q, k, v)]

    flash_attention(q, k, v, causal=causal).sum().backward()
    naive_attention(*ref, causal=causal).sum().backward()

    for got, want, name in zip((q, k, v), ref, "qkv"):
        torch.testing.assert_close(
            got.grad.float(), want.grad, atol=atol, rtol=rtol, msg=f"d{name} mismatch"
        )


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_long_context(dtype):
    """2048 tokens -- the regime the project is about.

    The naive reference would allocate a 2048x2048 score matrix per head; this
    is checked against SDPA instead, which does not.
    """
    torch.manual_seed(0)
    (atol, rtol), _ = TOL[dtype]
    q, k, v = (torch.randn(1, 4, 2048, 64, device="cuda", dtype=dtype) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float64])
def test_rejects_unsupported_dtypes(bad_dtype):
    """fp32 would silently fall off the tensor-core path; fp64 is unsupported.
    Both should fail loudly at the API boundary, not deep inside Triton."""
    q, k, v = (torch.randn(1, 1, 64, 64, device="cuda", dtype=bad_dtype) for _ in range(3))
    with pytest.raises(ValueError, match="fp16 and bf16"):
        flash_attention_forward(q, k, v)


@pytest.mark.parametrize("d", [16, 32, 64, 128])
def test_every_supported_head_dim(d):
    """Head dims spanning the legal range, each with its own warp and pipeline
    configuration."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, d, device="cuda", dtype=torch.float16) for _ in range(3))

    got = flash_attention_forward(q, k, v, causal=True)
    want = naive_attention(q.float(), k.float(), v.float(), causal=True)

    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("d", [48, 96, 100])
def test_rejects_non_power_of_two_head_dims(d):
    q, k, v = (torch.randn(1, 1, 64, d, device="cuda", dtype=torch.float16) for _ in range(3))
    with pytest.raises(ValueError, match="power of two"):
        flash_attention_forward(q, k, v)


def test_deterministic_across_repeated_calls():
    """Same inputs, same output, bit for bit -- no atomics, no nondeterminism."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))

    first = flash_attention_forward(q, k, v, causal=True)
    for _ in range(3):
        assert torch.equal(flash_attention_forward(q, k, v, causal=True), first)
