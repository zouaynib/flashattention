"""Step 8: the running output accumulator -- online softmax, complete.

Tolerances are looser here than in steps 4-7, for a substantive reason. Until
now both sides consumed identical fp16 tensors, so fp16 rounding was common to
both and cancelled, which let the earlier tests hold 1e-3. This kernel casts P
to fp16 internally (to reach tensor cores) while the fp32 reference does not,
so the difference is a real consequence of a deliberate design choice rather
than noise. 2e-2 is the honest size of that choice.
"""

import math

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")

from flash_attn.naive import naive_attention  # noqa: E402
from flash_attn.triton_fwd import running_output_accumulator  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 2e-2
RTOL = 1e-2


def _reference(q, k, v, sm_scale):
    """Unnormalized accumulator and stats, in fp32."""
    s = (q.float() @ k.float().transpose(-2, -1)) * sm_scale
    m = s.max(dim=-1).values
    p = torch.exp(s - m[..., None])
    return p @ v.float(), m, p.sum(dim=-1)


@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 64, 64),
        (2, 4, 256, 64),
        (1, 2, 1024, 64),
        (1, 1, 100, 32),  # partial last block
        (2, 2, 17, 16),  # N smaller than one block
        (1, 1, 128, 128),  # larger head dim
    ],
)
def test_normalized_output_matches_naive_attention(b, h, n, d):
    """The real check: o / l is attention, against the step-2 ground truth."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    o, _, l = running_output_accumulator(q, k, v, block_m=64, block_n=64)
    got = o / l[..., None]
    want = naive_attention(q.float(), k.float(), v.float())

    assert got.shape == (b, h, n, d)
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_unnormalized_accumulator_matches_reference():
    """Check the accumulator itself, not just the normalized result -- a bug in
    alpha could cancel out of o/l while leaving o and l both wrong."""
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    got_o, got_m, got_l = running_output_accumulator(q, k, v, block_m=64, block_n=64)
    want_o, want_m, want_l = _reference(q, k, v, 1.0 / math.sqrt(d))

    torch.testing.assert_close(got_m, want_m, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(got_l, want_l, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(got_o, want_o, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("spike_at", ["first", "last"])
def test_accumulator_rescaling_with_a_late_maximum(spike_at):
    """alpha must rescale the (BLOCK_M, HEAD_DIM) accumulator, not just l.

    Forgetting `acc = acc * alpha[:, None]` leaves l correct and o wrong, so
    this catches a bug the step-7 tests cannot see.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16).abs() + 0.5
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    idx = 0 if spike_at == "first" else n - 1
    k[:, :, idx, :] = 4.0

    o, _, l = running_output_accumulator(q, k, v, block_m=64, block_n=64)
    want = naive_attention(q.float(), k.float(), v.float())

    torch.testing.assert_close(o / l[..., None], want, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 64), (64, 32), (128, 64)])
def test_result_independent_of_block_sizes(block_m, block_n):
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 300, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    o, _, l = running_output_accumulator(q, k, v, block_m=block_m, block_n=block_n)
    want = naive_attention(q.float(), k.float(), v.float())

    torch.testing.assert_close(o / l[..., None], want, atol=ATOL, rtol=RTOL)


def test_output_is_a_convex_combination_of_values():
    """Attention output must lie within the range of V.

    Weights are non-negative and sum to 1, so every output coordinate is a
    weighted average of that coordinate across V. Anything outside those bounds
    means negative or unnormalized weights.
    """
    torch.manual_seed(0)
    b, h, n, d = 1, 2, 256, 64
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    o, _, l = running_output_accumulator(q, k, v)
    got = (o / l[..., None]).float()

    lo = v.float().amin(dim=2, keepdim=True)
    hi = v.float().amax(dim=2, keepdim=True)
    assert (got >= lo - ATOL).all() and (got <= hi + ATOL).all()


def test_padded_positions_do_not_corrupt_output():
    torch.manual_seed(0)
    b, h, n, d = 1, 1, 70, 32
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float16) for _ in range(3))

    o, _, l = running_output_accumulator(q, k, v, block_m=64, block_n=64)
    want = naive_attention(q.float(), k.float(), v.float())

    torch.testing.assert_close(o / l[..., None], want, atol=ATOL, rtol=RTOL)


def test_all_finite():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 100, 32, device="cuda", dtype=torch.float16) for _ in range(3))
    for tensor in running_output_accumulator(q, k, v):
        assert torch.isfinite(tensor).all()
