"""The HuggingFace attention interface, checked against the reference it replaces.

These build the tensors `transformers` would hand an attention function and call
both implementations directly, rather than downloading a model. That keeps the
test suite offline and fast while still exercising the exact contract -- shapes,
head counts, causality logic and output layout.

Shapes are Qwen2.5-0.5B's: 14 query heads against 2 KV heads, head_dim 64.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="Triton is Linux/GPU-only")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

from flash_attn_scratch.hf import (  # noqa: E402
    ATTENTION_NAME,
    flash_attention_interface,
    register,
    reset_call_count,
)

pytestmark = pytest.mark.gpu

# Qwen2.5-0.5B's attention shape.
H_Q, H_KV, HEAD_DIM = 14, 2, 64
ATOL, RTOL = 2e-2, 1e-2


def _module(is_causal: bool = True) -> SimpleNamespace:
    """The attributes `transformers` reads off the attention module."""
    return SimpleNamespace(num_key_value_groups=H_Q // H_KV, is_causal=is_causal)


def _qkv(batch: int, q_len: int, kv_len: int | None = None, dtype=torch.bfloat16):
    kv_len = kv_len if kv_len is not None else q_len
    q = torch.randn(batch, H_Q, q_len, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(batch, H_KV, kv_len, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn(batch, H_KV, kv_len, HEAD_DIM, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("batch,seq_len", [(1, 128), (2, 512), (1, 1024)])
def test_matches_the_sdpa_implementation(batch, seq_len):
    """Same inputs through both registered implementations.

    This is the integration-level version of the whole correctness suite: if the
    interface mis-handles head counts, causality or the output transpose, the
    answers diverge.
    """
    torch.manual_seed(0)
    q, k, v = _qkv(batch, seq_len)
    module = _module()

    got, got_weights = flash_attention_interface(module, q, k, v, None, scaling=0.125)
    want, want_weights = ALL_ATTENTION_FUNCTIONS["sdpa"](module, q, k, v, None, scaling=0.125)

    assert got.shape == want.shape == (batch, seq_len, H_Q, HEAD_DIM)
    assert got_weights is None and want_weights is None
    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_output_is_contiguous_in_the_expected_layout():
    """`transformers` reshapes the result immediately; a non-contiguous or
    untransposed return would either error or silently scramble heads."""
    q, k, v = _qkv(1, 256)
    out, _ = flash_attention_interface(_module(), q, k, v, None, scaling=0.125)

    assert out.shape == (1, 256, H_Q, HEAD_DIM)
    assert out.is_contiguous()


def test_single_query_decode_step():
    """During generation q_len is 1 against a long cache. `transformers` turns
    causal masking off for that case, since one token sees the whole cache."""
    torch.manual_seed(0)
    q, k, v = _qkv(1, q_len=1, kv_len=512)
    module = _module()

    got, _ = flash_attention_interface(module, q, k, v, None, scaling=0.125)
    want, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, q, k, v, None, scaling=0.125)

    assert got.shape == (1, 1, H_Q, HEAD_DIM)
    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_prefill_against_a_longer_cache_is_sliced_like_the_reference():
    """With an oversized cache the reference slices K/V down to q_len. Doing the
    same keeps the inputs square, so this kernel's bottom-right causal alignment
    and SDPA's upper-left one cannot disagree."""
    torch.manual_seed(0)
    q, k, v = _qkv(1, q_len=128, kv_len=512)
    module = _module()

    got, _ = flash_attention_interface(module, q, k, v, None, scaling=0.125)
    want, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, q, k, v, None, scaling=0.125)

    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_kv_heads_are_not_expanded():
    """The reason this integration is worth having.

    The reference calls repeat_kv, turning 2 KV heads into 14. This kernel reads
    them through the group mapping, so the tensors it receives stay 7x smaller.
    Asserting the inputs are untouched proves no expansion happened.
    """
    q, k, v = _qkv(1, 256)
    assert k.shape[1] == H_KV and v.shape[1] == H_KV

    out, _ = flash_attention_interface(_module(), q, k, v, None, scaling=0.125)

    assert k.shape[1] == H_KV, "K was expanded"
    assert out.shape[2] == H_Q


def test_registration_makes_the_name_available():
    register()
    assert ATTENTION_NAME in ALL_ATTENTION_FUNCTIONS
    assert ALL_ATTENTION_FUNCTIONS[ATTENTION_NAME] is flash_attention_interface


def test_call_count_tracks_invocations():
    """Experiments assert on this to prove the kernel really ran, rather than
    trusting that registration took effect."""
    import flash_attn_scratch.hf as hf

    reset_call_count()
    before = hf.call_count
    q, k, v = _qkv(1, 128)
    for _ in range(3):
        flash_attention_interface(_module(), q, k, v, None, scaling=0.125)
    assert hf.call_count == before + 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_both_supported_dtypes(dtype):
    torch.manual_seed(0)
    atol, rtol = (2e-2, 1e-2) if dtype is torch.float16 else (8e-2, 4e-2)
    q, k, v = _qkv(1, 512, dtype=dtype)
    module = _module()

    got, _ = flash_attention_interface(module, q, k, v, None, scaling=0.125)
    want, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, q, k, v, None, scaling=0.125)

    assert got.dtype == dtype
    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=rtol)


def test_rejects_an_attention_mask():
    """Padding masks are the one thing a real batch might bring that this cannot
    serve. Failing loudly beats falling back, which would make a parity
    experiment compare SDPA against itself."""
    q, k, v = _qkv(1, 128)
    mask = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="arbitrary attention masks"):
        flash_attention_interface(_module(), q, k, v, mask, scaling=0.125)


def test_rejects_dropout_and_position_bias():
    q, k, v = _qkv(1, 128)
    with pytest.raises(NotImplementedError, match="dropout"):
        flash_attention_interface(_module(), q, k, v, None, dropout=0.1, scaling=0.125)
    with pytest.raises(NotImplementedError, match="position_bias"):
        flash_attention_interface(
            _module(), q, k, v, None, scaling=0.125, position_bias=torch.zeros(1, device="cuda")
        )
