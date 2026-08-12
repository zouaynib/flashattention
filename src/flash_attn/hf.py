"""Use the Triton kernel as a HuggingFace `transformers` attention implementation.

Registers under the name "flash_triton", so a model picks it up with:

    from flash_attn.hf import register
    register()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, attn_implementation="flash_triton"
    )

Nothing else about the model changes. That is the whole point of the exercise:
the kernel has to survive contact with weights and shapes it was not written
against.

Grouped-query attention is where this pays off. `transformers`' own SDPA path
calls `repeat_kv` to expand K and V from `num_key_value_heads` up to
`num_attention_heads` before attending -- for Qwen2.5-0.5B that is 2 heads
becoming 14. This kernel reads them through the group mapping instead, so the
expansion never happens and the KV tensors stay 7x smaller at the attention
call.

Unsupported cases raise rather than silently falling back to SDPA. A quiet
fallback would make a parity experiment meaningless: the numbers would agree
because the same kernel ran twice.
"""

from __future__ import annotations

from typing import Any

import torch

from flash_attn.autograd import flash_attention

ATTENTION_NAME = "flash_triton"

# Incremented on every call, so an experiment can assert the kernel actually ran
# rather than trusting that registration took effect.
call_count = 0


def reset_call_count() -> None:
    global call_count
    call_count = 0


def flash_attention_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attention entry point matching the `transformers` interface.

    Args mirror `sdpa_attention_forward`. Query arrives as (B, H_q, N, D) and
    key/value as (B, H_kv, N, D), already rotated by RoPE. The return is
    (B, N, H_q, D) plus `None` for attention weights, which this kernel never
    materializes.
    """
    global call_count

    if position_bias is not None:
        raise NotImplementedError(
            "position_bias (T5-style relative attention) is not supported; it is an "
            "additive N x N term, which this kernel deliberately has no path for"
        )
    if dropout > 0.0:
        raise NotImplementedError(
            f"attention dropout is not implemented (got {dropout}); set "
            "attention_dropout=0.0 in the model config"
        )
    if attention_mask is not None:
        raise NotImplementedError(
            "arbitrary attention masks are not supported -- a dense mask is an N x N "
            "tensor and would reintroduce the quadratic memory this kernel avoids. "
            "Only causal and full attention are available, so batches must be free "
            "of padding (use batch_size=1 or pack sequences to equal length)"
        )

    # Mirror the reference implementation's causality logic exactly. A single
    # query token during decoding attends to the whole cache, so causal masking
    # is a no-op there and `transformers` turns it off.
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    q_length, kv_length = query.shape[2], key.shape[2]
    is_causal = bool(q_length > 1 and is_causal)

    # Prefill against an empty static cache leaves K/V longer than Q, padded with
    # unwritten entries. The reference slices them off; doing the same keeps
    # q_length == kv_length, which sidesteps the fact that this kernel aligns
    # causal masking bottom-right while SDPA aligns it upper-left. The two agree
    # on square inputs, so slicing removes the only case where they would not.
    if is_causal and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]

    # K and V keep their own (smaller) head count -- no repeat_kv.
    attn_output = flash_attention(query, key, value, causal=is_causal, sm_scale=scaling)

    call_count += 1
    return attn_output.transpose(1, 2).contiguous(), None


def register() -> None:
    """Add the kernel to the `transformers` attention registry.

    Import is deferred so that `transformers` is only a dependency of the
    integration examples, not of the kernel library itself.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[ATTENTION_NAME] = flash_attention_interface
