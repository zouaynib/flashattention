"""Does the Triton kernel change what a real pretrained model predicts?

Loads Qwen2.5-0.5B twice -- once with `transformers`' own SDPA attention, once
with this project's kernel -- and evaluates both on the same held-out text, in
the same order, with the same weights. Nothing differs but the attention
implementation.

This is a stronger correctness argument than the unit tests. Those check shapes
the author chose against a reference the author chose. This checks weights
trained by someone else, on text neither party picked, through 24 layers of a
model that was never aware the kernel exists. If a gradient scale were subtly
wrong or a head were mis-indexed, perplexity would move.

Three things are reported:

* perplexity for each implementation, and the gap between them
* the largest per-chunk loss disagreement, since a matching average can hide
  compensating errors in opposite directions
* peak memory for each, which is where grouped-query attention shows up:
  `transformers` expands 2 KV heads into 14 with `repeat_kv` before calling
  SDPA, and this kernel does not

Run:
    python examples/perplexity_parity.py                 # full test split
    python examples/perplexity_parity.py --limit 20      # quick check
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flash_attn_scratch.hf as hf

MODEL_ID = "Qwen/Qwen2.5-0.5B"
RESULTS = Path(__file__).parent.parent / "benchmarks" / "results"


@dataclass
class EvalResult:
    impl: str
    seq_len: int
    chunks: int
    tokens: int
    loss: float
    perplexity: float
    peak_memory_mb: float
    kernel_calls: int


def build_chunks(seq_len: int, limit: int | None) -> torch.Tensor:
    """Tokenize the WikiText-103 test split into equal-length blocks.

    Concatenating the whole split and cutting it into fixed blocks is the
    standard way to measure language-model perplexity: it avoids padding (which
    this kernel cannot mask anyway) and gives both implementations byte-identical
    inputs.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.numel() // seq_len
    if limit is not None:
        n_chunks = min(n_chunks, limit)

    chunks = ids[: n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"{ids.numel():,} tokens -> {n_chunks} chunks of {seq_len}")
    return chunks


def load_model(impl: str):
    from transformers import AutoModelForCausalLM

    if impl == hf.ATTENTION_NAME:
        hf.register()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation=impl
    )
    return model.cuda().eval()


@torch.no_grad()
def evaluate(impl: str, chunks: torch.Tensor) -> tuple[EvalResult, list[float]]:
    model = load_model(impl)
    hf.reset_call_count()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    losses: list[float] = []
    for i, chunk in enumerate(chunks):
        ids = chunk.unsqueeze(0).cuda()
        # `labels=ids` makes the model shift internally and return mean
        # cross-entropy over the seq_len - 1 predicted positions.
        loss = model(ids, labels=ids).loss
        losses.append(loss.item())
        if (i + 1) % 25 == 0:
            print(f"  {impl:>13}  {i + 1}/{len(chunks)} chunks", end="\r")

    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    mean_loss = sum(losses) / len(losses)

    result = EvalResult(
        impl=impl,
        seq_len=chunks.shape[1],
        chunks=len(chunks),
        tokens=chunks.numel(),
        loss=mean_loss,
        perplexity=float(torch.tensor(mean_loss).exp()),
        peak_memory_mb=peak_mb,
        kernel_calls=hf.call_count,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, losses


def logit_agreement(seq_len: int) -> float:
    """Largest absolute logit difference on one fixed batch.

    Sharper than perplexity: it compares the model's raw output rather than a
    quantity averaged over 150k vocabulary entries and thousands of positions.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, 150000, (1, seq_len), device="cuda")

    outputs = []
    for impl in ("sdpa", hf.ATTENTION_NAME):
        model = load_model(impl)
        with torch.no_grad():
            outputs.append(model(ids).logits.float().cpu())
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return (outputs[0] - outputs[1]).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="cap the number of chunks")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this needs a CUDA device")

    chunks = build_chunks(args.seq_len, args.limit)

    results, all_losses = [], {}
    for impl in ("sdpa", hf.ATTENTION_NAME):
        result, losses = evaluate(impl, chunks)
        results.append(result)
        all_losses[impl] = losses
        print(
            f"{impl:>13}  loss {result.loss:.6f}  ppl {result.perplexity:.4f}  "
            f"peak {result.peak_memory_mb:8.1f} MiB  kernel calls {result.kernel_calls}"
        )

    reference, ours = results
    assert ours.kernel_calls > 0, "the Triton kernel never ran -- registration failed"
    assert reference.kernel_calls == 0, "the reference run used the Triton kernel"

    per_chunk = [abs(a - b) for a, b in zip(all_losses["sdpa"], all_losses[hf.ATTENTION_NAME])]
    max_logit_diff = logit_agreement(args.seq_len)

    print("\n--- parity ---")
    print(f"perplexity      sdpa {reference.perplexity:.4f}   ours {ours.perplexity:.4f}")
    print(
        f"                gap  {abs(reference.perplexity - ours.perplexity):.6f}  "
        f"({100 * abs(reference.perplexity - ours.perplexity) / reference.perplexity:.4f}%)"
    )
    print(f"worst per-chunk loss difference : {max(per_chunk):.6f}")
    print(f"largest absolute logit difference: {max_logit_diff:.5f}")
    print(
        f"peak memory     sdpa {reference.peak_memory_mb:.1f} MiB   "
        f"ours {ours.peak_memory_mb:.1f} MiB   "
        f"({reference.peak_memory_mb / ours.peak_memory_mb:.2f}x)"
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "perplexity_parity.json"
    out.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "dataset": "wikitext-103-raw-v1 (test)",
                "gpu": torch.cuda.get_device_name(0),
                "results": [asdict(r) for r in results],
                "max_per_chunk_loss_difference": max(per_chunk),
                "max_absolute_logit_difference": max_logit_diff,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
