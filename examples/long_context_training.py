"""Training Qwen2.5-0.5B at context lengths the reference attention cannot reach.

Two experiments:

`sweep` runs a single training step (forward, backward, optimizer) at growing
context lengths for each attention implementation, and records where each one
runs out of memory. Eager attention materializes a 14 x N x N score matrix per
layer -- 1.9 GB per layer at N=8192, times 24 layers -- so it fails early. The
tiled kernel never allocates one.

`curves` trains for a fixed number of steps at a context both implementations
can handle, from the same seed on the same batches in the same order, and
records the loss at every step. Overlapping curves mean the kernel is correct
under optimization, not merely on a forward pass. Diverging curves would mean a
gradient is subtly wrong in a way the unit tests missed.

A caveat the sweep will make obvious: this model has a 151,936-token
vocabulary, so at long context the output logits dominate memory -- 4.7 GB at
N=16384 in bf16, and cross-entropy upcasts to fp32 for more. Once attention
stops being quadratic, the vocabulary projection becomes the next wall. That is
a real property of long-context training, not an artifact of this script, and
it is why production setups use chunked or fused cross-entropy.

Run:
    python examples/long_context_training.py sweep
    python examples/long_context_training.py curves --steps 60
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flash_attn.hf as hf

MODEL_ID = "Qwen/Qwen2.5-0.5B"
RESULTS = Path(__file__).parent.parent / "benchmarks" / "results"

IMPLEMENTATIONS = ["eager", "sdpa", hf.ATTENTION_NAME]
SWEEP_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]


@dataclass
class StepResult:
    impl: str
    seq_len: int
    peak_memory_mb: float | None
    step_time_ms: float | None
    loss: float | None
    status: str  # "ok" or "oom"


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def load_model(impl: str, train: bool = True):
    from transformers import AutoModelForCausalLM

    if impl == hf.ATTENTION_NAME:
        hf.register()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation=impl
    ).cuda()
    model.train(train)
    return model


def load_chunks(seq_len: int, n_chunks: int) -> torch.Tensor:
    """Fixed training batches, identical across implementations.

    Both runs must see the same tokens in the same order for the loss curves to
    be comparable; anything else and a divergence could be the data rather than
    the kernel.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]")
    ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0]

    need = n_chunks * seq_len
    if ids.numel() < need:
        raise SystemExit(f"need {need:,} tokens, dataset slice gave {ids.numel():,}")
    return ids[:need].view(n_chunks, seq_len)


def one_training_step(impl: str, seq_len: int) -> StepResult:
    """Forward, backward and an optimizer step at one context length."""
    common = dict(impl=impl, seq_len=seq_len)
    model = optimizer = None
    try:
        model = load_model(impl)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        ids = torch.randint(0, 150000, (1, seq_len), device="cuda")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()

        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3

        return StepResult(
            **common,
            peak_memory_mb=torch.cuda.max_memory_allocated() / 2**20,
            step_time_ms=elapsed_ms,
            loss=loss.item(),
            status="ok",
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_oom(exc):
            raise
        return StepResult(**common, peak_memory_mb=None, step_time_ms=None, loss=None, status="oom")
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_sweep() -> list[StepResult]:
    results: list[StepResult] = []
    exhausted: set[str] = set()

    for impl in IMPLEMENTATIONS:
        for seq_len in SWEEP_LENGTHS:
            if impl in exhausted:
                continue
            result = one_training_step(impl, seq_len)
            results.append(result)

            label = f"{impl:>13}  N={seq_len:<6}"
            if result.status == "oom":
                print(f"{label} OOM")
                exhausted.add(impl)
            else:
                print(
                    f"{label} {result.peak_memory_mb:9.1f} MiB  "
                    f"{result.step_time_ms:8.1f} ms  loss {result.loss:.4f}"
                )
    return results


def run_curves(impl: str, chunks: torch.Tensor, steps: int, lr: float) -> list[float]:
    """Train for `steps` steps and return the loss at each one.

    The seed is reset before the model loads so both implementations start from
    identical optimizer state and see identical batches.
    """
    torch.manual_seed(0)
    model = load_model(impl)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    hf.reset_call_count()

    losses = []
    for step in range(steps):
        ids = chunks[step % len(chunks)].unsqueeze(0).cuda()
        loss = model(ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            print(f"  {impl:>13}  step {step + 1}/{steps}  loss {loss.item():.4f}")

    if impl == hf.ATTENTION_NAME:
        assert hf.call_count > 0, "the Triton kernel never ran"

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["sweep", "curves"])
    parser.add_argument("--seq-len", type=int, default=1024, help="context for `curves`")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this needs a CUDA device")

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.experiment == "sweep":
        results = run_sweep()
        reached = {
            impl: max((r.seq_len for r in results if r.impl == impl and r.status == "ok"), default=0)
            for impl in IMPLEMENTATIONS
        }
        print("\nlongest context reached:")
        for impl, n in reached.items():
            print(f"  {impl:>13}  {n:,}")

        out = RESULTS / "long_context_sweep.json"
        out.write_text(
            json.dumps(
                {
                    "model": MODEL_ID,
                    "gpu": torch.cuda.get_device_name(0),
                    "results": [asdict(r) for r in results],
                    "longest_context_reached": reached,
                },
                indent=2,
            )
        )
        print(f"wrote {out}")
        return

    chunks = load_chunks(args.seq_len, n_chunks=min(args.steps, 32))
    curves = {impl: run_curves(impl, chunks, args.steps, args.lr) for impl in ("sdpa", hf.ATTENTION_NAME)}

    a, b = curves["sdpa"], curves[hf.ATTENTION_NAME]
    diffs = [abs(x - y) for x, y in zip(a, b)]
    print("\n--- loss curve agreement ---")
    print(f"first step   sdpa {a[0]:.5f}   ours {b[0]:.5f}   diff {diffs[0]:.6f}")
    print(f"final step   sdpa {a[-1]:.5f}   ours {b[-1]:.5f}   diff {diffs[-1]:.6f}")
    print(f"worst step-wise difference: {max(diffs):.6f}")
    print(f"loss reduction   sdpa {a[0] - a[-1]:.4f}   ours {b[0] - b[-1]:.4f}")

    # Also write a compact CSV. JSON with 120 floats is unreadable in a terminal,
    # and on an ephemeral machine the terminal is often the only way results get
    # off the box before it is destroyed.
    csv_path = RESULTS / "long_context_curves.csv"
    with csv_path.open("w") as f:
        f.write("step,sdpa,flash_triton\n")
        for i, (x, y) in enumerate(zip(a, b)):
            f.write(f"{i},{x:.6f},{y:.6f}\n")
    print(f"wrote {csv_path}")

    out = RESULTS / "long_context_curves.json"
    out.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "seq_len": args.seq_len,
                "steps": args.steps,
                "lr": args.lr,
                "gpu": torch.cuda.get_device_name(0),
                "curves": curves,
                "max_stepwise_difference": max(diffs),
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
